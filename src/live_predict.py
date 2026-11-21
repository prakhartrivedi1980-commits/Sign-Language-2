"""Real-time ASL recognition, continuous text composition and human corrections.

Run from project root:
    python src/live_predict.py --model finetuned

Controls: A-Z correct labels, Enter repeat, Space add space, Backspace delete,
Tab clear text, Esc quit. Only supported model letters can be corrected.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
CORRECTION_FILE = ROOT / "dataset" / "corrections" / "confirmed_corrections.csv"

CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_FRAMES = 7
MIN_STABLE_VOTES = 5
LETTER_HOLD_SECONDS = 0.35
UNCERTAIN_GRACE_SECONDS = 0.25
HAND_RELEASE_SECONDS = 0.35


def save_correction(correct_label, predicted_label, confidence, features, session_id):
    CORRECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "session", "timestamp", "predicted_label", "correct_label", "confidence"]
    for i in range(21):
        header.extend((f"x{i}", f"y{i}", f"z{i}"))
    is_new = not CORRECTION_FILE.exists() or CORRECTION_FILE.stat().st_size == 0
    with CORRECTION_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(header)
        writer.writerow([
            str(uuid.uuid4()), session_id, datetime.now(timezone.utc).isoformat(),
            predicted_label, correct_label, confidence, *features.tolist()
        ])


def main():
    parser = argparse.ArgumentParser(description="ASL live recognition with human feedback")
    model_files = {
        "original": ("asl_pilot.keras", "asl_preprocessing.npz"),
        "candidate": ("asl_candidate.keras", "asl_candidate_preprocessing.npz"),
        "full24": ("asl_24_candidate.keras", "asl_24_candidate_preprocessing.npz"),
        "improved": ("asl_24_improved.keras", "asl_24_improved_preprocessing.npz"),
        "finetuned": (
            "asl_24_finetuned_candidate.keras",
            "asl_24_finetuned_candidate_preprocessing.npz",
        ),
    }
    parser.add_argument(
        "--model",
        choices=list(model_files),
        default="finetuned",
        help="Choose the trained ASL model (default: finetuned)",
    )
    args = parser.parse_args()
    model_filename, preprocessing_filename = model_files[args.model]
    model_path = MODEL_DIR / model_filename
    preprocessing_path = MODEL_DIR / preprocessing_filename

    for path in (HAND_MODEL, model_path, preprocessing_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Loading {args.model} ASL model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    # The original preprocessing archive may contain an object-typed class array.
    # Only load locally generated, trusted NPZ files with allow_pickle=True.
    with np.load(preprocessing_path, allow_pickle=True) as saved:
        mean = saved["mean"].astype(np.float32)
        std = saved["std"].astype(np.float32)
        classes = saved["classes"].astype(str)
    if mean.shape != (63,) or std.shape != (63,):
        raise ValueError("Preprocessing arrays must each have 63 features")
    if len(classes) != model.output_shape[-1]:
        raise ValueError("Class labels do not match model output size")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid normalization parameters")
    std = np.where(std < 1e-6, 1.0, std)
    supported_keys = {ord(letter.lower()): letter for letter in classes if len(letter) == 1}
    print("Supported letters:", ", ".join(classes))
    print("Keys: Supported A-Z letters correct labels | Space insert space | Backspace delete")
    print("Enter: repeat | Tab: clear text | Esc: quit")
    print("Switch directly between different signs. For repeats, release briefly or press Enter.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions or other camera apps.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    history = deque(maxlen=SMOOTHING_FRAMES)
    generated_text = ""
    stable_letter = None
    stable_since = None
    last_stable_seen = None
    last_committed_letter = None
    current_gesture_added = False
    last_commit_index = None
    last_commit_time = 0.0
    no_hand_since = None
    correction_count = 0
    status_message = "Show a sign to begin"
    status_until = 0.0
    start_time = time.monotonic()
    last_timestamp = -1
    previous_frame_time = time.perf_counter()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Could not read webcam frame")
                    break
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int((time.monotonic() - start_time) * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)

                now = time.monotonic()
                predicted_letter = "No Hand"
                raw_prediction = None
                confidence = 0.0
                raw_features = None

                if result.hand_landmarks:
                    no_hand_since = None
                    hand = result.hand_landmarks[0]
                    wrist = hand[0]
                    raw_features = np.array([
                        coordinate
                        for point in hand
                        for coordinate in (point.x - wrist.x, point.y - wrist.y, point.z - wrist.z)
                    ], dtype=np.float32)
                    features = ((raw_features - mean) / std).reshape(1, 63)
                    probabilities = model(features, training=False).numpy()[0]
                    index = int(np.argmax(probabilities))
                    raw_prediction = str(classes[index])
                    confidence = float(probabilities[index])

                    # Keep a rolling window, including uncertain frames, instead
                    # of clearing all progress on one low-confidence prediction.
                    history.append(raw_prediction if confidence >= CONFIDENCE_THRESHOLD else None)
                    valid = [label for label in history if label is not None]
                    if valid:
                        candidate, votes = Counter(valid).most_common(1)[0]
                        if votes >= MIN_STABLE_VOTES:
                            predicted_letter = candidate
                        else:
                            predicted_letter = "Uncertain"
                    else:
                        predicted_letter = "Uncertain"

                    # Continuous text: each NEW stable letter may be appended,
                    # even without removing the hand. A held letter is appended once.
                    if predicted_letter in classes:
                        if predicted_letter != stable_letter:
                            stable_letter = predicted_letter
                            stable_since = now
                        last_stable_seen = now
                        if (stable_since is not None
                                and now - stable_since >= LETTER_HOLD_SECONDS
                                and predicted_letter != last_committed_letter
                                and now - last_commit_time >= 0.25):
                            generated_text += predicted_letter
                            last_committed_letter = predicted_letter
                            last_commit_index = len(generated_text) - 1
                            last_commit_time = now
                            current_gesture_added = True
                            status_message = f"Added {predicted_letter}"
                            status_until = now + 1.2
                            print("Generated text:", generated_text)
                    elif (last_stable_seen is None or
                          now - last_stable_seen > UNCERTAIN_GRACE_SECONDS):
                        stable_letter = None
                        stable_since = None

                    for point in hand:
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                else:
                    history.clear()
                    stable_letter = None
                    stable_since = None
                    last_stable_seen = None
                    if no_hand_since is None:
                        no_hand_since = now
                    elif now - no_hand_since >= HAND_RELEASE_SECONDS:
                        last_committed_letter = None
                        current_gesture_added = False
                        last_commit_index = None

                # Controls are handled before rendering so text changes appear immediately.
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Escape: do not conflict with letter Q
                    break
                elif key == 32:  # Space
                    generated_text += " "
                    last_committed_letter = None
                    last_commit_index = None
                    current_gesture_added = False
                    status_message = "Space added"
                    status_until = now + 1.5
                elif key in (8, 127):  # Backspace
                    generated_text = generated_text[:-1]
                    last_committed_letter = None
                    last_commit_index = None
                    current_gesture_added = False
                    status_message = "Last character removed"
                    status_until = now + 1.5
                elif key == 9:  # Tab: do not conflict with letter X
                    generated_text = ""
                    last_committed_letter = None
                    last_commit_index = None
                    current_gesture_added = False
                    status_message = "Text cleared"
                    status_until = now + 1.5
                elif key in (10, 13):  # Enter: do not conflict with letter R
                    # Explicitly repeat the currently held sign once.
                    if predicted_letter in classes and confidence >= CONFIDENCE_THRESHOLD:
                        generated_text += predicted_letter
                        last_committed_letter = predicted_letter
                        last_commit_index = len(generated_text) - 1
                        last_commit_time = now
                        current_gesture_added = True
                        status_message = f"Repeated {predicted_letter}"
                        status_until = now + 1.5
                elif key in supported_keys:
                    correct_letter = supported_keys[key]
                    if raw_features is None:
                        status_message = "No hand: correction not saved"
                    elif raw_prediction == correct_letter:
                        status_message = f"{correct_letter} already predicted correctly"
                    else:
                        save_correction(correct_letter, raw_prediction, confidence, raw_features, session_id)
                        correction_count += 1
                        # Only replace a letter just appended from this gesture.
                        if (current_gesture_added and last_commit_index is not None
                                and last_commit_index < len(generated_text)
                                and now - last_commit_time < 2.5):
                            generated_text = (generated_text[:last_commit_index]
                                              + correct_letter
                                              + generated_text[last_commit_index + 1:])
                        else:
                            generated_text += correct_letter
                            last_commit_index = len(generated_text) - 1
                        history.clear()
                        history.append(correct_letter)
                        predicted_letter = correct_letter
                        stable_letter = correct_letter
                        stable_since = now
                        last_stable_seen = now
                        last_committed_letter = correct_letter
                        last_commit_time = now
                        current_gesture_added = True
                        status_message = f"Correction saved: {raw_prediction} -> {correct_letter}"
                        print(status_message)
                    status_until = now + 1.5

                current_frame_time = time.perf_counter()
                fps = 1.0 / max(current_frame_time - previous_frame_time, 1e-6)
                previous_frame_time = current_frame_time

                # Top panel
                cv2.rectangle(frame, (0, 0), (width, 155), (30, 30, 30), -1)
                cv2.putText(frame, f"Prediction: {predicted_letter}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (15, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Corrections: {correction_count} | FPS: {fps:.1f}", (15, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, "A-Z: Correct | Enter: Repeat | Space | Backspace | Tab: Clear | Esc",
                            (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

                # Bottom panel
                cv2.rectangle(frame, (0, height - 112), (width, height), (30, 30, 30), -1)
                cv2.putText(frame, "Generated text:", (15, height - 83),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, generated_text[-32:] or "_", (15, height - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if stable_since is not None:
                    elapsed = min(now - stable_since, LETTER_HOLD_SECONDS)
                    diagnostic = f"Stable: {stable_letter} | Hold: {elapsed:.2f}/{LETTER_HOLD_SECONDS:.2f}s | Last: {last_committed_letter}"
                else:
                    diagnostic = f"Stable: - | Hold: 0.00/{LETTER_HOLD_SECONDS:.2f}s | Last: {last_committed_letter}"
                cv2.putText(frame, diagnostic, (15, height - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

                cv2.imshow("ASL Recognition - Human in the Loop", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed. Corrections saved this session:", correction_count)
        print("Correction dataset:", CORRECTION_FILE)


if __name__ == "__main__":
    main()
