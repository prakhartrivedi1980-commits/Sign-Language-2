"""Hybrid ASL: static letters and gated J/Z text generation in one webcam window.

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
I_HOLD_SECONDS = 1.0  # allow time to distinguish I from moving J
UNCERTAIN_GRACE_SECONDS = 0.25
HAND_RELEASE_SECONDS = 0.35
DYNAMIC_WINDOW = 30
DYNAMIC_EVERY = 5
MOTION_WINDOW = 10
MOTION_THRESHOLD = 0.045  # normalized image coordinate distance; tune on webcam
DYNAMIC_ALERT_THRESHOLD = 0.82  # experimental; validate against OTHER movements
DYNAMIC_REQUIRED_VOTES = 3
DYNAMIC_SETTLE_SECONDS = 0.30
DYNAMIC_RESET_SECONDS = 0.65
DYNAMIC_COOLDOWN_SECONDS = 1.0



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

    dynamic_model_path = MODEL_DIR / "asl_jz_other_lstm.keras"
    dynamic_preprocess_path = MODEL_DIR / "asl_jz_other_preprocessing.npz"
    for path in (dynamic_model_path, dynamic_preprocess_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dynamic model file missing: {path}")
    dynamic_model = tf.keras.models.load_model(dynamic_model_path, compile=False)
    with np.load(dynamic_preprocess_path, allow_pickle=False) as saved:
        dynamic_mean = saved["mean"].astype(np.float32)
        dynamic_std = saved["std"].astype(np.float32)
        dynamic_classes = saved["classes"].astype(str)
        feature_version = str(saved["feature_version"].item()) if "feature_version" in saved else ""
    if feature_version != "wrist_xy_delta_v1":
        raise ValueError(f"Unexpected dynamic feature version: {feature_version}")
    if dynamic_mean.shape != (66,) or dynamic_std.shape != (66,):
        raise ValueError("Dynamic preprocessing must have 66 features")
    if set(dynamic_classes) != {"J", "Z", "OTHER"} or dynamic_model.output_shape[-1] != 3:
        raise ValueError("Expected J/Z/OTHER three-class LSTM")
    dynamic_std = np.where(dynamic_std < 1e-6, 1.0, dynamic_std)
    print("Dynamic LSTM loaded; gated J/Z automatic text generation enabled.")
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
    dynamic_history = deque(maxlen=DYNAMIC_WINDOW)
    motion_history = deque(maxlen=MOTION_WINDOW)
    frame_number = 0
    dynamic_prediction = "Collecting frames"
    dynamic_confidence = 0.0
    motion_detected = False
    dynamic_votes = deque(maxlen=5)
    gesture_active = False
    gesture_last_motion = None
    suppress_i_until = 0.0
    dynamic_locked = False
    dynamic_cooldown_until = 0.0
    static_resume_at = 0.0
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
                frame_number += 1
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
                    dynamic_frame = np.concatenate((raw_features, np.array([
                        wrist.x, wrist.y, wrist.z
                    ], dtype=np.float32)))
                    dynamic_history.append(dynamic_frame)
                    motion_history.append(np.array([
                        wrist.x, wrist.y, hand[8].x, hand[8].y
                    ], dtype=np.float32))
                    motion_detected = False
                    if len(motion_history) == MOTION_WINDOW:
                        start = motion_history[0]
                        end = motion_history[-1]
                        wrist_travel = np.linalg.norm(end[:2] - start[:2])
                        fingertip_travel = np.linalg.norm(end[2:] - start[2:])
                        motion_detected = max(wrist_travel, fingertip_travel) > MOTION_THRESHOLD

                    if motion_detected:
                        suppress_i_until = max(suppress_i_until, now + 1.2)
                        if not gesture_active and not dynamic_locked and now >= dynamic_cooldown_until:
                            gesture_active = True
                            dynamic_votes.clear()
                        if gesture_active:
                            gesture_last_motion = now
                        static_resume_at = max(static_resume_at, now + DYNAMIC_RESET_SECONDS)

                    if len(dynamic_history) == DYNAMIC_WINDOW and frame_number % DYNAMIC_EVERY == 0:
                        seq = np.stack(dynamic_history).astype(np.float32)
                        seq[:, 63:65] -= seq[:1, 63:65]
                        seq[:, 65] = 0.0
                        seq = (seq - dynamic_mean) / dynamic_std
                        probs = dynamic_model(seq[None, ...], training=False).numpy()[0]
                        dynamic_idx = int(np.argmax(probs))
                        dynamic_prediction = str(dynamic_classes[dynamic_idx])
                        dynamic_confidence = float(probs[dynamic_idx])
                        if gesture_active:
                            dynamic_votes.append(
                                dynamic_prediction if dynamic_prediction in ("J", "Z")
                                and dynamic_confidence >= DYNAMIC_ALERT_THRESHOLD else "OTHER"
                            )

                    # Commit at most once after the movement has settled.
                    if (gesture_active and gesture_last_motion is not None
                            and now - gesture_last_motion >= DYNAMIC_SETTLE_SECONDS):
                        counts = Counter(dynamic_votes)
                        letter, votes = counts.most_common(1)[0] if counts else ("OTHER", 0)
                        if letter in ("J", "Z") and votes >= DYNAMIC_REQUIRED_VOTES:
                            generated_text += letter
                            last_committed_letter = letter
                            last_commit_index = len(generated_text) - 1
                            last_commit_time = now
                            current_gesture_added = True
                            suppress_i_until = max(suppress_i_until, now + 1.5)
                            history.clear()
                            stable_letter = None
                            stable_since = None
                            status_message = f"Added dynamic {letter}"
                            status_until = now + 1.5
                            print("Generated text:", generated_text)
                        else:
                            status_message = "Movement rejected (not confirmed J/Z)"
                            status_until = now + 1.0
                        gesture_active = False
                        dynamic_locked = True
                        dynamic_cooldown_until = now + DYNAMIC_COOLDOWN_SECONDS
                        static_resume_at = max(static_resume_at, now + DYNAMIC_RESET_SECONDS)
                        dynamic_votes.clear()

                    # Re-arm only after a quiet interval; not on every LSTM window.
                    if (dynamic_locked and not motion_detected
                            and gesture_last_motion is not None
                            and now - gesture_last_motion >= DYNAMIC_RESET_SECONDS
                            and now >= dynamic_cooldown_until):
                        dynamic_locked = False

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
                    # Pause static auto-commit during significant movement.
                    # Block static letters during/shortly after dynamic motion.
                    if (motion_detected or gesture_active or now < static_resume_at
                            or (predicted_letter == "I" and now < suppress_i_until)):
                        stable_letter = None
                        stable_since = None
                    elif predicted_letter in classes:
                        if predicted_letter != stable_letter:
                            stable_letter = predicted_letter
                            stable_since = now
                        last_stable_seen = now
                        if (stable_since is not None
                                and now - stable_since >= (I_HOLD_SECONDS if predicted_letter == "I" else LETTER_HOLD_SECONDS)
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
                    dynamic_history.clear()
                    suppress_i_until = 0.0
                    motion_history.clear()
                    dynamic_prediction = "No Hand"
                    dynamic_confidence = 0.0
                    motion_detected = False
                    gesture_active = False
                    dynamic_locked = False
                    gesture_last_motion = None
                    dynamic_votes.clear()
                    static_resume_at = now + 0.2
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

                # Top panel: static and dynamic diagnostics in the same window
                cv2.rectangle(frame, (0, 0), (width, 195), (30, 30, 30), -1)
                cv2.putText(frame, f"Prediction: {predicted_letter}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (15, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Corrections: {correction_count} | FPS: {fps:.1f}", (15, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, f"LSTM: {dynamic_prediction} ({dynamic_confidence * 100:.1f}%) | Motion: {'YES' if motion_detected else 'no'}",
                            (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 170, 255) if motion_detected else (200, 200, 200), 2)
                cv2.putText(frame, "J/Z auto text ON | A-Y: Correct | Enter: Repeat | Tab: Clear | Esc",
                            (15, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

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

                if now < status_until:
                    cv2.putText(frame, status_message[:75], (15, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
                cv2.imshow("ASL Recognition - Human in the Loop", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed. Corrections saved this session:", correction_count)
        print("Correction dataset:", CORRECTION_FILE)


if __name__ == "__main__":
    main()
