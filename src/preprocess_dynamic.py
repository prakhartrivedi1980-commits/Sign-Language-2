
import cv2
import csv
import json
import numpy as np
import mediapipe as mp

from pathlib import Path
from collections import Counter

# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "dataset" / "dynamic" / "raw" / "signnteam"
OUTPUT_DIR = ROOT / "dataset" / "dynamic" / "processed"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 66
MIN_DETECTION_RATE = 0.70

LABELS = ["J", "Z"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# MEDIAPIPE
# =====================================

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# EXTRACT FRAME FEATURES
# =====================================

def extract_features(hand):
    wrist = hand[0]
    features = []

    # 63 wrist-relative coordinates
    for point in hand:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])

    # Preserve wrist trajectory
    features.extend([
        wrist.x,
        wrist.y,
        wrist.z
    ])

    return np.asarray(features, dtype=np.float32)


# =====================================
# RESAMPLE TO 30 FRAMES
# =====================================

def resample_sequence(sequence, target_length=30):
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) < 2:
        return None

    old_positions = np.linspace(0, 1, len(sequence))
    new_positions = np.linspace(0, 1, target_length)

    output = np.zeros(
        (target_length, sequence.shape[1]),
        dtype=np.float32
    )

    for feature in range(sequence.shape[1]):
        output[:, feature] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature]
        )

    return output


# =====================================
# PROCESS ONE VIDEO
# =====================================

def process_video(video_path, detector,timestamp_state):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, 0.0, "cannot_open"

    sequence = []
    detected = []
    total_frames = 0
    timestamp = -1

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            total_frames += 1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_state[0] += max(1, round(1000 / fps))
            timestamp = timestamp_state[0]

            result = detector.detect_for_video(
                image,
                timestamp
            )

            if result.hand_landmarks:
                features = extract_features(
                    result.hand_landmarks[0]
                )
                sequence.append(features)
                detected.append(total_frames - 1)

    finally:
        cap.release()

    if total_frames < 2:
        return None, 0.0, "too_short"

    detection_rate = len(detected) / total_frames

    if detection_rate < MIN_DETECTION_RATE:
        return None, detection_rate, "low_detection"

    if len(sequence) < 2:
        return None, detection_rate, "too_few_landmarks"

    # Interpolate missing detections over original
    # frame positions, retaining approximate timing.
    detected = np.asarray(detected)
    sequence = np.asarray(sequence, dtype=np.float32)

    full_sequence = np.zeros(
        (total_frames, FEATURE_COUNT),
        dtype=np.float32
    )

    for feature in range(FEATURE_COUNT):
        full_sequence[:, feature] = np.interp(
            np.arange(total_frames),
            detected,
            sequence[:, feature]
        )

    processed = resample_sequence(
        full_sequence,
        SEQUENCE_LENGTH
    )

    if processed is None:
        return None, detection_rate, "resample_failed"

    if not np.isfinite(processed).all():
        return None, detection_rate, "invalid_values"

    return processed, detection_rate, "success"


# =====================================
# MAIN PROCESSING
# =====================================

def main():
    all_sequences = []
    all_labels = []
    metadata = []
    statistics = Counter()
    # Shared timestamp across all videos
    timestamp_state = [-1]
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        for label in LABELS:
            folder = RAW_DIR / label
            videos = sorted(folder.glob("*.avi"))

            print(f"\nProcessing {label}: {len(videos)} videos")

            for index, video in enumerate(videos, 1):
                sequence, rate, status = process_video(
                    video,
                    detector,
                    timestamp_state
                )

                statistics[f"{label}_{status}"] += 1

                if sequence is not None:
                    all_sequences.append(sequence)
                    all_labels.append(label)

                metadata.append({
                    "file": str(video.relative_to(RAW_DIR)),
                    "label": label,
                    "detection_rate": round(rate, 4),
                    "status": status
                })

                if index % 25 == 0:
                    print(
                        f"{label}: {index}/{len(videos)} "
                        f"processed"
                    )

    if not all_sequences:
        raise RuntimeError(
            "No usable sequences were extracted."
        )

    X = np.stack(all_sequences).astype(np.float32)
    y = np.asarray(all_labels, dtype=str)

    np.save(OUTPUT_DIR / "X_jz.npy", X)
    np.save(OUTPUT_DIR / "y_jz.npy", y)

    with open(
        OUTPUT_DIR / "video_metadata.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "label",
                "detection_rate",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(metadata)

    with open(
        OUTPUT_DIR / "extraction_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(dict(statistics), file, indent=2)

    print("\n================================")
    print("DYNAMIC PREPROCESSING COMPLETED")
    print("================================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    for label in LABELS:
        print(label, ":", np.sum(y == label))

    print("Statistics:", dict(statistics))
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
