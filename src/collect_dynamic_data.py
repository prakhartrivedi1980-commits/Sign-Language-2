"""Collect labeled 30-frame ASL motion sequences with MediaPipe Tasks.

Example:
  python src/collect_dynamic_data.py --label OTHER --split train --samples 100 --session train1

Press S to start each sequence, Q to quit. Existing files are never overwritten.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
DATA_DIR = ROOT / "dataset" / "dynamic"
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 66


def main():
    parser = argparse.ArgumentParser(description="Record dynamic ASL landmarks")
    parser.add_argument("--label", choices=["J", "Z", "OTHER"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not args.session.isalnum():
        parser.error("--session must contain only letters and numbers")
    if not HAND_MODEL.is_file():
        raise FileNotFoundError(f"MediaPipe model missing: {HAND_MODEL}")

    output_dir = DATA_DIR / args.split / args.label / args.session
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = [int(p.stem) for p in output_dir.glob("*.npy") if p.stem.isdigit()]
    next_id = max(existing_ids, default=-1) + 1
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
        raise RuntimeError("Webcam unavailable; close other camera applications")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    recording = False
    sequence = []
    status = "Ready - press S to record"
    start = time.monotonic()
    last_timestamp = -1
    print(f"Recording {args.label} / {args.split} / {args.session}")
    print("S = start one 30-frame recording; Q = quit")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while count < args.samples:
                ok, frame = cap.read()
                if not ok:
                    print("Unable to read camera frame")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int((time.monotonic() - start) * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                hand_present = bool(result.hand_landmarks)
                if recording:
                    if not hand_present:
                        recording = False
                        sequence = []
                        status = "Hand lost: recording discarded; retry"
                        print(status)
                    else:
                        hand = result.hand_landmarks[0]
                        wrist = hand[0]
                        features = [coordinate for point in hand for coordinate in
                                    (point.x - wrist.x, point.y - wrist.y, point.z - wrist.z)]
                        features.extend([wrist.x, wrist.y, wrist.z])
                        sequence.append(features)
                        if len(sequence) == SEQUENCE_LENGTH:
                            arr = np.asarray(sequence, dtype=np.float32)
                            if arr.shape != (SEQUENCE_LENGTH, FEATURE_COUNT) or not np.isfinite(arr).all():
                                raise ValueError(f"Invalid sequence shape or values: {arr.shape}")
                            path = output_dir / f"{next_id:04d}.npy"
                            while path.exists():
                                next_id += 1
                                path = output_dir / f"{next_id:04d}.npy"
                            np.save(path, arr)
                            count += 1
                            next_id += 1
                            recording = False
                            sequence = []
                            status = f"Saved {count}/{args.samples}"
                            print(f"{status}: {path}")

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 155), (30, 30, 30), -1)
                lines = [f"Label: {args.label} | Split: {args.split}",
                         f"Saved this run: {count}/{args.samples}",
                         f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}", status]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (12, 32 + i * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                (0, 255, 0) if i == 0 else (255, 255, 255), 2)
                cv2.imshow("ASL Dynamic Dataset Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("s") and not recording:
                    if not hand_present:
                        status = "Show your hand before pressing S"
                    else:
                        sequence = []
                        recording = True
                        status = "RECORDING - perform gesture now"
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Finished. Saved {count} new sequences to {output_dir}")


if __name__ == "__main__":
    main()
