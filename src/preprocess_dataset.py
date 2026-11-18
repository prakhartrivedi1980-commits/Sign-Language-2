
import cv2
import csv
import random
import mediapipe as mp

from pathlib import Path
from collections import Counter

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
OUTPUT = DATASET_DIR / "asl_landmarks_pilot.csv"

LETTERS = ["A", "B", "C", "D", "E"]
SAMPLES_PER_LETTER = 200
SEED = 42

random.seed(SEED)

# Find the folder containing alphabet directories
def find_training_folder():
    for folder in DATASET_DIR.rglob("*"):
        if folder.is_dir():
            if all((folder / letter).is_dir()
                   for letter in LETTERS):
                return folder

    raise FileNotFoundError(
        "Cannot find ASL training folders A, B, C, D, E"
    )

TRAIN_DIR = find_training_folder()
print("Training folder:", TRAIN_DIR)

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

# CSV column names
header = ["label", "source_file"]

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

counts = Counter()
failures = Counter()

with HandLandmarker.create_from_options(options) as detector:
    with open(OUTPUT, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for letter in LETTERS:
            image_dir = TRAIN_DIR / letter

            images = sorted(
                p for p in image_dir.iterdir()
                if p.suffix.lower() in
                {".jpg", ".jpeg", ".png"}
            )

            random.shuffle(images)
            images = images[:SAMPLES_PER_LETTER]

            print(f"\nProcessing letter {letter}...")

            for image_path in images:
                image = cv2.imread(str(image_path))

                if image is None:
                    failures[letter] += 1
                    continue

                rgb = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect(mp_image)

                if not result.hand_landmarks:
                    failures[letter] += 1
                    continue

                hand = result.hand_landmarks[0]

                # Normalize relative to wrist
                wrist = hand[0]
                features = []

                for point in hand:
                    features.extend([
                        point.x - wrist.x,
                        point.y - wrist.y,
                        point.z - wrist.z
                    ])

                writer.writerow([
                    letter,
                    str(image_path.relative_to(DATASET_DIR)),
                    *features
                ])

                counts[letter] += 1

            print(
                f"{letter}: {counts[letter]} saved, "
                f"{failures[letter]} failed"
            )

print("\nPreprocessing completed!")
print("Saved file:", OUTPUT)
print("Successful detections:", dict(counts))
print("Failed detections:", dict(failures))
