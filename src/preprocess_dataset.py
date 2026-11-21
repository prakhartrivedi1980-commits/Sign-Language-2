
import argparse
import csv
import random
import cv2
import mediapipe as mp

from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

LETTERS = [
    c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if c not in "JZ"
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--samples",
    type=int,
    default=500,
    help="Maximum images per letter; 0 means all"
)

parser.add_argument(
    "--resume",
    action="store_true"
)

args = parser.parse_args()

if args.samples < 0:
    parser.error("--samples cannot be negative")

OUTPUT = DATASET_DIR / "asl_landmarks_full.csv"

def find_training_folder():
    candidates = [DATASET_DIR]

    candidates.extend(
        p for p in DATASET_DIR.rglob("*")
        if p.is_dir()
        and "test" not in p.parts[-1].lower()
    )

    for folder in candidates:
        if all(
            (folder / letter).is_dir()
            for letter in LETTERS
        ):
            return folder

    raise FileNotFoundError(
        "Cannot find the 24 ASL training folders."
    )

TRAIN_DIR = find_training_folder()

print("Training directory:", TRAIN_DIR)
print("Letters:", LETTERS)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

header = ["label", "source_file"]

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

# Read already processed filenames when resuming
processed = set()

if args.resume and OUTPUT.exists():
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != header:
            raise ValueError(
                "Existing CSV has incompatible columns."
            )

        for row in reader:
            processed.add(row["source_file"])

elif OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. "
        "Use --resume or move the old file first."
    )

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

counts = Counter()
failures = Counter()

mode = "a" if args.resume and OUTPUT.exists() else "w"

with open(
    OUTPUT, mode, newline="", encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if mode == "w":
        writer.writerow(header)

    with HandLandmarker.create_from_options(options) as detector:

        for letter in LETTERS:

            image_dir = TRAIN_DIR / letter

            images = sorted(
                p for p in image_dir.iterdir()
                if p.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            )

            # Deterministic image selection
            rng = random.Random(42)
            rng.shuffle(images)

            if args.samples > 0:
                images = images[:args.samples]

            print(f"\nProcessing {letter}: {len(images)} images")

            for index, image_path in enumerate(images, 1):

                source = str(
                    image_path.relative_to(DATASET_DIR)
                )

                if source in processed:
                    continue

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
                wrist = hand[0]

                features = []

                for point in hand:
                    features.extend([
                        point.x - wrist.x,
                        point.y - wrist.y,
                        point.z - wrist.z
                    ])

                writer.writerow(
                    [letter, source] + features
                )

                counts[letter] += 1

                if index % 100 == 0:
                    print(
                        f"{letter}: {index}/{len(images)} "
                        f"processed"
                    )
                    file.flush()

            file.flush()

            print(
                f"{letter}: {counts[letter]} new saved, "
                f"{failures[letter]} failed"
            )

print("\nPreprocessing finished!")
print("Output:", OUTPUT)
print("New successful detections:", sum(counts.values()))
print("New failed detections:", sum(failures.values()))
