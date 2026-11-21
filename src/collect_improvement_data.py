
import cv2
import csv
import time
import argparse
import mediapipe as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
OUTPUT_DIR = ROOT / "dataset" / "webcam_improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--label", choices=["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"], required=True)
parser.add_argument("--samples", type=int, default=100)
parser.add_argument("--session", required=True)
args = parser.parse_args()

if args.samples <= 0:
    parser.error("Samples must be greater than zero")

OUTPUT = OUTPUT_DIR / f"{args.session}_{args.label}.csv"

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Use a new session name."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

header = ["label", "session"]

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

count = 0
recording = False
last_save = 0

start_time = time.monotonic()
last_timestamp = -1

try:
    with open(OUTPUT, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        with HandLandmarker.create_from_options(options) as detector:

            while count < args.samples:
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )
                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp

                result = detector.detect_for_video(
                    mp_image, timestamp
                )

                if recording and result.hand_landmarks:
                    now = time.monotonic()

                    if now - last_save >= 0.20:
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
                            [args.label, args.session] + features
                        )

                        count += 1
                        last_save = now

                status = "RECORDING" if recording else "PAUSED"

                cv2.putText(
                    frame,
                    f"Letter: {args.label} | {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Samples: {count}/{args.samples}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

                cv2.putText(
                    frame,
                    "S: Start/Pause | Q: Quit",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2
                )

                cv2.imshow("ASL Improvement Data", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    recording = not recording

                elif key == ord("q"):
                    break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Saved {count} samples to {OUTPUT}")
