
import cv2
import mediapipe as mp
from pathlib import Path

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model file path
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

# Configure hand detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam started. Press Q to exit.")

# Hand landmark connections
connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

with HandLandmarker.create_from_options(options) as detector:

    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot read webcam frame")
            break

        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand landmarks
        result = detector.detect(mp_image)

        height, width, _ = frame.shape

        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                # Draw landmark connections
                for connection in connections:

                    start = hand[connection.start]
                    end = hand[connection.end]

                    x1 = int(start.x * width)
                    y1 = int(start.y * height)
                    x2 = int(end.x * width)
                    y2 = int(end.y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # Draw 21 landmarks
                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 0, 255),
                        -1
                    )

            status = "Hand Detected"
            color = (0, 255, 0)

        else:
            status = "No Hand Detected"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow("ASL Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
