import cv2
import os
from tqdm import tqdm

DATASET_PATH = "datasets/Celeb-DF-v2/Celeb-real"
OUTPUT_PATH = "outputs/faces"

TARGET_FPS = 10  
FACE_SIZE = 224

os.makedirs(OUTPUT_PATH, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces(video_path):

    video_name = os.path.basename(video_path).split(".")[0]
    save_dir = os.path.join(OUTPUT_PATH, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(int(video_fps / TARGET_FPS), 1)

    frame_count = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            margin = int(0.2 * h)

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)

            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

            save_path = os.path.join(save_dir, f"{saved_count}.jpg")

            cv2.imwrite(save_path, face)

            saved_count += 1

        frame_count += 1

    cap.release()


def process_dataset():

    videos = [v for v in os.listdir(DATASET_PATH) if v.endswith(".mp4")]

    for video in tqdm(videos):

        video_path = os.path.join(DATASET_PATH, video)

        extract_faces(video_path)


if __name__ == "__main__":
    process_dataset()