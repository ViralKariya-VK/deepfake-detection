import cv2
import numpy as np
import sys
import os
import mediapipe as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_FPS = 15
MIN_FRAMES = 30
MIN_CONFIDENCE = 0.7


mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = MIN_CONFIDENCE)


def _get_sample_indices(native_fps: float, total_frames: int) -> list[int]:
    step = native_fps / TARGET_FPS
    indices = np.arange(0, total_frames, step).astype(int)

    indices = indices[indices < total_frames]
    return indices.tolist()


def _detect_best_face(frame_rgb: np.ndarray) -> list[int] | None:
    h, w = frame_rgb.shape[:2]

    results = face_detector.process(frame_rgb)

    if not results.detections:
        return None
    
    best_box = None
    best_score = -1.0

    for detection in results.detections:
        score = detection.score[0]

        if score < MIN_CONFIDENCE:
            continue

        if score > best_score:
            best_score = score

            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            best_box = [x1, y1, x2, y2]

    return best_box


def _crop_face(frame_rgb: np.ndarray, box: list[int]) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box

    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.10)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame_rgb[y1:y2, x1:x2]


def process_video(video_path: str) -> tuple:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video_processor] ERROR: Cannot open video: {video_path}")
        return None, None

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0 or total_frames <= 0:
        print(f"[video_processor] ERROR: Invalid video metadata "
              f"(fps={native_fps}, frames={total_frames})")
        cap.release()
        return None, None

    print(f"[video_processor] Video opened : {os.path.basename(video_path)}")
    print(f"[video_processor] Native FPS   : {native_fps:.1f} | "
          f"Total frames: {total_frames}")

    sample_indices = _get_sample_indices(native_fps, total_frames)
    print(f"[video_processor] Sampling {len(sample_indices)} frames "
          f"at {TARGET_FPS}fps")

    face_crops     = []
    frames_no_face = 0

    for idx in sample_indices:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()

        if not ret:
            continue 

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        box = _detect_best_face(frame_rgb)
        if box is None:
            frames_no_face += 1
            continue

        crop = _crop_face(frame_rgb, box)

        if crop.size == 0:
            frames_no_face += 1
            continue

        face_crops.append(crop)

    cap.release()

    print(f"[video_processor] Frames with face : {len(face_crops)} | "
          f"Skipped (no face): {frames_no_face}")

    if len(face_crops) < MIN_FRAMES:
        print(f"[video_processor] ERROR: Not enough face frames "
              f"({len(face_crops)} < {MIN_FRAMES}). "
              f"Video too short or face not detected.")
        return None, None

    print(f"[video_processor] Done. Returning {len(face_crops)} "
          f"face crops at {TARGET_FPS}fps.")

    return face_crops, float(TARGET_FPS)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python video_processor.py path/to/video.mp4")
        sys.exit(1)

    test_path = sys.argv[1]
    crops, fps = process_video(test_path)

    if crops is not None:
        print(f"\n✓ Success: {len(crops)} face crops extracted")
        print(f"✓ FPS passed downstream : {fps}")
        print(f"✓ First crop shape      : {crops[0].shape}")  # (H, W, 3)
        print(f"✓ All crops have data   : {all(c.size > 0 for c in crops)}")
    else:
        print("\n✗ process_video returned None — check errors above")