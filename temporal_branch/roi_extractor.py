import cv2
import numpy as np
import sys
import os
import mediapipe as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_VALID_FRAMES = 30

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = 1,
    refine_landmarks = False,
    min_detection_confidence = 0.5  
)


FOREHEAD_LANDMARKS    = [10, 338, 297, 332, 103, 67, 109]
LEFT_CHEEK_LANDMARKS  = [36, 31, 50, 187, 207, 206, 205, 137, 123, 116, 111, 117]
RIGHT_CHEEK_LANDMARKS = [266, 261, 280, 411, 427, 426, 425, 366, 352, 345, 340, 346]


def _landmarks_to_rect(
    landmark_indices: list[int],
    landmarks,
    frame_h: int,
    frame_w: int
) -> list[int]:
    
    xs = [int(landmarks[i].x * frame_w) for i in landmark_indices]
    ys = [int(landmarks[i].y * frame_h) for i in landmark_indices]

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    pad_x = max(1, int((x2 - x1) * 0.05))
    pad_y = max(1, int((y2 - y1) * 0.05))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    return [x1, y1, x2, y2]


def _mean_rgb(crop: np.ndarray, rect: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = rect

    region = crop[y1:y2, x1:x2]

    if region.size == 0:
        return None
    
    return np.mean(region, axis =(0, 1))


def _save_debug_frame(
    crop: np.ndarray,
    forehead_rect: list[int],
    left_rect: list[int],
    right_rect: list[int],
    frame_idx: int,
    debug_dir: str      
) -> None:
    
    os.makedirs(debug_dir, exist_ok=True)

    vis = cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR)

    for rect, color in [
        (forehead_rect, (0, 255, 0)),
        (left_rect, (255, 0, 0)),
        (right_rect, (0, 0, 255))
    ]:
    
        x1, y1, x2, y2 = rect
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    
    cv2.imwrite(os.path.join(debug_dir, f"{frame_idx:04d}.jpg"), vis)


def extract_roi_signals(
    face_crops: list,
    fps: float,
    debug: bool = False
) -> tuple:
    
    rgb_signals = []
    failed_frames = 0
    debug_dir = os.path.join(os.getcwd(), "debug_roi")
    debug_saved = 0
    DEBUG_MAX = 5

    for i, crop in enumerate(face_crops):
        h, w = crop.shape[:2]
        results = face_mesh.process(crop)

        if not results.multi_face_landmarks:
            failed_frames += 1
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        forehead_rect = _landmarks_to_rect(FOREHEAD_LANDMARKS,    landmarks, h, w)
        left_rect     = _landmarks_to_rect(LEFT_CHEEK_LANDMARKS,  landmarks, h, w)
        right_rect    = _landmarks_to_rect(RIGHT_CHEEK_LANDMARKS, landmarks, h, w)

        # Mean RGB per region
        forehead_rgb = _mean_rgb(crop, forehead_rect)
        left_rgb     = _mean_rgb(crop, left_rect)
        right_rgb    = _mean_rgb(crop, right_rect)

        if forehead_rgb is None or left_rgb is None or right_rgb is None:
            failed_frames += 1
            continue

        rgb_signals.append(np.mean([forehead_rgb, left_rgb, right_rgb], axis=0))

        if debug and debug_saved < DEBUG_MAX:
            _save_debug_frame(
                crop, forehead_rect, left_rect, right_rect,
                i, debug_dir
            )
            debug_saved += 1

    print(f"[roi_extractor] Valid frames={len(rgb_signals)} | "
          f"Failed={failed_frames}")
    
    if len(rgb_signals) < MIN_VALID_FRAMES:
        print(f"[roi_extractor] ERROR: Not enough valid frames "
              f"({len(rgb_signals)} < {MIN_VALID_FRAMES})")
        return None, None

    return np.array(rgb_signals), fps

