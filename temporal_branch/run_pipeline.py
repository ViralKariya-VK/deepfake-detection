import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processor import process_video
from roi_extractor import extract_roi_signals
from pos_algorithm import extract_pulse_signal
from signal_processing import process_signal
from vppg_scorer import get_temporal_score


def run_temporal_pipeline(video_path: str) -> float | None:

    print(f"\n{'='*55}")
    print(f"TEMPORAL PIPELINE: {os.path.basename(video_path)}")
    print(f"{'='*55}")
 
    # Stage 1: Extract face crops
    face_crops, fps = process_video(video_path)
    if face_crops is None:
        print("[run_pipeline] FAILED at stage 1: video_processor")
        return None
 
    # Stage 2: Extract ROI signals 
    rgb_signals, fps = extract_roi_signals(face_crops, fps)
    if rgb_signals is None:
        print("[run_pipeline] FAILED at stage 2: roi_extractor")
        return None
 
    # Stage 3: Extract pulse signal 
    pulse_signal, fps = extract_pulse_signal(rgb_signals, fps)
    if pulse_signal is None:
        print("[run_pipeline] FAILED at stage 3: pos_algorithm")
        return None
 
    # Stage 4: Filter and FFT
    filtered, freqs, magnitudes, fps = process_signal(pulse_signal, fps)
    if filtered is None:
        print("[run_pipeline] FAILED at stage 4: signal_processing")
        return None
    

    temporal_score = get_temporal_score(filtered, freqs, magnitudes, fps)
    if temporal_score is None:
        print("[run_pipeline] FAILED at stage 5: vppg_scorer")
        print("[run_pipeline] Have you trained the model? Run train.py first.")
        return None
    

    print(f"\n{'='*55}")
    print(f"temporal_score : {temporal_score:.4f}")
    print(f"{'='*55}\n")
 
    return temporal_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py path/to/video.mp4")
        sys.exit(1)

    score = run_temporal_pipeline(sys.argv[1])

    if score is None:
        print("x Pipeline failed - check errors above")
        sys.exit(1)
    else:
        print(f"✓ temporal_score : {score:.4f}")