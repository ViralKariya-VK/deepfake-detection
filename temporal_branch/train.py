import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processor import process_video
from roi_extractor import extract_roi_signals
from pos_algorithm import extract_pulse_signal
from signal_processing import process_signal
from vppg_scorer import extract_features, train_ocsvm

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_real.csv")

FEATURE_NAMES = ["signal_power", "snr", "peak_regularity", "freq_entropy", "heart_rate", "pulse_amplitude"]


def _extract_one(video_path: str) -> np.ndarray | None:
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull

    try:
        face_crops, fps = process_video(video_path)
        if face_crops is None:
            return None
    
        rgb_signals, fps = extract_roi_signals(face_crops, fps)
        if rgb_signals is None:
            return None
        
        pulse_signal, fps = extract_pulse_signal(rgb_signals, fps)
        if pulse_signal is None:
            return None
        
        filtered, freqs, magnitudes, fps = process_signal(pulse_signal, fps)
        if filtered is None:
            return None
        
        feature_vector = extract_features(filtered, freqs, magnitudes, fps)
        return feature_vector
    
    except Exception as e:
        tqdm.write(f"[train] ERROR: {e}")
        return None

    finally:
        sys.stdout = old_stdout
        devnull.close()


def train(real_folder: str) -> None:
    video_paths = sorted([
        os.path.join(real_folder, f)
        for f in os.listdir(real_folder)
        if f.endswith(".mp4")
    ])

    if len(video_paths) == 0:
        print(f"[train] ERROR: No .mp4 files found in {real_folder}")
        sys.exit(1)
 
    print(f"[train] Found {len(video_paths)} videos in {real_folder}")
 
    # ── Check if CSV already exists (resume support) ──
    if os.path.exists(CSV_PATH):
        existing_df    = pd.read_csv(CSV_PATH)
        processed_names = set(existing_df["filename"].tolist())
        feature_rows   = existing_df[FEATURE_NAMES].values.tolist()
        print(f"[train] Resuming — {len(feature_rows)} videos already processed")
    else:
        processed_names = set()
        feature_rows    = []
 
    # ── Loop videos with progress bar ──
    failed  = 0
    skipped = 0
 
    with tqdm(
        total=len(video_paths),
        desc="Extracting features",
        unit="video",
        initial=len(processed_names)
    ) as pbar:
 
        for video_path in video_paths:
            filename = os.path.basename(video_path)
 
            # Skip already processed
            if filename in processed_names:
                skipped += 1
                pbar.update(1)
                continue
 
            fv = _extract_one(video_path)
 
            if fv is None:
                failed += 1
                tqdm.write(f"[train] SKIP: {filename}")
            else:
                feature_rows.append(fv.tolist())
                
                df = pd.DataFrame(
                    feature_rows,
                    columns=FEATURE_NAMES
                )
                
                all_names = list(processed_names) + [
                    os.path.basename(video_paths[i])
                    for i in range(len(processed_names),
                                   len(processed_names) + len(feature_rows) - skipped)
                ]
                df.insert(0, "filename", all_names[:len(feature_rows)])
                df.to_csv(CSV_PATH, index=False)
 
            pbar.update(1)
            pbar.set_postfix({"success": len(feature_rows), "failed": failed})
 
 
    print(f"\n[train] Feature extraction complete")
    print(f"[train] Success : {len(feature_rows)} | Failed : {failed}")
 
    if len(feature_rows) < 10:
        print(f"[train] ERROR: Too few valid videos ({len(feature_rows)}) to train.")
        sys.exit(1)
 
    # ── Train One-Class SVM ──
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    print(f"[train] Training One-Class SVM on feature matrix "
          f"{feature_matrix.shape}...")
    train_ocsvm(feature_matrix)
    print(f"[train] Done. Model saved to ocsvm_temporal.pkl")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py path/to/Celeb-real/")
        sys.exit(1)
 
    train(sys.argv[1])