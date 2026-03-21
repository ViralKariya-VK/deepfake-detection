import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The POS algorithm works best when applied over a short sliding window
# rather than the entire signal at once

WINDOW_SECONDS = 1.6
OVERLAP = 0.5


def _pos_window(window: np.ndarray) -> np.ndarray:
    epsilon = 1e-6

    # Step 1 — Normalize by channel mean
    mean = np.mean(window, axis=0)                    # shape (3,)
    C_n  = window / (mean + epsilon)                  # shape (W, 3)

    R_n = C_n[:, 0]
    G_n = C_n[:, 1]
    B_n = C_n[:, 2]

   
    S1 = R_n - G_n
    S2 = R_n + G_n - 2.0 * B_n

    alpha = np.std(S1) / (np.std(S2) + epsilon)
    pulse = S1 + alpha * S2
    pulse = pulse - np.mean(pulse)
    

    return pulse


def _overlap_add(
        segments: list[np.ndarray],
        step    : int,
        n_frames: int
) -> np.ndarray:
    
    accumulator = np.zeros(n_frames)
    count       = np.zeros(n_frames)

    for i, seg in enumerate(segments):
        start = i * step
        end = start + len(seg)

        if end > n_frames:
            seg = seg[:n_frames - start]
            end = n_frames

        accumulator[start:end] += seg
        count[start:end] += 1.0

    count[count == 0] = 1.0

    return accumulator / count


def extract_pulse_signal(
    rgb_signals : np.ndarray,
    fps         : float
) -> tuple:
    
    n_frames = len(rgb_signals)
    window_len = int(WINDOW_SECONDS * fps)
    step = int(window_len * OVERLAP)

    if n_frames < window_len:
        print(f"[pos_algorithm] WARNING: Signal ({n_frames} frames) shorter "
              f"than window ({window_len} frames). Processing as single window.")
        pulse = _pos_window(rgb_signals)
        pulse = pulse - np.mean(pulse)
       
        return pulse, fps
    
    
    segments = []
    start = 0

    while start < n_frames:
        end = min(start + window_len, n_frames)
        window = rgb_signals[start:end]

        if len(window) < 4:
            break

        seg = _pos_window(window)
        segments.append(seg)
        start += step

    pulse = _overlap_add(segments, step, n_frames)
    pulse = pulse - np.mean(pulse)

    if np.std(pulse) < 1e-6:
        print("[pos_algorithm] ERROR: Pulse signal is flat. "
              "No heartbeat information captured.")
        return None, None

    return pulse, fps

