import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The POS algorithm works best when applied over a short sliding window
# rather than the entire signal at once

WINDOWS_SECONDS = 1.6


def _pos_window(window: np.ndarray) -> np.ndarray:
    epsilon = 1e-6

    # Step 1 — Normalize by channel mean
    mean = np.mean(window, axis=0)                    # shape (3,)
    C_n  = window / (mean + epsilon)                  # shape (W, 3)

    R_n = C_n[:, 0]
    G_n = C_n[:, 1]
    B_n = C_n[:, 2]

    # Step 2 — Build projection signals
    S1 = R_n - G_n
    S2 = R_n + G_n - 2.0 * B_n

    # Step 3 — Scale S2
    std_S1 = np.std(S1)
    std_S2 = np.std(S2)
    alpha  = std_S1 / (std_S2 + epsilon)
    S2     = alpha * S2

    # Step 4 — Combine
    pulse = S1 + S2

    # Step 5 — Zero mean
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
    window_len = int(WINDOWS_SECONDS * fps)
    step = window_len // 2

    if n_frames < window_len:
        print(f"[pos_algorithm] WARNING: Signal ({n_frames} frames) shorter "
              f"than window ({window_len} frames). Processing as single window.")
        pulse = _pos_window(rgb_signals)
        pulse = pulse - np.mean(pulse)
        print(f"[pos_algorithm] Done. Pulse signal shape: {pulse.shape}")
        return pulse, fps

    print(f"[pos_algorithm] Running POS | frames={n_frames} | "
          f"window={window_len} | step={step}")
    
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

    print(f"[pos_algorithm] Done. Pulse signal shape : {pulse.shape}")
    print(f"[pos_algorithm] Signal std               : {np.std(pulse):.6f}")
    print(f"[pos_algorithm] Signal range             : "
          f"[{pulse.min():.4f}, {pulse.max():.4f}]")

    return pulse, fps



if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from video_processor import process_video
    from roi_extractor import extract_roi_signals

    if len(sys.argv) < 2:
        print("Usage: python pos_algorithm.py path/to/video.mp4")
        sys.exit()

    test_path = sys.argv[1]

    print("=" * 50)
    print("STEP 1: video_processor")
    print("=" * 50)
    crops, fps = process_video(test_path)
    if crops is None:
        print("✗ video_processor failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STEP 2: roi_extractor")
    print("=" * 50)
    rgb_signals, fps = extract_roi_signals(crops, fps, debug=False)
    if rgb_signals is None:
        print("✗ roi_extractor failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STEP 3: pos_algorithm")
    print("=" * 50)
    pulse, fps = extract_pulse_signal(rgb_signals, fps)

    if pulse is not None:
        print(f"\n✓ Success: pulse signal shape : {pulse.shape}")
        print(f"✓ FPS passed downstream       : {fps}")

        # Save a plot of the raw pulse signal so we can visually
        # inspect it — should look like a noisy periodic wave
        plot_path = os.path.join(os.getcwd(), "debug_pulse_raw.png")
        plt.figure(figsize=(12, 4))
        time_axis = np.arange(len(pulse)) / fps
        plt.plot(time_axis, pulse, color="royalblue", linewidth=0.8)
        plt.title("Raw POS Pulse Signal (before bandpass filter)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Raw pulse plot saved        : debug_pulse_raw.png")
    else:
        print("\n✗ extract_pulse_signal returned None — check errors above")
