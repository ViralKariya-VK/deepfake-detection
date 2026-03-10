import numpy as np
import sys
import os
from scipy.signal import butter, filtfilt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOW_CUTOFF = 0.7
HIGH_CUTOFF = 3.0
FILTER_ORDER = 4


def _bandpass_filter(signal: np.ndarray, fps: float) -> np.ndarray:
    nyquist = fps / 2.0
    low = LOW_CUTOFF  / nyquist
    high = min(HIGH_CUTOFF, nyquist * 0.99) / nyquist

    low = max(low,  1e-4)
    high = min(high, 1.0 - 1e-4)

    b, a = butter(FILTER_ORDER, [low, high], btype='band')

    min_len = 3 * FILTER_ORDER + 1
    if len(signal) <= min_len:
        from scipy.signal import lfilter
        print(f"[signal_processing] WARNING: Signal too short for filtfilt "
              f"({len(signal)} samples). Using lfilter instead.")
        return lfilter(b, a, signal)

    return filtfilt(b, a, signal)


def _compute_fft(
    filtered_signal : np.ndarray,
    fps             : float
) -> tuple[np.ndarray, np.ndarray]:

    n = len(filtered_signal)
    fft_vals = np.fft.rfft(filtered_signal)
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fps)
    magnitudes = np.abs(fft_vals)

    mask = (fft_freqs >= LOW_CUTOFF) & (fft_freqs <= HIGH_CUTOFF)
    freqs = fft_freqs[mask]
    magnitudes = magnitudes[mask]

    return freqs, magnitudes


def process_signal(
    pulse_signal : np.ndarray,
    fps          : float
) -> tuple:

    print(f"[signal_processing] Applying bandpass filter "
          f"({LOW_CUTOFF}–{HIGH_CUTOFF} Hz) | fps={fps}")

    # ── Step 1: Bandpass filter ──
    filtered = _bandpass_filter(pulse_signal, fps)

    # ── Step 2: Validate filtered signal ──
    if np.std(filtered) < 1e-8:
        print("[signal_processing] ERROR: Filtered signal is flat. "
              "Bandpass filter removed all signal content.")
        return None, None, None, None

    print(f"[signal_processing] Filtered signal std   : {np.std(filtered):.6f}")
    print(f"[signal_processing] Filtered signal range : "
          f"[{filtered.min():.4f}, {filtered.max():.4f}]")

    # ── Step 3: Compute FFT ──
    freqs, magnitudes = _compute_fft(filtered, fps)

    if len(freqs) == 0:
        print("[signal_processing] ERROR: No frequencies found in "
              "heartbeat range after FFT.")
        return None, None, None, None

    # ── Step 4: Log dominant frequency ──
    dominant_idx = np.argmax(magnitudes)
    dominant_freq = freqs[dominant_idx]
    dominant_bpm = dominant_freq * 60.0

    print(f"[signal_processing] Dominant frequency    : "
          f"{dominant_freq:.3f} Hz ({dominant_bpm:.1f} BPM)")
    print(f"[signal_processing] Done.")

    return filtered, freqs, magnitudes, fps


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from video_processor  import process_video
    from roi_extractor    import extract_roi_signals
    from pos_algorithm    import extract_pulse_signal

    if len(sys.argv) < 2:
        print("Usage: python signal_processing.py path/to/video.mp4")
        sys.exit(1)

    test_path = sys.argv[1]

    print("=" * 50)
    print("STEP 1: video_processor")
    print("=" * 50)
    crops, fps = process_video(test_path)
    if crops is None:
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STEP 2: roi_extractor")
    print("=" * 50)
    rgb_signals, fps = extract_roi_signals(crops, fps, debug=False)
    if rgb_signals is None:
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STEP 3: pos_algorithm")
    print("=" * 50)
    pulse, fps = extract_pulse_signal(rgb_signals, fps)
    if pulse is None:
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STEP 4: signal_processing")
    print("=" * 50)
    filtered, freqs, magnitudes, fps = process_signal(pulse, fps)

    if filtered is not None:
        print(f"\n✓ Filtered signal shape : {filtered.shape}")
        print(f"✓ FFT freqs shape       : {freqs.shape}")
        print(f"✓ FPS passed downstream : {fps}")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        time_axis = np.arange(len(pulse)) / fps

        axes[0].plot(time_axis, pulse,
                     color="royalblue", linewidth=0.8, label="Raw POS signal")
        axes[0].set_title("Raw POS Signal (before bandpass filter)")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()

        axes[1].plot(time_axis, filtered,
                     color="crimson", linewidth=0.8, label="Filtered signal")
        axes[1].set_title(f"Filtered Signal ({LOW_CUTOFF}–{HIGH_CUTOFF} Hz bandpass)")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()

        plt.tight_layout()
        plot1_path = os.path.join(os.getcwd(), "debug_filtered_signal.png")
        plt.savefig(plot1_path, dpi=150)
        plt.close()
        print(f"✓ Signal plot saved     : debug_filtered_signal.png")

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, magnitudes, color="darkorange", linewidth=1.2)
        dominant_freq = freqs[np.argmax(magnitudes)]
        plt.axvline(x=dominant_freq, color="red", linestyle="--",
                    label=f"Peak: {dominant_freq:.2f} Hz "
                          f"({dominant_freq*60:.1f} BPM)")
        plt.title("FFT Frequency Spectrum (heartbeat range)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        plot2_path = os.path.join(os.getcwd(), "debug_fft_spectrum.png")
        plt.savefig(plot2_path, dpi=150)
        plt.close()
        print(f"✓ FFT spectrum saved    : debug_fft_spectrum.png")

    else:
        print("\n✗ process_signal returned None — check errors above")