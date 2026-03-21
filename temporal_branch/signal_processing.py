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
    
    return fft_freqs[mask], magnitudes[mask]


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


