import numpy as np
import sys
import os
import joblib
from scipy.signal import find_peaks
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocsvm_temporal.pkl")

PEAK_MIN_DISTANCE = 5
PEAK_HEIGHT_FACTOR = 0.3


def extract_features(
        filtered_signal : np.ndarray,
        freqs : np.ndarray,
        magnitudes : np.ndarray,
        fps : float
) -> np.ndarray:
    
    # ── Feature 1: Signal Power ──
    signal_power = np.mean(filtered_signal ** 2)

    # ── Feature 2: SNR ──
    epsilon = 1e-10
    dominant_idx = np.argmax(magnitudes)
    signal_mag = magnitudes[dominant_idx] ** 2
    noise_mag = np.sum(magnitudes ** 2) - signal_mag
    snr = signal_mag / (noise_mag + epsilon)


    # ── Feature 3: Peak Regularity ──
    min_height = PEAK_HEIGHT_FACTOR * np.std(filtered_signal)
    peaks, _ = find_peaks(
        filtered_signal,
        distance = PEAK_MIN_DISTANCE,
        height = min_height
    )

    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        peak_regularity = 1.0 - (std_interval / (mean_interval + epsilon))
        peak_regularity = float(np.clip(peak_regularity, 0.0, 1.0))
    else:
        peak_regularity = 0.0

    # ── Feature 4: Frequency Entropy ──
    mag_sum = np.sum(magnitudes) + epsilon
    p = magnitudes / mag_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    frq_entropy = -entropy

    # ── Feature 5: Heart Rate ──
    heart_rate = freqs[dominant_idx] * 60.0

    # ── Feature 6: Pulse Amplitude ──
    if len(peaks) >= 1:
        pulse_amplitude = float(np.mean(filtered_signal[peaks]))
    else:
        pulse_amplitude = 0.0

    fetaure_vector = np.array([
        signal_power,
        snr,
        peak_regularity,
        frq_entropy,
        heart_rate,
        pulse_amplitude
    ], dtype = np.float32)

    return fetaure_vector

def train_ocsvm(feature_matrix: np.ndarray) -> tuple:
    print(f"[vppg_scorer] Training One-Class SVM on "
          f"{feature_matrix.shape[0]} real video samples...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)

    model = OneClassSVM(kernel='rbf', nu = 0.1, gamma='scale')
    model.fit(features_scaled)

    save_dict = {"scaler": scaler, "model": model}
    joblib.dump(save_dict, MODEL_PATH)
    print(f"[vppg_scorer] Model saved to: {MODEL_PATH}")

    return scaler, model


def _load_model() -> tuple:
    if not os.path.exists(MODEL_PATH):
        print(f"[vppg_scorer] No trained model found at {MODEL_PATH}. "
              f"Run train_ocsvm() first.")
        return None, None
    
    save_dict = joblib.load(MODEL_PATH)
    scaler = save_dict["scaler"]
    model = save_dict["model"]
    print(f"[vppg_scorer] Model loaded from: {MODEL_PATH}")
    
    return scaler, model


def score_video(feature_vector: np.ndarray) -> float:
    scaler, model = _load_model()

    if scaler is None:
        print("[vppg_scorer] ERROR: Cannot score without trained model")
        return None
    
    fv_scaled = scaler.transform(feature_vector.reshape(1, -1))

    raw_score = model.decision_function(fv_scaled)[0]

    temporal_score = 1.0 / (1.0 + np.exp(raw_score))

    return float(temporal_score)


def get_temporal_score(
    filtered_signal : np.ndarray,
    freqs           : np.ndarray,
    magnitudes      : np.ndarray,
    fps             : float
) -> float | None:
    
    feature_vector = extract_features(
        filtered_signal, freqs, magnitudes, fps
    )

    print(f"[vppg_scorer] Feature vector:")
    print(f"  signal_power    : {feature_vector[0]:.6f}")
    print(f"  snr             : {feature_vector[1]:.6f}")
    print(f"  peak_regularity : {feature_vector[2]:.6f}")
    print(f"  freq_entropy    : {feature_vector[3]:.6f}")
    print(f"  heart_rate      : {feature_vector[4]:.2f} BPM")
    print(f"  pulse_amplitude : {feature_vector[5]:.6f}")

    temporal_score = score_video(feature_vector)

    if temporal_score is not None:
        print(f"[vppg_scorer] temporal_score: {temporal_score:.4f} "
              f"→ {'FAKE' if temporal_score > 0.5 else 'REAL'}")

    return temporal_score




if __name__ == "__main__":
    from video_processor  import process_video
    from roi_extractor    import extract_roi_signals
    from pos_algorithm    import extract_pulse_signal
    from signal_processing import process_signal

    if len(sys.argv) < 2:
        print("Usage: python vppg_scorer.py path/to/video.mp4")
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
    if filtered is None:
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STEP 5: vppg_scorer — feature extraction")
    print("=" * 50)
    feature_vector = extract_features(filtered, freqs, magnitudes, fps)

    print(f"\n✓ Feature vector extracted successfully:")
    print(f"  [0] signal_power    : {feature_vector[0]:.6f}")
    print(f"  [1] snr             : {feature_vector[1]:.6f}")
    print(f"  [2] peak_regularity : {feature_vector[2]:.6f}")
    print(f"  [3] freq_entropy    : {feature_vector[3]:.6f}")
    print(f"  [4] heart_rate      : {feature_vector[4]:.2f} BPM")
    print(f"  [5] pulse_amplitude : {feature_vector[5]:.6f}")

    # ── Train a dummy model on this single video ──
    # Just to confirm the full pipeline runs end to end.
    # Real training happens in the notebook with all real videos.
    print("\n" + "=" * 50)
    print("STEP 5b: train dummy model on single video")
    print("(Real training uses all real videos in notebook)")
    print("=" * 50)
    dummy_matrix = np.tile(feature_vector, (10, 1))   # repeat 10x
    # Add small noise so SVM doesn't get a degenerate matrix
    dummy_matrix += np.random.normal(0, 0.01, dummy_matrix.shape)
    train_ocsvm(dummy_matrix)

    print("\n" + "=" * 50)
    print("STEP 6: get_temporal_score")
    print("=" * 50)
    score = get_temporal_score(filtered, freqs, magnitudes, fps)

    if score is not None:
        print(f"\n✓ temporal_score : {score:.4f}")
        print(f"✓ Verdict        : {'FAKE' if score > 0.5 else 'REAL'}")
        print(f"\nNOTE: This score used a dummy model trained on 1 video.")
        print(f"Real scores require training on full Celeb-DF real videos.")
    else:
        print("\n✗ get_temporal_score returned None — check errors above")
