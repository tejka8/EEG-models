"""
direct_predict.py - Compare training-style vs dashboard-style preprocessing.

Place at D:/EEG-FM-Bench/direct_predict.py
Run from the SAME terminal/env that ran uvicorn (where torch loads):
    python direct_predict.py 151
    python direct_predict.py v107
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mne

from baseline.neurogpt.model import NeuroGPTModel

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CKPT_PATH    = ROOT / "dashboard" / "models" / "neurogpt_adhd_epoch_7.pt"
SUBJECTS_DIR = ROOT / "assets" / "data" / "raw" / "ADHD" / "subjects"

CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
            'FZ', 'CZ', 'PZ']

SFREQ_INPUT    = 128.0
SFREQ_OUTPUT   = 256.0
WINDOW_SAMPLES = 1024  # 4 sec at 256Hz


# ─── TRAINING-STYLE PREPROCESSING ────────────────────────────────────────────
# Mimics adhd.py _read_raw_data + builder.py _resample_and_filter exactly.
def preprocess_training_style(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)

    csv_cols_upper = {c.upper(): c for c in df.columns}
    selected_cols  = [csv_cols_upper[ch] for ch in CHANNELS]
    eeg_df         = df[selected_cols]

    # Transpose + convert µV to V (training does this!)
    data = eeg_df.values.T.astype(np.float64)
    data = data * 1e-6

    info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ_INPUT,
                           ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)

    # set_montage like training does
    dig_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(dig_montage, match_case=False,
                    on_missing='ignore', verbose=False)

    # Filter FIRST (training order)
    # h_freq = None because orig_fs=128 < filter_high*2=200
    raw = raw.filter(l_freq=0.1, h_freq=None, verbose=False)

    # THEN notch
    raw = raw.notch_filter(freqs=[50.0], verbose=False)

    # THEN resample
    raw = raw.resample(sfreq=SFREQ_OUTPUT, verbose=False)

    # Get data in µV (training does this!)
    final_data = raw.get_data(units='uV').astype(np.float32)

    return _window(final_data)


# ─── DASHBOARD-STYLE PREPROCESSING ───────────────────────────────────────────
# Mimics current dashboard/backend/preprocessor.py
def preprocess_dashboard_style(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)

    col_map = {c.upper(): c for c in df.columns}
    eeg_df  = df[[col_map[ch] for ch in CHANNELS]]
    data    = eeg_df.to_numpy(dtype=np.float64).T
    # NO * 1e-6 (dashboard skips this)

    info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ_INPUT,
                           ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    # NO set_montage

    # Notch FIRST (dashboard order, opposite of training!)
    raw.notch_filter(freqs=50.0, verbose=False)
    raw.filter(l_freq=0.1, h_freq=None, verbose=False)
    raw.resample(SFREQ_OUTPUT, verbose=False)

    # get_data WITHOUT units arg
    final_data = raw.get_data().astype(np.float32)

    return _window(final_data)


def _window(data: np.ndarray) -> np.ndarray:
    n_samples = data.shape[1]
    n_windows = n_samples // WINDOW_SAMPLES
    if n_windows == 0:
        raise ValueError(f"Recording too short ({n_samples} samples)")

    trimmed = data[:, :n_windows * WINDOW_SAMPLES]
    windows = trimmed.reshape(19, n_windows, WINDOW_SAMPLES)
    return windows.transpose(1, 0, 2)  # (N, 19, 1024)


# ─── MODEL ──────────────────────────────────────────────────────────────────
def load_neurogpt():
    model = NeuroGPTModel(
        n_chans=19, n_times=1024, num_classes=2,
        ds_name='adhd', num_chunks=2, chunk_len=500,
        ft_only_encoder=True,
    )
    checkpoint = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        print(f"  [WARN] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


# ─── INFERENCE ───────────────────────────────────────────────────────────────
def predict(windows: np.ndarray, model) -> dict:
    tensor = torch.FloatTensor(windows)
    with torch.no_grad():
        output = model({'data': tensor * 0.001})  # µV → mV
        probs = F.softmax(output, dim=1).numpy()

    predicted = np.argmax(probs, axis=1)
    adhd_votes    = int(np.sum(predicted == 0))
    control_votes = int(np.sum(predicted == 1))

    overall    = "ADHD" if adhd_votes >= control_votes else "Control"
    winning    = 0 if overall == "ADHD" else 1
    confidence = float(np.mean(probs[:, winning])) * 100

    # Per-window summary
    per_window = [
        {"window": i+1, "prediction": "ADHD" if c == 0 else "Control",
         "confidence": float(probs[i, c])}
        for i, c in enumerate(predicted)
    ]

    return {
        "prediction":    overall,
        "confidence":    confidence,
        "adhd_votes":    adhd_votes,
        "control_votes": control_votes,
        "n_windows":     len(predicted),
        "per_window":    per_window,
    }


def print_result(label: str, result: dict):
    print(f"\n[{label}]")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Confidence:  {result['confidence']:.1f}%")
    print(f"  Votes:       {result['adhd_votes']} ADHD / {result['control_votes']} Control "
          f"(of {result['n_windows']} windows)")
    print(f"  Window-by-window confidences (first 10):")
    for w in result['per_window'][:10]:
        bar_len = int(w['confidence'] * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"    W{w['window']:2d}: {w['prediction']:8s} {bar} {w['confidence']*100:5.1f}%")


def compare_data(t: np.ndarray, d: np.ndarray):
    print("\n" + "=" * 60)
    print("DATA STATISTICS COMPARISON")
    print("=" * 60)
    print(f"\n  Training-style:  shape={t.shape}")
    print(f"    min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}  std={t.std():.4f}")
    print(f"\n  Dashboard-style: shape={d.shape}")
    print(f"    min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}  std={d.std():.4f}")

    if t.shape == d.shape:
        abs_diff  = np.abs(t - d)
        max_t     = max(np.abs(t).max(), 1e-10)
        rel_diff  = abs_diff / max_t

        print(f"\n  Element-wise (training as reference):")
        print(f"    max abs diff:  {abs_diff.max():.6f}")
        print(f"    mean abs diff: {abs_diff.mean():.6f}")
        print(f"    max rel diff:  {rel_diff.max() * 100:.2f}%")
        print(f"    mean rel diff: {rel_diff.mean() * 100:.2f}%")

        # Ratio (in case they're scaled differently)
        ratio = d.mean() / t.mean() if t.mean() != 0 else 0
        print(f"    ratio (dashboard/training mean): {ratio:.6f}")


def main(subject_id: str):
    csv_path = SUBJECTS_DIR / f"{subject_id}.csv"
    if not csv_path.exists():
        # Try common alternatives
        for alt in [f"v{subject_id}.csv", f"V{subject_id}.csv"]:
            p = SUBJECTS_DIR / alt
            if p.exists():
                csv_path = p
                break
        else:
            print(f"[ERROR] Not found: {csv_path}")
            print(f"  Available files (first 20):")
            for f in sorted(SUBJECTS_DIR.glob("*.csv"))[:20]:
                print(f"    {f.name}")
            return

    print(f"Subject:    {csv_path.name}")
    print(f"Checkpoint: {CKPT_PATH.name}")

    print(f"\nLoading model...")
    model = load_neurogpt()

    print(f"\nPreprocessing TRAINING-STYLE...")
    t_windows = preprocess_training_style(csv_path)

    print(f"Preprocessing DASHBOARD-STYLE...")
    d_windows = preprocess_dashboard_style(csv_path)

    compare_data(t_windows, d_windows)

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)

    t_result = predict(t_windows, model)
    print_result("TRAINING-STYLE PREPROCESSING", t_result)

    d_result = predict(d_windows, model)
    print_result("DASHBOARD-STYLE PREPROCESSING", d_result)

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    diff = abs(t_result['confidence'] - d_result['confidence'])

    if t_result['confidence'] >= 80 and d_result['confidence'] < 70:
        print("\n  PREPROCESSING IS THE PROBLEM.")
        print(f"  Training-style:  {t_result['confidence']:.1f}% (high confidence)")
        print(f"  Dashboard-style: {d_result['confidence']:.1f}% (low confidence)")
        print(f"\n  Action: fix dashboard/backend/preprocessor.py to match training.")
    elif diff < 5:
        print("\n  Preprocessing is NOT the problem.")
        print(f"  Both give ~same result: train={t_result['confidence']:.1f}%, "
              f"dash={d_result['confidence']:.1f}%")
        print(f"  Confidence is genuinely low for this patient.")
    else:
        print(f"\n  Mixed result. Both differ by {diff:.1f}%.")
        print(f"  Training-style:  {t_result['confidence']:.1f}%")
        print(f"  Dashboard-style: {d_result['confidence']:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python direct_predict.py <subject_id>")
        print("Examples:")
        print("  python direct_predict.py 151")
        print("  python direct_predict.py v107")
        sys.exit(1)
    main(sys.argv[1])