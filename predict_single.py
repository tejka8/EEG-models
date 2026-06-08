"""
predict_single.py — Standalone CLI for ADHD prediction from EEG.

Uses EEG-FM-Bench model code directly (NOT dashboard backend).
Loads a .pt checkpoint, processes one CSV recording, prints prediction.

Place this file at:  D:/EEG-FM-Bench/predict_single.py

Usage examples:
    python predict_single.py v310
    python predict_single.py v310 --model neurogpt
    python predict_single.py v310 --verbose
    python predict_single.py v310 --model eegnet
    python predict_single.py D:\\custom\\path\\to\\file.csv --csv
    python predict_single.py v310 --checkpoint D:\\custom\\path.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mne

# ─── Make EEG-FM-Bench importable (we're inside it) ──────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SUBJECTS_DIR = ROOT / "assets" / "data" / "raw" / "ADHD" / "subjects"
MODELS_DIR   = ROOT / "dashboard" / "models"

DEFAULT_CHECKPOINTS = {
    "neurogpt": MODELS_DIR / "neurogpt_adhd_epoch_7.pt",
    "eegnet":   MODELS_DIR / "eegnet_adhd_epoch_10.pt",
}

CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
            'FZ', 'CZ', 'PZ']

SFREQ_INPUT    = 128.0
SFREQ_OUTPUT   = 256.0
WINDOW_SAMPLES = 1024   # 4 sec at 256 Hz

LABELS = {0: "ADHD", 1: "Control"}


# ─── PREPROCESSING ───────────────────────────────────────────────────────────
def preprocess_csv(csv_path: Path) -> np.ndarray:
    """Read CSV → notch → bandpass → resample → window. Returns (N, 19, 1024)."""
    df = pd.read_csv(csv_path)

    # Case-insensitive channel mapping
    col_map = {c.upper(): c for c in df.columns}
    missing = [ch for ch in CHANNELS if ch not in col_map]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    eeg = df[[col_map[ch] for ch in CHANNELS]].to_numpy(dtype=np.float64).T

    info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ_INPUT,
                           ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(eeg, info, verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    raw.filter(l_freq=0.1, h_freq=60.0, verbose=False)
    raw.resample(SFREQ_OUTPUT, verbose=False)

    data = raw.get_data()
    n_total = data.shape[1]
    n_windows = n_total // WINDOW_SAMPLES
    if n_windows == 0:
        raise ValueError(f"Recording too short for one 4-second window")

    trimmed = data[:, :n_windows * WINDOW_SAMPLES]
    windows = trimmed.reshape(19, n_windows, WINDOW_SAMPLES).transpose(1, 0, 2)
    return windows  # (N, 19, 1024)


# ─── MODEL LOADING (using EEG-FM-Bench classes) ──────────────────────────────
def load_neurogpt(ckpt_path: Path):
    from baseline.neurogpt.model import NeuroGPTModel

    model = NeuroGPTModel(
        n_chans=19, n_times=1024, num_classes=2,
        ds_name='adhd', num_chunks=2, chunk_len=500,
        ft_only_encoder=True,
    )

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Strip 'module.' prefix from DDP-trained checkpoints
    clean = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        clean[key] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"  [warn] Missing {len(missing)} keys", file=sys.stderr)
    model.eval()
    return model


def load_eegnet(ckpt_path: Path):
    import braindecode.models

    model = braindecode.models.EEGNet(
        n_outputs=2, n_chans=19, n_times=1024, sfreq=256.0,
    )

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    clean = {}
    for k, v in state_dict.items():
        if k.startswith('module.encoder.'):
            clean[k[len('module.encoder.'):]] = v

    if not clean:
        raise RuntimeError("No weights matched 'module.encoder.' prefix in checkpoint")

    model.load_state_dict(clean, strict=False)
    model.eval()
    return model


# ─── INFERENCE ───────────────────────────────────────────────────────────────
def predict(windows: np.ndarray, model, model_type: str) -> dict:
    tensor = torch.FloatTensor(windows)  # (N, 19, 1024)

    with torch.no_grad():
        if model_type == "neurogpt":
            output = model({'data': tensor * 0.001})  # µV → mV
        else:  # eegnet
            output = model(tensor)

        probs = F.softmax(output, dim=1).numpy()  # (N, 2)

    predicted = np.argmax(probs, axis=1)
    adhd_votes    = int(np.sum(predicted == 0))
    control_votes = int(np.sum(predicted == 1))
    n_windows = len(predicted)

    overall = "ADHD" if adhd_votes >= control_votes else "Control"
    winning_idx = 0 if overall == "ADHD" else 1
    confidence = float(np.mean(probs[:, winning_idx])) * 100

    reliability = "High" if confidence > 75 else "Medium" if confidence > 55 else "Low"

    per_window = [
        {"window": i + 1, "prediction": LABELS[int(c)],
         "confidence": float(probs[i, c])}
        for i, c in enumerate(predicted)
    ]

    return {
        "prediction":    overall,
        "confidence":    confidence,
        "reliability":   reliability,
        "adhd_votes":    adhd_votes,
        "control_votes": control_votes,
        "n_windows":     n_windows,
        "per_window":    per_window,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict ADHD/Control for one EEG recording (CLI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_single.py v310
  python predict_single.py v310 --model neurogpt
  python predict_single.py v310 --verbose
  python predict_single.py D:\\custom\\path\\file.csv --csv
        """
    )
    parser.add_argument("subject",
                        help="Subject ID (e.g., v310) — auto-resolved to "
                             "assets/data/raw/ADHD/subjects/<id>.csv. "
                             "OR pass full CSV path with --csv flag.")
    parser.add_argument("--csv", action="store_true",
                        help="Treat 'subject' as a full path to a CSV file")
    parser.add_argument("--model", choices=["neurogpt", "eegnet"],
                        default="neurogpt",
                        help="Which model to use (default: neurogpt)")
    parser.add_argument("--checkpoint", type=Path,
                        help="Override checkpoint path (default: dashboard/models/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-window predictions")
    args = parser.parse_args()

    # Resolve CSV path
    if args.csv:
        csv_path = Path(args.subject)
    else:
        csv_path = SUBJECTS_DIR / f"{args.subject}.csv"

    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve checkpoint
    ckpt_path = args.checkpoint or DEFAULT_CHECKPOINTS[args.model]
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Subject:    {csv_path.name}")
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {ckpt_path.name}")
    print()

    # Preprocess
    print("Preprocessing...")
    windows = preprocess_csv(csv_path)
    print(f"  -> {windows.shape[0]} windows of 4 seconds each")

    # Load model
    print("Loading model...")
    if args.model == "neurogpt":
        model = load_neurogpt(ckpt_path)
    else:
        model = load_eegnet(ckpt_path)

    # Predict
    print("Running inference...")
    result = predict(windows, model, args.model)

    print()
    print("=" * 56)
    print(f"  PREDICTION:   {result['prediction']}")
    print(f"  Confidence:   {result['confidence']:.1f}%")
    print(f"  Reliability:  {result['reliability']}")
    print(f"  Votes:        {result['adhd_votes']} ADHD / "
          f"{result['control_votes']} Control")
    print(f"  Total windows: {result['n_windows']}")
    print("=" * 56)

    if args.verbose:
        print()
        print("Per-window predictions:")
        for w in result["per_window"]:
            bar_len = int(w["confidence"] * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            print(f"  W{w['window']:2d}: {w['prediction']:8s} "
                  f"{bar} {w['confidence']*100:5.1f}%")


if __name__ == "__main__":
    main()