"""
crown_dataset_builder.py — Build 8-channel Neurosity Crown dataset from the 
19-channel ADHD dataset using MNE spherical spline interpolation.

Place at D:/EEG-FM-Bench/crown_dataset_builder.py
Run from the same env that ran your existing training:
    python crown_dataset_builder.py --dry-run      # test on 3 subjects first
    python crown_dataset_builder.py --subject v10p # test one subject
    python crown_dataset_builder.py                # full run on all subjects

Output: assets/data/raw/ADHD_Crown/subjects/*.csv with 8 Crown channels each.
"""
import sys
import shutil
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mne


# ─── CONFIGURATION ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

SOURCE_DIR = ROOT / "assets" / "data" / "raw" / "ADHD"
TARGET_DIR = ROOT / "assets" / "data" / "raw" / "ADHD_Crown"

ADHD_19CH = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
             'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
             'FZ', 'CZ', 'PZ']

CROWN_8CH = ['F5', 'F6', 'C3', 'C4', 'CP3', 'CP4', 'PO3', 'PO4']

# Channels that need spatial interpolation (in Crown but not in 19-ch source)
TO_INTERPOLATE = [ch for ch in CROWN_8CH if ch not in ADHD_19CH]
# = ['F5', 'F6', 'CP3', 'CP4', 'PO3', 'PO4']  (6 channels)

# Channels that already exist in source (just copy)
DIRECT_COPY = [ch for ch in CROWN_8CH if ch in ADHD_19CH]
# = ['C3', 'C4']  (2 channels)

SFREQ = 128.0  # ADHD dataset native sampling rate


# ─── SPATIAL INTERPOLATION ───────────────────────────────────────────────────
def interpolate_to_crown(data_19ch: np.ndarray) -> np.ndarray:
    """
    Convert 19-channel EEG to 8-channel Crown via MNE spherical spline interpolation.

    Spherical spline interpolation (Perrin et al. 1989) projects values from
    known sensor positions onto unknown positions on the spherical scalp model,
    using all available sensors weighted by spherical distance.

    Args:
        data_19ch: (19, n_samples) array in µV, channel order = ADHD_19CH

    Returns:
        data_8ch: (8, n_samples) array in µV, channel order = CROWN_8CH
    """
    n_samples = data_19ch.shape[1]

    # Build extended channel list: 19 originals + 6 to interpolate
    all_channels = list(ADHD_19CH) + TO_INTERPOLATE
    n_total = len(all_channels)

    # Build extended data: real for first 19, zeros for last 6 (placeholders)
    full_data = np.zeros((n_total, n_samples), dtype=np.float64)
    full_data[:19] = data_19ch

    # Convert µV → V for MNE (MNE stores everything in SI units = Volts)
    full_data = full_data * 1e-6

    # Build MNE Raw object
    info = mne.create_info(all_channels, sfreq=SFREQ, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(full_data, info, verbose=False)

    # Use 10-05 montage — contains both 10-20 positions AND 10-10 positions
    # (F5, F6, CP3, CP4, PO3, PO4 are all defined in 10-05)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)

    # Mark the 6 placeholder channels as "bad" so MNE will interpolate them
    raw.info['bads'] = TO_INTERPOLATE

    # Apply spherical spline interpolation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.interpolate_bads(reset_bads=True, method='spline', verbose=False)

    # Extract the 8 Crown channels in canonical order, back in µV
    raw_crown = raw.copy().pick_channels(CROWN_8CH, ordered=True)
    data_crown = raw_crown.get_data(units='uV').astype(np.float32)

    return data_crown


# ─── CSV PROCESSING ──────────────────────────────────────────────────────────
def process_csv(source_csv: Path, target_csv: Path) -> dict:
    """Process one subject CSV: read 19ch, interpolate to 8ch, save."""
    df = pd.read_csv(source_csv)

    # Case-insensitive column lookup
    col_map = {c.upper(): c for c in df.columns}
    missing = [ch for ch in ADHD_19CH if ch not in col_map]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    # Original case names for the 19 EEG columns
    eeg_orig_cols = [col_map[ch] for ch in ADHD_19CH]

    # Extract 19-channel data in canonical order
    data_19ch = df[eeg_orig_cols].to_numpy(dtype=np.float64).T

    # Preserve any non-EEG columns (timestamps, labels, etc.)
    non_eeg_cols = [c for c in df.columns if c not in eeg_orig_cols]

    # Apply spatial interpolation
    data_8ch = interpolate_to_crown(data_19ch)

    # Build output dataframe with Crown channel names in order
    out_df = pd.DataFrame(data_8ch.T, columns=CROWN_8CH)

    # Re-attach non-EEG columns
    for col in non_eeg_cols:
        out_df[col] = df[col].values

    # Save
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(target_csv, index=False)

    # Verification: check that C3/C4 are preserved (they should be ~identical)
    c3_idx_src = ADHD_19CH.index('C3')
    c3_idx_dst = CROWN_8CH.index('C3')
    c3_max_diff = np.abs(data_19ch[c3_idx_src] - data_8ch[c3_idx_dst]).max()

    return {
        'n_samples':   data_19ch.shape[1],
        'src_range':   (float(data_19ch.min()), float(data_19ch.max())),
        'dst_range':   (float(data_8ch.min()),  float(data_8ch.max())),
        'extra_cols':  non_eeg_cols,
        'c3_max_diff': float(c3_max_diff),
    }


# ─── METADATA COPY ───────────────────────────────────────────────────────────
def copy_metadata():
    """Copy summary/ (with split info) to target dataset."""
    src_summary = SOURCE_DIR / "summary"
    if src_summary.exists():
        dst_summary = TARGET_DIR / "summary"
        if dst_summary.exists():
            shutil.rmtree(dst_summary)
        shutil.copytree(src_summary, dst_summary)
        print(f"  Copied summary/ → {dst_summary}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Process only the first 3 subjects (verification)')
    parser.add_argument('--subject', type=str, default=None,
                        help='Process only one specific subject (e.g., v10p)')
    args = parser.parse_args()

    source_subjects = SOURCE_DIR / "subjects"
    target_subjects = TARGET_DIR / "subjects"

    if not source_subjects.exists():
        print(f"ERROR: source directory not found: {source_subjects}")
        sys.exit(1)

    csv_files = sorted(source_subjects.glob("*.csv"))

    if args.subject:
        csv_files = [f for f in csv_files if args.subject in f.stem]
        if not csv_files:
            print(f"ERROR: subject '{args.subject}' not found in {source_subjects}")
            sys.exit(1)
    elif args.dry_run:
        csv_files = csv_files[:3]

    print("=" * 70)
    print("Crown 8-channel dataset builder")
    print("=" * 70)
    print(f"Source:        {source_subjects}")
    print(f"Target:        {target_subjects}")
    print(f"Subjects:      {len(csv_files)}")
    print(f"Interpolating: {TO_INTERPOLATE}")
    print(f"Direct copy:   {DIRECT_COPY}")
    print()

    target_subjects.mkdir(parents=True, exist_ok=True)

    success = 0
    failures = []

    for i, src in enumerate(csv_files, 1):
        dst = target_subjects / src.name
        try:
            stats = process_csv(src, dst)
            success += 1

            extras_str = f", extras: {stats['extra_cols']}" if stats['extra_cols'] else ""
            print(f"[{i:3d}/{len(csv_files)}] {src.name:12s} | "
                  f"{stats['n_samples']:>6d} samples | "
                  f"19ch [{stats['src_range'][0]:>+8.1f}, {stats['src_range'][1]:>+8.1f}] µV → "
                  f"8ch [{stats['dst_range'][0]:>+8.1f}, {stats['dst_range'][1]:>+8.1f}] µV | "
                  f"C3 diff: {stats['c3_max_diff']:.4f}{extras_str}")
        except Exception as e:
            failures.append((src.name, str(e)))
            print(f"[{i:3d}/{len(csv_files)}] {src.name}: FAILED — {e}")

    print()
    print("=" * 70)
    print(f"Results: {success}/{len(csv_files)} succeeded")
    print("=" * 70)

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  {name}: {err}")

    if not args.dry_run and not args.subject and success > 0:
        copy_metadata()
        print(f"\nNew dataset ready at: {TARGET_DIR}")
        print(f"Next step: create adhd_crown.py dataset class (Phase 2)")


if __name__ == "__main__":
    main()
