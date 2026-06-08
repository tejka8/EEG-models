"""
diagnose_checkpoint.py — Find out why NeuroGPT predictions are ~50%.

Place this at D:/EEG-FM-Bench/diagnose_checkpoint.py
Run from a terminal where torch loads (e.g. the same terminal that ran uvicorn):
    python diagnose_checkpoint.py
"""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CKPT_PATH = ROOT / "dashboard" / "models" / "neurogpt_adhd_epoch_7.pt"

print(f"Loading checkpoint from {CKPT_PATH}...")
checkpoint = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"  Top-level checkpoint keys: {list(checkpoint.keys())}")
else:
    state_dict = checkpoint
    print(f"  Checkpoint IS the state dict")

print(f"  Total parameters in checkpoint: {len(state_dict)}")

# Strip module. prefix
clean = {}
for k, v in state_dict.items():
    key = k[7:] if k.startswith('module.') else k
    clean[key] = v

# ─── Check for LoRA / parametrization keys ───
print("\n" + "="*60)
print("LORA / PARAMETRIZATION CHECK")
print("="*60)
lora_keys = [k for k in clean.keys() if 'lora' in k.lower() or 'parametrizations' in k]
if lora_keys:
    print(f"\n*** FOUND {len(lora_keys)} LoRA/parametrization keys ***")
    print("This means LoRA was applied during training.\n")
    print("Sample of these keys:")
    for k in lora_keys[:15]:
        print(f"  {k}: {tuple(clean[k].shape)}")
    if len(lora_keys) > 15:
        print(f"  ... and {len(lora_keys) - 15} more")
else:
    print("\nNo LoRA / parametrization keys found.")

# ─── Categorize all keys ───
print("\n" + "="*60)
print("KEY CATEGORIES IN CHECKPOINT")
print("="*60)
categories = {
    'encoder.patch_embedding': [],
    'encoder.transformer':     [],
    'input_proj':              [],
    'decoder':                 [],
    'pooler':                  [],
    'classifier':              [],
    'OTHER':                   [],
}
for k in clean.keys():
    matched = False
    for cat in categories:
        if cat != 'OTHER' and k.startswith(cat):
            categories[cat].append(k)
            matched = True
            break
    if not matched:
        categories['OTHER'].append(k)

for cat, keys in categories.items():
    if not keys:
        continue
    print(f"\n  [{cat}] {len(keys)} keys")
    for k in keys[:3]:
        print(f"    {k}: {tuple(clean[k].shape)}")
    if len(keys) > 3:
        print(f"    ... and {len(keys) - 3} more")

# ─── Try to build the model and load ───
print("\n" + "="*60)
print("BUILDING NeuroGPTModel AND LOADING WEIGHTS")
print("="*60)

from baseline.neurogpt.model import NeuroGPTModel
model = NeuroGPTModel(
    n_chans=19, n_times=1024, num_classes=2,
    ds_name='adhd', num_chunks=2, chunk_len=500,
    ft_only_encoder=True,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model expects {len(model.state_dict())} parameter tensors ({total_params:,} total scalars)")

missing, unexpected = model.load_state_dict(clean, strict=False)

print(f"\n  MISSING from checkpoint (these stay RANDOM!): {len(missing)}")
for k in missing[:25]:
    print(f"    {k}")
if len(missing) > 25:
    print(f"    ... and {len(missing) - 25} more")

print(f"\n  UNEXPECTED in checkpoint (not loaded into model): {len(unexpected)}")
for k in unexpected[:25]:
    print(f"    {k}")
if len(unexpected) > 25:
    print(f"    ... and {len(unexpected) - 25} more")

# ─── Verdict ───
print("\n" + "="*60)
print("VERDICT")
print("="*60)
n_missing_encoder = sum(1 for k in missing if k.startswith('encoder.'))
n_missing_classifier = sum(1 for k in missing if k.startswith('classifier.'))

if not missing and not unexpected:
    print("  PERFECT: All weights match.")
elif not missing:
    print(f"  GOOD: All model weights loaded.")
    print(f"  ({len(unexpected)} extra checkpoint keys ignored — usually harmless)")
elif lora_keys and missing:
    print(f"  ROOT CAUSE FOUND: LoRA parametrizations.")
    print(f"  {len(missing)} model weights stay RANDOM because checkpoint uses")
    print(f"  '.parametrizations.weight.original' naming instead of '.weight'.")
    print(f"\n  Specifically:")
    print(f"    - encoder weights missing:    {n_missing_encoder}")
    print(f"    - classifier weights missing: {n_missing_classifier}")
    print(f"\n  >>> Fix: model_loader.py needs to rename parametrization keys.")
else:
    print(f"  CRITICAL: {len(missing)} model weights are random.")
    print(f"    - encoder weights missing:    {n_missing_encoder}")
    print(f"    - classifier weights missing: {n_missing_classifier}")
    print(f"\n  This explains the ~50% confidence in dashboard predictions.")

print("\n" + "="*60)
print("Send the full output of this script to continue troubleshooting.")
print("="*60)

# import pandas as pd

# info_csv = "D:/EEG-FM-Bench/assets/data/raw/ADHD/summary/finetune/adhd_finetune_info.csv"
# df = pd.read_csv(info_csv)

# # Покажи дистрибуција
# print("Splits distribution:")
# print(df.drop_duplicates(subset='subject').groupby(['split', 'group']).size())
# print()

# # Тренинг пациенти (по 3 ADHD и 3 Control)
# print("=== TRAIN ADHD (test these first!) ===")
# train_adhd = df[(df['split']=='train') & (df['group']=='ADHD')]['subject'].unique()[:3]
# print(train_adhd)

# print("\n=== TRAIN Control ===")
# train_ctrl = df[(df['split']=='train') & (df['group']=='Control')]['subject'].unique()[:3]
# print(train_ctrl)

# print("\n=== TEST set ===")
# test_subs = df[df['split']=='test'][['subject','group']].drop_duplicates()
# print(test_subs.to_string(index=False))

# # Проверка дали 151 е воопшто во датасетот
# print("\n=== Is '151' in dataset? ===")
# print(df[df['subject'].astype(str).str.contains('151')][['subject','group','split']].drop_duplicates())