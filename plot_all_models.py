import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re

# ============================================================
# CONFIG — paths to log files
# ============================================================
MODELS = {
    'EEG-GPT': {
        'log': r'D:\EEG-FM-Bench\assets\run\log\baseline\eegpt\local_260525180037\eegpt_trainer.log',
        'color_val':  '#2196F3',
        'color_test': '#F44336',
    },
    'EEGNet': {
        'log': r'D:\EEG-FM-Bench\assets\run\log\baseline\eegnet\local_260525134515\eegnet_trainer.log',
        'color_val':  '#4CAF50',
        'color_test': '#FF9800',
    },
    'Neuro-GPT': {
        'log': r'D:\EEG-FM-Bench\assets\run\log\baseline\neurogpt\local_260525184145\neurogpt_trainer.log',
        'color_val':  '#9C27B0',
        'color_test': '#E91E63',
    },
}


# ============================================================
# PARSE LOG FILE
# ============================================================
def parse_log(filepath):
    eval_rows = []
    test_rows = []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if 'eval epoch' not in line and 'test epoch' not in line:
                continue

            epoch_match   = re.search(r'epoch:\s*(\d+)', line)
            loss_match    = re.search(r'loss:\s*([\d.]+)', line)
            acc_match     = re.search(r'acc:\s*([\d.]+)', line)
            bal_match     = re.search(r'balanced_acc:\s*([\d.]+)', line)
            auroc_match   = re.search(r'auroc:\s*([\d.]+)', line)
            auc_pr_match  = re.search(r'auc_pr:\s*([\d.]+)', line)

            if not epoch_match:
                continue

            row = {
                'epoch':        int(epoch_match.group(1)),
                'loss':         float(loss_match.group(1))   if loss_match   else None,
                'acc':          float(acc_match.group(1))    if acc_match    else None,
                'balanced_acc': float(bal_match.group(1))    if bal_match    else None,
                'auroc':        float(auroc_match.group(1))  if auroc_match  else None,
                'auc_pr':       float(auc_pr_match.group(1)) if auc_pr_match else None,
            }

            if 'eval epoch' in line:
                eval_rows.append(row)
            else:
                test_rows.append(row)

    eval_df = pd.DataFrame(eval_rows).sort_values('epoch').reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values('epoch').reset_index(drop=True)
    return eval_df, test_df


# ============================================================
# PLOT — one figure per model (4 subplots each)
# ============================================================
def plot_model(name, eval_df, test_df, cv, ct):
    best_idx = test_df['auroc'].idxmax()
    best     = test_df.loc[best_idx]
    final    = test_df.iloc[-1]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{name} — ADHD Classification Results',
                 fontsize=18, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── AUROC ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eval_df['epoch'], eval_df['auroc'], color=cv, lw=2.5,
             marker='o', markersize=4, label='Validation')
    ax1.plot(test_df['epoch'], test_df['auroc'], color=ct, lw=2.5,
             marker='s', markersize=4, label='Test')
    ax1.axvline(x=best['epoch'], color='green', ls='--', alpha=0.6,
                label=f'Best epoch ({int(best["epoch"])})')
    ax1.set_title('AUROC over Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('AUROC')
    ax1.set_ylim([0.4, 1.0]); ax1.legend()

    # ── Accuracy ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eval_df['epoch'], eval_df['acc'], color=cv, lw=2.5,
             marker='o', markersize=4, label='Val Accuracy')
    ax2.plot(eval_df['epoch'], eval_df['balanced_acc'], color=cv, lw=2,
             ls='--', marker='o', markersize=4, label='Val Balanced Acc', alpha=0.7)
    ax2.plot(test_df['epoch'], test_df['acc'], color=ct, lw=2.5,
             marker='s', markersize=4, label='Test Accuracy')
    ax2.plot(test_df['epoch'], test_df['balanced_acc'], color=ct, lw=2,
             ls='--', marker='s', markersize=4, label='Test Balanced Acc', alpha=0.7)
    ax2.set_title('Accuracy over Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0.4, 1.0]); ax2.legend(fontsize=8)

    # ── Loss ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(eval_df['epoch'], eval_df['loss'], color=cv, lw=2.5,
             marker='o', markersize=4, label='Validation Loss')
    ax3.plot(test_df['epoch'], test_df['loss'], color=ct, lw=2.5,
             marker='s', markersize=4, label='Test Loss')
    ax3.set_title('Loss over Epochs', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
    ax3.legend()

    # ── Bar chart ──
    ax4 = fig.add_subplot(gs[1, 1])
    labels = ['AUROC', 'Accuracy', 'Balanced Acc', 'AUC-PR']
    best_vals  = [best['auroc'],  best['acc'],  best['balanced_acc'],  best['auc_pr']]
    final_vals = [final['auroc'], final['acc'], final['balanced_acc'], final['auc_pr']]

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax4.bar(x - w/2, best_vals,  w, label=f'Best Epoch ({int(best["epoch"])})',
                 color='#4CAF50', alpha=0.85, edgecolor='white')
    b2 = ax4.bar(x + w/2, final_vals, w, label=f'Final Epoch ({int(final["epoch"])})',
                 color=ct,       alpha=0.85, edgecolor='white')
    for bar in list(b1) + list(b2):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')
    ax4.set_title('Best vs Final Epoch Metrics', fontsize=13, fontweight='bold')
    ax4.set_xticks(x); ax4.set_xticklabels(labels)
    ax4.set_ylabel('Score'); ax4.set_ylim([0, 1.15]); ax4.legend()

    out = f'{name.lower().replace("-","").replace(" ","_")}_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✅ Saved: {out}')
    plt.close()

    print(f'\n=== {name} BEST EPOCH ===')
    print(f'Epoch:        {int(best["epoch"])}')
    print(f'Test AUROC:   {best["auroc"]:.4f}')
    print(f'Test ACC:     {best["acc"]:.4f}')
    print(f'Test Bal ACC: {best["balanced_acc"]:.4f}')
    print(f'Test AUC-PR:  {best["auc_pr"]:.4f}')


# ============================================================
# PLOT — combined comparison figure
# ============================================================
def plot_comparison(all_data):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison — ADHD Classification',
                 fontsize=16, fontweight='bold')

    for name, (eval_df, test_df, cv, ct) in all_data.items():
        axes[0].plot(test_df['epoch'], test_df['auroc'], lw=2.5,
                     marker='o', markersize=3, label=name)
        axes[1].plot(test_df['epoch'], test_df['acc'], lw=2.5,
                     marker='o', markersize=3, label=name)

    axes[0].set_title('Test AUROC Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('AUROC')
    axes[0].set_ylim([0.4, 1.0]); axes[0].legend()

    axes[1].set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0.4, 1.0]); axes[1].legend()

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('✅ Saved: comparison_results.png')
    plt.close()


# ============================================================
# MAIN
# ============================================================
all_data = {}
for name, cfg in MODELS.items():
    print(f'\nParsing {name}...')
    try:
        eval_df, test_df = parse_log(cfg['log'])
        print(f'  Eval rows: {len(eval_df)}, Test rows: {len(test_df)}')
        plot_model(name, eval_df, test_df, cfg['color_val'], cfg['color_test'])
        all_data[name] = (eval_df, test_df, cfg['color_val'], cfg['color_test'])
    except Exception as e:
        print(f'  ERROR: {e}')

if len(all_data) > 1:
    plot_comparison(all_data)