#!/usr/bin/env python3
"""
Regenerate ALL paper figures matching reference artboard colors.
=========================================================================
Figures:
  1. fig2_f1_comparison  — F1 bar chart
  2. fig3_auprc_comparison — AUPRC bar chart (12 methods)
  3. fig4_stability_boxplot — F1 & AUPRC stability (Few-shot + CoT+fewshot)
  4. fig5_ctane_comparison — CTANE F1 across hyperparams
  5. fig6_pr_curve — PR curves (9 curves + random baseline)
"""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['lines.linewidth'] = 1.0

BASE = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn"
RESULTS_DIR = os.path.join(BASE, "experiments/results")
FIGURES_PATH = os.path.join(BASE, "experiments/results/figures")
os.makedirs(FIGURES_PATH, exist_ok=True)

# Exact palette from reference artboards (Standard matplotlib cycle)
CL = {
    # PR curve line colors
    'pr_lm':        '#c44e52',   # red     — LLM-CFDMiner (thickest)
    'pr_llm':       '#4c72b0',   # blue    — LLM-CFD
    'pr_cfdminer':  '#55a467',   # green   — CFDMiner
    'pr_ctane':     '#ccb974',   # olive   — CTANE
    'pr_3way':      '#8172b2',   # purple  — LLM-CFD+EIF+COPOD
    'pr_copod':     '#64b5cd',   # cyan    — LLM-CFD+COPOD
    'pr_eif':       '#dd8452',   # orange  — EIF
    'pr_copod2':    '#937860',   # brown   — COPOD
    'pr_eif_copod': '#8c8c8c',   # gray    — EIF+COPOD
    'pr_random':    '#bbbbbb',   # lt gray — Random
    # Bar chart fills (matching reference order)
    'bar_lm_avg':   '#da8bc3',   # pink    — LLM-CFDMiner(avg)
    'bar_lm_filt':  '#a98679',   # rust    — LLM-CFDMiner(filtered)
    'bar_llm':      '#937860',   # brown   — LLM-CFD
    'bar_copod_w':  '#b0b0b0',   # gray    — LLM-CFD+COPOD(w)
    'bar_eif_w':    '#ccb974',   # olive   — LLM-CFD+EIF(w)
    'bar_cfdminer': '#937860',   # brown   — CFDMiner (hatched)
    'bar_3way':     '#937860',   # brown   — LLM-CFD+EIF+COPOD (hatched)
    'bar_copod_s':  '#64b5cd',   # cyan    — LLM-CFD+COPOD(simple)
    'bar_ctane':    '#da8bc3',   # pink    — CTANE (hatched)
    'bar_eif':      '#ccb974',   # olive   — EIF (hatched)
    'bar_eif_copod':'#b0b0b0',   # gray    — EIF+COPOD (hatched)
    'bar_copod':    '#a98679',   # rust    — COPOD (hatched)
    # Stability boxplot
    'box_fs':       '#8fa8c4',   # lt blue — Few-shot
    'box_cot':      '#4c72b0',   # blue    — CoT+fewshot
    # F1 bar chart
    'f1_llm':       '#8fa8c4',   # lt blue — LLM-CFD
    'f1_ctane':     '#8c8c8c',   # gray    — CTANE (hatched)
    'f1_cfdminer':  '#8c8c8c',   # gray    — CFDMiner (hatched)
    # CTANE comparison
    'ct_best':      '#c44e52',   # red     — best config
    'ct_other':     '#e8dcc8',   # lt beige— other configs
    'ct_ref':       '#555555',   # dk gray — ref lines
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Load data
telco     = load_json(os.path.join(RESULTS_DIR, "unified/unified_results.json"))
adult     = load_json(os.path.join(RESULTS_DIR, "unified/adult_unified_results.json"))
ctane     = load_json(os.path.join(RESULTS_DIR, "ctane/ctane_experiment_results.json"))
all_pr    = load_json(os.path.join(RESULTS_DIR, "supplementary/all_methods_pr_data.json"))
lc_pr     = load_json(os.path.join(RESULTS_DIR, "supplementary/llm_cfdminer_pr_data.json"))
fusion_pr = load_json(os.path.join(RESULTS_DIR, "supplementary/llm_fusion_pr_data.json"))
fewshot   = load_json(os.path.join(RESULTS_DIR, "supplementary/fewshot_stability.json"))

# ============================================================
# FIG 2: F1 Comparison
# ============================================================
def generate_fig2_f1_comparison():
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = [
        ('LLM-CFD\n(CoT+fewshot)',  0.7368, CL['f1_llm']),
        ('CTANE',                    0.0276, CL['f1_ctane']),
        ('CFDMiner',                 0.0033, CL['f1_cfdminer']),
    ]
    labels, values, colors = [m[0] for m in methods], [m[1] for m in methods], [m[2] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.4, width=0.55)
    for i, (name, _, _) in enumerate(methods):
        if 'CTANE' in name or 'CFDMiner' in name:
            bars[i].set_hatch('//')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    legend_elements = [
        Patch(facecolor=CL['f1_llm'], edgecolor='white', label='LLM methods'),
        Patch(facecolor=CL['f1_ctane'], edgecolor='white', hatch='//', label='Statistical CFD discovery'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, edgecolor='#cccccc', fancybox=False, fontsize=8)
    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig2_f1_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIG 3: AUPRC Comparison (12 methods)
# ============================================================
def generate_fig3_auprc_comparison():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    methods = [
        ('LLM-CFDMiner\n(avg)',       0.5614, CL['bar_lm_avg']),
        ('LLM-CFDMiner\n(filtered)',  0.5595, CL['bar_lm_filt']),
        ('LLM-CFD',                    0.4953, CL['bar_llm']),
        ('LLM-CFD+COPOD\n(w)',          0.4370, CL['bar_copod_w']),
        ('LLM-CFD+EIF\n(w)',           0.4060, CL['bar_eif_w']),
        ('CFDMiner',                   0.3867, CL['bar_cfdminer']),
        ('LLM-CFD+EIF\n+COPOD',        0.3837, CL['bar_3way']),
        ('LLM-CFD+COPOD\n(simple)',    0.3089, CL['bar_copod_s']),
        ('CTANE',                      0.2721, CL['bar_ctane']),
        ('EIF',                        0.2051, CL['bar_eif']),
        ('EIF+COPOD',                  0.1924, CL['bar_eif_copod']),
        ('COPOD',                      0.1667, CL['bar_copod']),
    ]
    labels, values, colors = [m[0] for m in methods], [m[1] for m in methods], [m[2] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.4, width=0.65)
    for i, (name, _, _) in enumerate(methods):
        if name in ('EIF', 'COPOD', 'CFDMiner', 'CTANE', 'EIF+COPOD'):
            bars[i].set_hatch('//')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.010,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color='#333333', rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('AUPRC', fontsize=11)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    legend_elements = [
        Patch(facecolor=CL['bar_llm'], edgecolor='white', label='LLM-CFD'),
        Patch(facecolor=CL['bar_cfdminer'], edgecolor='white', hatch='//', label='CFD discovery baselines'),
        Patch(facecolor=CL['bar_eif'], edgecolor='white', hatch='//', label='Statistical baselines'),
        Patch(facecolor=CL['bar_lm_avg'], edgecolor='white', label='Fusion methods'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, edgecolor='#cccccc', fancybox=False, fontsize=7)
    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig3_auprc_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIG 4: Stability Boxplot
# ============================================================
def generate_fig4_stability_boxplot():
    cot_f1 = [r['f1'] for r in telco['individual_runs']['per_run']]
    cot_auprc = [r['auprc'] for r in telco['individual_runs']['per_run']]
    few_f1 = [r['f1'] for r in fewshot['per_run']]
    few_auprc = [r['auprc'] for r in fewshot['per_run']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.5))
    rng = np.random.RandomState(42)

    def boxplot_setup(ax, data1, data2, ylabel, title, hscv_val):
        bp1 = ax.boxplot([data1], positions=[1], widths=0.35, patch_artist=True,
                         boxprops=dict(facecolor=CL['box_fs'], alpha=0.25, edgecolor=CL['box_fs'], linewidth=0.6),
                         medianprops=dict(color=CL['box_fs'], linewidth=1.2),
                         whiskerprops=dict(color=CL['box_fs'], linewidth=0.6),
                         capprops=dict(color=CL['box_fs'], linewidth=0.6),
                         flierprops=dict(marker='o', markerfacecolor=CL['box_fs'], markersize=4, markeredgecolor='none', alpha=0.7))
        x1 = 1 + rng.normal(0, 0.025, len(data1))
        ax.scatter(x1, data1, color=CL['box_fs'], alpha=0.5, s=14, zorder=3, edgecolors='white', linewidths=0.3)

        bp2 = ax.boxplot([data2], positions=[2], widths=0.35, patch_artist=True,
                         boxprops=dict(facecolor=CL['box_cot'], alpha=0.25, edgecolor=CL['box_cot'], linewidth=0.8),
                         medianprops=dict(color=CL['box_cot'], linewidth=1.5),
                         whiskerprops=dict(color=CL['box_cot'], linewidth=0.8),
                         capprops=dict(color=CL['box_cot'], linewidth=0.8),
                         flierprops=dict(marker='o', markerfacecolor=CL['box_cot'], markersize=4, markeredgecolor='none', alpha=0.7))
        x2 = 2 + rng.normal(0, 0.025, len(data2))
        ax.scatter(x2, data2, color=CL['box_cot'], alpha=0.5, s=14, zorder=3, edgecolors='white', linewidths=0.3)

        ax.axhline(y=hscv_val, color=CL['ct_ref'], linestyle='--', linewidth=0.8,
                   label=f'HSCV ({hscv_val:.3f})')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Few-shot\n(non-deterministic)', 'CoT+few-shot\n(deterministic)'], fontsize=7.5)
        ax.legend(fontsize=7, framealpha=0.9, edgecolor='#cccccc', fancybox=False, loc='lower right')
        ax.grid(axis='y', alpha=0.2, linewidth=0.4)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        m1 = np.median(data1); i1 = np.subtract(*np.percentile(data1, [75, 25]))
        m2 = np.median(data2); i2 = np.subtract(*np.percentile(data2, [75, 25]))
        ax.text(0.65, m1 + 0.02, f'med={m1:.3f}\nIQR={i1:.3f}', fontsize=7, va='bottom', color='#555555')
        ax.text(2.35, m2, f'med={m2:.3f}\nIQR={i2:.3f}', fontsize=7, va='center', color='#555555')

    boxplot_setup(ax1, few_f1, cot_f1, 'F1 Score', 'F1 Stability', 0.500)
    boxplot_setup(ax2, few_auprc, cot_auprc, 'AUPRC', 'AUPRC Stability', 0.730)

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig4_stability_boxplot.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIG 5: CTANE F1 Comparison
# ============================================================
def generate_fig5_ctane_comparison():
    tc = ctane.get('telco', {}).get('all_configs', [])
    telco_f1 = [c['f1'] for c in tc]; telco_best = int(np.argmax(telco_f1))
    ac = ctane.get('adult', {}).get('all_configs', [])
    adult_f1 = [c['f1'] for c in ac]; adult_best = int(np.argmax(adult_f1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    width = 0.55

    # Telco
    colors_t = [CL['ct_best'] if i == telco_best else CL['ct_other'] for i in range(len(telco_f1))]
    bars_t = ax1.bar(np.arange(len(telco_f1)), telco_f1, width, color=colors_t, edgecolor='white', linewidth=0.4)
    for i, bar in enumerate(bars_t):
        if i != telco_best: bar.set_hatch('///')
    telco_labels = [f"s={c['min_support']:.3f}\nc={c['min_confidence']:.2f}" for c in tc]
    ax1.set_xticks(np.arange(len(telco_f1))); ax1.set_xticklabels(telco_labels, fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title(f'Telco Churn (GT=15)\nBest CTANE: F1={max(telco_f1):.3f}', fontweight='bold', fontsize=10)
    ax1.axhline(y=0.737, color=CL['pr_llm'], linestyle='--', linewidth=0.7, label='LLM-CFD+HSCV (0.737)')
    ax1.axhline(y=0.737, color=CL['pr_llm'], linestyle=':', linewidth=0.6, alpha=0.5, label='LLM-CFD best (0.737)')
    ax1.legend(fontsize=8, framealpha=0.9, edgecolor='#cccccc', fancybox=False)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.annotate(f'{max(telco_f1):.3f}', xy=(telco_best, max(telco_f1)),
                 xytext=(telco_best, max(telco_f1)+0.008), ha='center', fontsize=8, fontweight='bold', color=CL['ct_best'])

    # Adult
    colors_a = [CL['ct_best'] if i == adult_best else CL['ct_other'] for i in range(len(adult_f1))]
    bars_a = ax2.bar(np.arange(len(adult_f1)), adult_f1, width, color=colors_a, edgecolor='white', linewidth=0.4)
    for i, bar in enumerate(bars_a):
        if i != adult_best: bar.set_hatch('///')
    adult_labels = [f"s={c['min_support']:.3f}\nc={c['min_confidence']:.2f}" for c in ac]
    ax2.set_xticks(np.arange(len(adult_f1))); ax2.set_xticklabels(adult_labels, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title(f'UCI Adult (GT=12)\nBest CTANE: F1={max(adult_f1):.3f}', fontweight='bold', fontsize=10)
    alf1 = max(r['f1'] for r in adult['individual_runs']['per_run'])
    ax2.axhline(y=alf1, color=CL['pr_llm'], linestyle='--', linewidth=0.7, label=f'LLM-CFD+HSCV ({alf1:.3f})')
    ax2.axhline(y=alf1, color=CL['pr_llm'], linestyle=':', linewidth=0.6, alpha=0.5, label=f'LLM-CFD best ({alf1:.3f})')
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor='#cccccc', fancybox=False)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.annotate(f'{max(adult_f1):.3f}', xy=(adult_best, max(adult_f1)),
                 xytext=(adult_best, max(adult_f1)+0.005), ha='center', fontsize=8, fontweight='bold', color=CL['ct_best'])

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig5_ctane_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIG 6: PR Curve (9 curves)
# ============================================================
def generate_fig6_pr_curve():
    fig, ax = plt.subplots(figsize=(7, 4.8))

    curves = [
        (lc_pr,                        'LLM-CFDMiner',              CL['pr_lm'],        2.0, '-'),
        (all_pr['LLM-CFD'],            'LLM-CFD',                   CL['pr_llm'],       1.5, '-'),
        (all_pr['CFDMiner'],           'CFDMiner',                  CL['pr_cfdminer'],  1.0, '--'),
        (all_pr['CTANE'],              'CTANE',                     CL['pr_ctane'],     1.0, '--'),
        (fusion_pr,                    'LLM-CFD+EIF+COPOD',         CL['pr_3way'],      1.2, '-.'),
        (all_pr['LLM-CFD+COPOD'],      'LLM-CFD+COPOD (simple)',    CL['pr_copod'],     1.0, ':'),
        (all_pr['EIF'],                'EIF',                       CL['pr_eif'],       1.0, '--'),
        (all_pr['COPOD'],              'COPOD',                     CL['pr_copod2'],    1.0, '--'),
        (all_pr['EIF+COPOD'],          'EIF+COPOD',                 CL['pr_eif_copod'], 0.8, '--'),
    ]

    zorder = len(curves)
    for data, name, color, lw, ls in curves:
        zorder -= 1
        rec = data['recall'] if isinstance(data, dict) else data.get('recall', [])
        prec = data['precision'] if isinstance(data, dict) else data.get('precision', [])
        auprc = data['auprc'] if isinstance(data, dict) else data.get('auprc', 0)
        label = f"{name} ({auprc:.3f})"
        ax.plot(rec, prec, color=color, linewidth=lw, linestyle=ls, label=label, zorder=zorder)

    ax.axhline(y=0.18, color=CL['pr_random'], linestyle=':', linewidth=0.6,
               label='Random (0.180)', zorder=1)

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7,
              framealpha=0.95, edgecolor='#cccccc', fancybox=False, borderpad=0.5, labelspacing=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig6_pr_curve.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

def main():
    print("=" * 60)
    print("REGENERATING ALL PAPER FIGURES — REFERENCE COLORS")
    print("=" * 60)
    print(f"\n[1/5] Fig2: F1 Comparison")
    generate_fig2_f1_comparison()
    print(f"\n[2/5] Fig3: AUPRC Comparison")
    generate_fig3_auprc_comparison()
    print(f"\n[3/5] Fig4: Stability Boxplot")
    generate_fig4_stability_boxplot()
    print(f"\n[4/5] Fig5: CTANE Comparison")
    generate_fig5_ctane_comparison()
    print(f"\n[5/5] Fig6: PR Curve")
    generate_fig6_pr_curve()
    print("\nALL 5 FIGURES GENERATED SUCCESSFULLY")

if __name__ == "__main__":
    main()
