#!/usr/bin/env python3
"""
Generate the Precision-Recall figure (paper Figure 2) and dump its source data.

Calibers
--------
* Curves are drawn from a single injection run (seed 42) — a PR curve is a
  per-run object, so five injections cannot be averaged into one curve.
* Legend numbers are the **5-seed (42-46) mean AP**, matching Table 2.
  `results/supplementary/fig_pr_multiseed.json` records that the AP ranking
  under seed 42 is identical to the ranking under the 5-seed mean, so the
  plotted curves represent the reported ordering.

Reproduced values (AP, 5-seed mean) — these equal Table 2:
    LLM+CTANE (0.7/0.3)   0.711
    CTANE + prior         0.705
    LLM-CFD               0.514
    CTANE (algorithm)     0.349
    EIF                   0.255
    LLM+COPOD             0.247
    LLM+EIF+COPOD         0.245
    EIF+COPOD             0.241
    COPOD                 0.230
    Random                0.181

Outputs
-------
results/supplementary/fig_pr_data.json     precision / recall per method (seed 42)
results/figures/fig2_pr_curves.png         >=300 dpi raster for the manuscript

Usage
-----
    cd experiments
    python3 src/fig_pr_multiseed.py     # once, produces fig_pr_multiseed.json
    python3 src/make_fig_pr.py
"""
import contextlib
import importlib.util
import io
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(EXP_ROOT, "results")
CACHE_LLM = os.path.join(RESULTS, "supplementary", "cache_cot", "cot_run_00.json")
MULTISEED = os.path.join(RESULTS, "supplementary", "fig_pr_multiseed.json")
OUT_DIR = os.path.join(RESULTS, "figures")
SEED = 42

# Draw order low -> high AP so the strongest curve ends up on top.
# Order matches the 5-seed means in fig_pr_multiseed.json (real EIF, AP 0.255).
STYLE = {
    "Random": ("#C8C8C8", ":", 1.0),
    "COPOD": ("#BDCBD6", "-", 1.2),
    "EIF+COPOD": ("#A9BCCB", "--", 1.2),
    "LLM+EIF+COPOD": ("#7EA2C0", "-.", 1.3),
    "LLM+COPOD": ("#4E7FA6", ":", 1.3),
    "EIF": ("#8FA6BA", "-", 1.3),
    "CTANE (algorithm)": ("#6E8CA6", "--", 1.4),
    "LLM-CFD": ("#2F5D8A", "-", 1.7),
    "CTANE + prior": ("#B9A88F", "-", 1.5),
    "LLM+CTANE (0.7/0.3)": ("#1E4468", "-", 1.9),
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_scores():
    """Rebuild every candidate method's anomaly score vector (seed 42)."""
    fx = _load_module("fx", "experiment_fixed.py")
    ct = _load_module("ct", "ctane_experiment_fixed.py")

    with contextlib.redirect_stdout(io.StringIO()):
        df, _ = fx.load_and_preprocess(fx.DATA_PATH)

    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_f = df.copy()
    for c in num:
        df_f[c] = df_f[c].astype(float)

    with contextlib.redirect_stdout(io.StringIO()):
        llm, _ = fx.statistical_validation(
            df,
            fx.parse_cfd_response(json.load(open(CACHE_LLM)), set(df.columns)),
        )

    with contextlib.redirect_stdout(io.StringIO()):
        cfds = ct.ctane_discover(df, min_support=0.05, min_confidence=0.95, max_level=2)
    for c in cfds:
        c.setdefault("condition_values", {})
        c.setdefault("expected_pattern", {})
    pure = [c for c in cfds if c["type"] == "fd"]

    fx.RANDOM_SEED = SEED
    np.random.seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        df_inj, labels = fx.inject_anomalies(df_f)
        s_llm = np.asarray(fx.compute_anomaly_scores(df_inj, llm))
        s_ct = np.asarray(fx.compute_anomaly_scores(df_inj, cfds))
        s_pure = np.asarray(fx.compute_anomaly_scores(df_inj, pure))
        X = df_inj[num].values.astype(float)
        s_eif = fx.run_eif(X, 0.18)
        s_cop = fx.run_copod(X)

    def z(x):
        x = np.asarray(x, float)
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x * 0

    counts = {
        "n_records": int(len(df)),
        "n_injected": int(labels.sum()),
        "n_llm_rules": len(llm),
        "n_ctane_rules": len(cfds),
        "n_ctane_fd_only": len(pure),
    }
    scores = {
        "Random": np.random.RandomState(0).rand(len(labels)),
        "COPOD": s_cop,
        "LLM+COPOD": 0.7 * s_llm + 0.3 * z(s_cop),
        "EIF+COPOD": (z(s_eif) + z(s_cop)) / 2,
        "LLM+EIF+COPOD": (s_llm + z(s_eif) + z(s_cop)) / 3,
        "EIF": s_eif,
        "CTANE (algorithm)": s_pure,
        "LLM-CFD": s_llm,
        "CTANE + prior": s_ct,
        "LLM+CTANE (0.7/0.3)": 0.7 * s_llm + 0.3 * s_ct,
    }
    return scores, np.asarray(labels), counts


def main():
    scores, labels, counts = build_scores()

    with open(MULTISEED, encoding="utf-8") as f:
        multiseed = json.load(f)["summary"]

    curves = {}
    curves_json = {}
    for name in STYLE:
        pr, rc, _ = precision_recall_curve(labels, scores[name])
        ap42 = float(average_precision_score(labels, scores[name]))
        # Plot from full precision so the emitted PNG is bit-reproducible;
        # round only in the JSON dump (6 decimals is far beyond audit needs
        # and keeps the file small enough to live in git beside the code).
        curves[name] = {"ap_seed42": ap42, "precision": pr, "recall": rc,
                        "ap_mean5": float(multiseed[name]["mean"]),
                        "ap_std5": float(multiseed[name]["std"])}
        curves_json[name] = {
            "ap_seed42": ap42,
            "ap_mean5": float(multiseed[name]["mean"]),
            "ap_std5": float(multiseed[name]["std"]),
            "precision": np.round(pr, 6).tolist(),
            "recall": np.round(rc, 6).tolist(),
        }
        c = curves[name]
        print(f"  {name:22s} AP(seed42)={c['ap_seed42']:.4f}  "
              f"AP(5seed)={c['ap_mean5']:.4f}+-{c['ap_std5']:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "description": "Figure 2 source data: PR curves (seed 42) + 5-seed mean AP.",
        "reproduce": "cd experiments && python3 src/fig_pr_multiseed.py && python3 src/make_fig_pr.py",
        "curve_caliber": "seed 42 single injection run",
        "legend_caliber": "AP = 5-seed (42-46) mean, matching Table 2",
        "inputs": {
            "data": "experiments/data/telco_churn.csv",
            "llm_cache": "experiments/results/supplementary/cache_cot/cot_run_00.json",
            "numpy_seed": SEED,
            "multiseed": "experiments/results/supplementary/fig_pr_multiseed.json",
        },
        "counts": counts,
        "curves": curves_json,
    }
    json_path = os.path.join(RESULTS, "supplementary", "fig_pr_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  saved: {json_path}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    # Explicit axes rect + figure-level legend: every curve crosses the
    # upper-right quadrant, so an in-axes legend would hide data. Reserving a
    # bottom strip keeps the legend clear of both curves and axis labels.
    fig = plt.figure(figsize=(5.4, 4.55))
    ax = fig.add_axes([0.105, 0.385, 0.875, 0.585])
    for name, (color, ls, lw) in STYLE.items():
        c = curves[name]
        ax.plot(c["recall"], c["precision"], color=color, lw=lw, ls=ls,
                label=f"{name} ({c['ap_mean5']:.3f})", zorder=3)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", length=3)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2,
               frameon=False, fontsize=6.6, handlelength=1.9,
               columnspacing=1.3, labelspacing=0.34)

    png_path = os.path.join(OUT_DIR, "fig2_pr_curves.png")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {png_path}")
    return png_path


if __name__ == "__main__":
    main()
