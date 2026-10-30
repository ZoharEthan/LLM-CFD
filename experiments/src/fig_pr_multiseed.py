#!/usr/bin/env python3
"""Compute 5-seed mean AP for all 10 candidate curves in Figure 2.

Writes results/supplementary/fig_pr_multiseed.json so that the figure legend
can report 5-seed means consistent with Table 2.
"""
import contextlib
import importlib.util
import io
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RES = os.path.join(EXP, "results")
SEEDS = [42, 43, 44, 45, 46]


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = load("fx", "experiment_fixed.py")
ct = load("ct", "ctane_experiment_fixed.py")

with contextlib.redirect_stdout(io.StringIO()):
    df, _ = fx.load_and_preprocess(fx.DATA_PATH)

num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
dff = df.copy()
for c in num:
    dff[c] = dff[c].astype(float)

with contextlib.redirect_stdout(io.StringIO()):
    llm, _ = fx.statistical_validation(
        df,
        fx.parse_cfd_response(
            json.load(open(os.path.join(RES, "supplementary", "cache_cot", "cot_run_00.json"))),
            set(df.columns),
        ),
    )
with contextlib.redirect_stdout(io.StringIO()):
    cfds = ct.ctane_discover(df, min_support=0.05, min_confidence=0.95, max_level=2)
for c in cfds:
    c.setdefault("condition_values", {})
    c.setdefault("expected_pattern", {})
pure = [c for c in cfds if c["type"] == "fd"]

print(f"rules: llm={len(llm)} ctane={len(cfds)} pure_fd={len(pure)}", flush=True)


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


per_seed = {}
for seed in SEEDS:
    fx.RANDOM_SEED = seed
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        inj, lab = fx.inject_anomalies(dff)
        s_llm = np.asarray(fx.compute_anomaly_scores(inj, llm))
        s_ct = np.asarray(fx.compute_anomaly_scores(inj, cfds))
        s_pure = np.asarray(fx.compute_anomaly_scores(inj, pure))
        X = inj[num].values.astype(float)
        s_eif = fx.run_eif(X, 0.18)
        s_cop = fx.run_copod(X)
    cand = {
        "LLM+CTANE (0.7/0.3)": 0.7 * s_llm + 0.3 * s_ct,
        "CTANE + prior": s_ct,
        "LLM-CFD": s_llm,
        "LLM+COPOD": 0.7 * s_llm + 0.3 * z(s_cop),
        "LLM+EIF+COPOD": (s_llm + z(s_eif) + z(s_cop)) / 3,
        "CTANE (algorithm)": s_pure,
        "EIF": s_eif,
        "EIF+COPOD": (z(s_eif) + z(s_cop)) / 2,
        "COPOD": s_cop,
        "Random": np.random.RandomState(0).rand(len(lab)),
    }
    row = {k: float(average_precision_score(lab, v)) for k, v in cand.items()}
    row["_n_pos"] = int(lab.sum())
    per_seed[str(seed)] = row
    print(f"seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if not k.startswith("_")), flush=True)

methods = [k for k in per_seed["42"] if not k.startswith("_")]
summary = {}
for m in methods:
    v = np.array([per_seed[str(s)][m] for s in SEEDS])
    summary[m] = {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)),
        "seed42": float(per_seed["42"][m]),
        "values": [round(float(x), 4) for x in v],
    }

out = {
    "description": "Figure 2 legend source: 5-seed mean AP per method (seed 42-46).",
    "reproduce": "cd experiments && python3 src/fig_pr_multiseed.py",
    "inputs": {
        "data": "experiments/data/telco_churn.csv",
        "llm_cache": "experiments/results/supplementary/cache_cot/cot_run_00.json",
        "seeds": SEEDS,
    },
    "per_seed": per_seed,
    "summary": summary,
    "ranking_consistent_seed42_vs_mean": (
        [k for k, _ in sorted(summary.items(), key=lambda x: -x[1]["seed42"])]
        == [k for k, _ in sorted(summary.items(), key=lambda x: -x[1]["mean"])]
    ),
}

p = os.path.join(RES, "supplementary", "fig_pr_multiseed.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\n=== 5-seed summary ===")
for m in sorted(methods, key=lambda x: -summary[x]["mean"]):
    s = summary[m]
    print(f"  {m:22s} mean={s['mean']:.4f} std={s['std']:.4f} seed42={s['seed42']:.4f}")
print(f"\nranking consistent: {out['ranking_consistent_seed42_vs_mean']}")
print(f"saved: {p}")
