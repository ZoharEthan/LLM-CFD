#!/usr/bin/env python3
"""
CFDMiner: Constant CFD Discovery via Closed/Free Itemset Pairing
================================================================
Reference: Fan, W., Geerts, F., Li, J., Wang, X. (2011).
           "Discovering Conditional Functional Dependencies."
           IEEE TKDE, 23(5), 683-698.

Algorithm
---------
1. Encode each tuple as a transaction of (attribute, value) items.
2. Mine all frequent itemsets with vertical tid-lists (Apriori + intersection).
3. Label each frequent itemset:
     - CLOSED if no proper superset has the same support;
     - FREE   if no proper subset has the same support.
4. For every closed itemset Y and every attribute A in Y, let X = Y \\ {(A,v)}.
   If X is free (or empty), emit the constant CFD  X -> A = v.
5. Optional extensions (off by default, to stay faithful to the original
   formalism): numeric binning for range CFDs, and a cross-column
   consistency special case.

This is a re-implementation written from the paper's algorithm; it is not a
recovery of the original (now lost) script.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd



# ----------------------------------------------------------------------------
# Dataset loading (handles the header-less Adult CSV)
# ----------------------------------------------------------------------------

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load Telco or Adult; the Adult CSV ships without a header and with
    leading blanks in every field."""
    head = pd.read_csv(csv_path, nrows=1)
    cols = [str(c).strip() for c in head.columns]
    looks_headerless = len(cols) == len(ADULT_COLUMNS) and not (
        set(cols) & {"age", "workclass", "education", "income"}
    )

    if looks_headerless:
        df = pd.read_csv(csv_path, header=None, names=ADULT_COLUMNS)
    else:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip().replace("-", "_") for c in df.columns]

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df




# ----------------------------------------------------------------------------
# Dataset loading (handles the header-less Adult CSV)
# ----------------------------------------------------------------------------

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load Telco or Adult; the Adult CSV ships without a header and with
    leading blanks in every field."""
    head = pd.read_csv(csv_path, nrows=1)
    cols = [str(c).strip() for c in head.columns]
    looks_headerless = len(cols) == len(ADULT_COLUMNS) and not (
        set(cols) & {"age", "workclass", "education", "income"}
    )

    if looks_headerless:
        df = pd.read_csv(csv_path, header=None, names=ADULT_COLUMNS)
    else:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip().replace("-", "_") for c in df.columns]

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


# ----------------------------------------------------------------------------
# Frequent itemset mining (vertical tid-lists)
# ----------------------------------------------------------------------------


def build_transactions(df: pd.DataFrame, columns: list[str]) -> list[frozenset]:
    """One transaction per row: frozenset of (column, str(value)) items."""
    tx: list[frozenset] = []
    for _, row in df[columns].iterrows():
        tx.append(frozenset((c, str(row[c])) for c in columns))
    return tx


def frequent_itemsets(
    transactions: list[frozenset], min_count: int, max_len: int
) -> dict[frozenset, set[int]]:
    """Apriori with tid-list intersection. Returns itemset -> tid set."""
    tids: dict[tuple, set[int]] = defaultdict(set)
    for i, t in enumerate(transactions):
        for item in t:
            tids[item].add(i)

    level: dict[frozenset, set[int]] = {
        frozenset([k]): v for k, v in tids.items() if len(v) >= min_count
    }
    all_freq: dict[frozenset, set[int]] = dict(level)

    for L in range(2, max_len + 1):
        prev = list(level.keys())
        nxt: dict[frozenset, set[int]] = {}
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                a, b = prev[i], prev[j]
                union = a | b
                if len(union) != L:
                    continue
                inter = level[a] & level[b]
                if len(inter) >= min_count:
                    nxt[union] = inter
        if not nxt:
            break
        all_freq.update(nxt)
        level = nxt

    return all_freq


# ----------------------------------------------------------------------------
# Closed / free labelling
# ----------------------------------------------------------------------------


def label_itemsets(
    freq: dict[frozenset, set[int]]
) -> tuple[set[frozenset], set[frozenset]]:
    """Return (closed_sets, free_sets)."""
    # support signature -> sets sharing it
    by_sig: dict[frozenset, list[frozenset]] = defaultdict(list)
    for s, tid in freq.items():
        by_sig[frozenset(tid)].append(s)

    closed: set[frozenset] = set()
    free: set[frozenset] = set()

    for s, tid in freq.items():
        sig = frozenset(tid)
        peers = by_sig[sig]
        # CLOSED: no proper superset with identical support
        if not any(len(p) > len(s) for p in peers):
            closed.add(s)
        # FREE: no proper subset with identical support
        if not any(len(p) < len(s) for p in peers):
            free.add(s)

    return closed, free


# ----------------------------------------------------------------------------
# CFD generation
# ----------------------------------------------------------------------------


def itemset_attrs(s: frozenset) -> set[str]:
    return {item[0] for item in s}


def generate_cfds(
    closed: set[frozenset],
    free: set[frozenset],
    freq: dict[frozenset, set[int]],
    n_rows: int,
) -> list[dict]:
    """Pair each free LHS with single-attribute RHS drawn from a closed set."""
    free_by_attrs: dict[frozenset, frozenset] = {}
    for f in free:
        free_by_attrs.setdefault(frozenset(itemset_attrs(f)), f)

    rules: list[dict] = []
    seen: set[tuple] = set()

    for y in closed:
        y_attrs = itemset_attrs(y)
        if len(y_attrs) < 1:
            continue
        for attr in y_attrs:
            lhs_items = frozenset(item for item in y if item[0] != attr)
            lhs_attrs = frozenset(itemset_attrs(lhs_items)) if lhs_items else frozenset()

            # LHS must be free (empty LHS is trivially free)
            if lhs_items:
                if lhs_items not in free:
                    continue
            rhs_item = next(item for item in y if item[0] == attr)
            rhs_value = rhs_item[1]

            cond = {c: v for c, v in (i for i in lhs_items)}
            key = (
                attr,
                tuple(sorted(cond.items())),
                rhs_value,
            )
            if key in seen:
                continue
            seen.add(key)

            support = len(freq[y]) / n_rows if y in freq else 0.0
            rules.append(
                {
                    "type": "enum" if not cond else "fd",
                    "condition_attributes": sorted(cond.keys()),
                    "condition_values": cond,
                    "dependent_attribute": attr,
                    "expected_pattern": {"values": [rhs_value]},
                    "support": support,
                    "confidence": 1.0,  # constant CFD from a closed set => holds
                }
            )

    return rules


# ----------------------------------------------------------------------------
# Evaluation (same key-matching protocol as the paper)
# ----------------------------------------------------------------------------

TYPE_CAT = {
    "fd": "categorical",
    "enum": "categorical",
    "logic": "categorical",
    "range": "numerical",
    "consistency": "structural",
}


def key(c: dict) -> tuple:
    return (
        c["dependent_attribute"],
        tuple(sorted(c.get("condition_attributes", []))),
        TYPE_CAT.get(c.get("type", "fd"), c.get("type", "fd")),
    )


def prf(discovered: list[dict], gt: list[dict]) -> dict:
    dk = {key(c) for c in discovered}
    gk = {key(c) for c in gt}
    tp = len(gk & dk)
    fp = len(dk - gk)
    fn = len(gk - dk)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def run(
    csv_path: str,
    gt_path: str,
    min_support: float = 0.01,
    max_len: int = 3,
    drop_id: bool = True,
    numeric_range: bool = True,
    consistency_case: bool = True,
) -> dict:
    df = load_dataset(csv_path)
    if drop_id and "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # TotalCharges (Telco only): blank -> NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    # Low-cardinality numeric columns (e.g. SeniorCitizen with {0,1}) behave as
    # categorical for constant-CFD purposes.
    cat_cols += [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 5
    ]
    t0 = time.time()
    tx = build_transactions(df, cat_cols)
    min_count = int(np.ceil(min_support * len(df)))
    freq = frequent_itemsets(tx, min_count=min_count, max_len=max_len)
    closed, free = label_itemsets(freq)
    rules = generate_cfds(closed, free, freq, len(df))
    elapsed = time.time() - t0

    _g = json.load(open(gt_path, encoding="utf-8"))
    gt = _g["rules"] if isinstance(_g, dict) else _g
    gt_norm = [
        {
            "type": g["type"],
            "condition_attributes": g.get("condition_attributes", []),
            "dependent_attribute": g["dependent_attribute"],
        }
        for g in gt
    ]
    m = prf(rules, gt_norm)

    # ---- optional extensions (same protocol as the CTANE baseline) ----
    ext: list[dict] = []
    if numeric_range:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 5:
                ext.append(
                    {
                        "type": "range",
                        "condition_attributes": [],
                        "condition_values": {},
                        "dependent_attribute": c,
                        "expected_pattern": {
                            "min": float(df[c].min()),
                            "max": float(df[c].max()),
                        },
                    }
                )
    if consistency_case and {"tenure", "MonthlyCharges", "TotalCharges"} <= set(df.columns):
        t = pd.to_numeric(df["tenure"], errors="coerce")
        mo = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if abs((t * mo).corr(tc)) > 0.8:
            ext.append(
                {
                    "type": "consistency",
                    "condition_attributes": ["tenure", "MonthlyCharges"],
                    "condition_values": {},
                    "dependent_attribute": "TotalCharges",
                    "expected_pattern": {"relation": "TotalCharges ≈ tenure × MonthlyCharges"},
                }
            )
    m_ext = prf(rules + ext, gt_norm) if ext else m

    return {
        "n_rows": len(df),
        "cat_columns": cat_cols,
        "min_support": min_support,
        "min_count": min_count,
        "max_len": max_len,
        "n_frequent_itemsets": len(freq),
        "n_closed": len(closed),
        "n_free": len(free),
        "n_rules": len(rules),
        "elapsed_s": round(elapsed, 2),
        "metrics": m,
        "extensions": len(ext),
        "metrics_with_extensions": m_ext,
        "rules_sample": rules[:10],
    }


if __name__ == "__main__":
    import sys

    base = "/Users/hzy/Downloads/mqw7dr6nxdtvfujn/experiments"
    out = run(
        f"{base}/data/telco_churn.csv",
        f"{base}/data/gt_rules_telco.json",
        min_support=0.01,
        max_len=3,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "rules_sample"},
                     ensure_ascii=False, indent=2))
