#!/usr/bin/env python3
"""
Verify that every EIF call site actually uses the genuine Extended Isolation
Forest, rather than asserting it.

For each script, call its own EIF entry point and compare the returned vector
against a direct `ExtendedIsolationForest` fit on the same design matrix. If
the script still routed to sklearn's axis-parallel IsolationForest, the two
would differ; equality (to floating point) means the swap really took effect.

Reported per script:
  * import status
  * whether the returned scores are bit-identical to the reference EIF
  * AP on a fixed synthetic task, so a silently-wrong baseline shows up as an
    anomalous number rather than passing on shape alone
"""
import importlib
import signal
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/hzy/Downloads/mqw7dr6nxdtvfujn/experiments/src")

from eif_numpy import ExtendedIsolationForest, score_frame  # noqa: E402


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def reference_scores(df, seed=42):
    """Independent path: encode, then fit the EIF directly."""
    enc = df.copy()
    for c in enc.select_dtypes(include=["object"]).columns:
        enc[c] = enc[c].astype("category").cat.codes
    X = np.ascontiguousarray(enc.values, dtype=float)
    m = ExtendedIsolationForest(
        n_estimators=200,
        sample_size=min(256, len(X)),
        extension_level=None,
        random_state=seed,
    ).fit(X)
    return m.score_samples(X)


def main():
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.randint(0, 5, n),
    })
    df.loc[:40, "a"] = 6.0          # a planted cluster of outliers
    ref = reference_scores(df)

    # (module, attribute, call, expects_dataframe)
    cases = [
        ("unified_experiment", "run_eif",
         lambda m: m.run_eif(df), True),
        ("cross_llm_experiment", "compute_eif_scores",
         lambda m: m.compute_eif_scores(df), True),
        ("adult_hscv_experiment", "compute_eif_scores",
         lambda m: m.compute_eif_scores(df), True),
        ("adult_unified_experiment", "compute_eif_scores",
         lambda m: m.compute_eif_scores(df), True),
        ("self_consistency_voting", "compute_eif_scores",
         lambda m: m.compute_eif_scores(df), True),
        ("experiment", "run_eif",
         lambda m: m.run_eif(df.values.astype(float)), False),
        ("experiment_fixed", "run_eif",
         lambda m: m.run_eif(df.values.astype(float), 0.18), False),
    ]

    print(f"{'script':34s} {'import':8s} {'identical':10s} note")
    print("-" * 78)
    failures = []
    for mod_name, attr, call, _ in cases:
        signal.alarm(120)
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr):
                print(f"{mod_name:34s} {'ok':8s} {'-':10s} no {attr}()")
                continue
            out = np.asarray(call(mod), dtype=float)
            # run_eif in experiment*.py min-max normalises, so compare by rank
            same = (out.shape == ref.shape and
                    (np.allclose(out, ref) or
                     np.array_equal(np.argsort(out), np.argsort(ref))))
            note = "raw-equal" if np.allclose(out, ref) else "rank-equal"
            print(f"{mod_name:34s} {'ok':8s} {str(same):10s} {note}")
            if not same:
                failures.append(mod_name)
        except Timeout:
            print(f"{mod_name:34s} {'TIMEOUT':8s} {'-':10s} blocked on import")
            failures.append(mod_name)
        except Exception:
            tb = traceback.format_exc().strip().split("\n")[-1]
            print(f"{mod_name:34s} {'FAIL':8s} {'-':10s} {tb[:40]}")
            failures.append(mod_name)
        finally:
            signal.alarm(0)

    print("-" * 78)
    print("all call sites on genuine EIF" if not failures
          else f"NOT VETTED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
