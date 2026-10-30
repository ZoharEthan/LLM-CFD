#!/usr/bin/env python3
"""
Equivalence test: src/eif_numpy.py vs the authors' reference implementation.

The reference is the PyPI package ``eif`` version 1.0.2 (Hariri & Carrasco
Kind), the same code that accompanies IEEE TKDE 2021, 33(4):1479-1489. That
release is a pure-Python wheel, so it can be installed without a compiler:

    pip install eif==1.0.2

``eif_numpy.ExtendedIsolationForest(legacy_global=True)`` routes all randomness
through the same sources as the reference (Python's ``random.sample`` for
subsampling, numpy's legacy global RNG for the hyperplane draws, in the same
order and with the same arguments). Under that mode the two must agree
exactly, which is what makes this file a real check rather than a smoke test.

Run directly (no pytest required):

    cd experiments && python3 tests/test_eif_equivalence.py
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from eif_numpy import ExtendedIsolationForest  # noqa: E402


def _reference_module():
    try:
        import eif as ref  # noqa: F401
        return ref
    except Exception as exc:  # pragma: no cover - dependency absent
        print(f"SKIP: reference package 'eif' unavailable ({exc}).")
        print("      Install it with: pip install eif==1.0.2")
        return None


def test_bitwise_equivalence():
    ref = _reference_module()
    if ref is None:
        return None
    X = np.random.RandomState(0).rand(400, 6)
    results = []
    for ext in (0, 2, 5):
        np.random.seed(2024)
        random.seed(2024)
        s_ref = ref.iForest(X, ntrees=25, sample_size=96,
                            ExtensionLevel=ext).compute_paths(X)

        np.random.seed(2024)
        random.seed(2024)
        mine = ExtendedIsolationForest(n_estimators=25, sample_size=96,
                                       extension_level=ext,
                                       legacy_global=True).fit(X)
        s_mine = mine.score_samples(X)

        identical = np.array_equal(s_ref, s_mine)
        max_diff = float(np.max(np.abs(s_ref - s_mine)))
        results.append((ext, identical, max_diff))
        print(f"extension_level={ext}: bit-identical={identical} "
              f"max|diff|={max_diff:.3e}")
        assert identical, (f"extension_level={ext} diverges from the "
                           f"reference by {max_diff:.3e}")
    return results


def test_extension_level_zero_is_axis_parallel():
    """extension_level=0 must reproduce standard (axis-parallel) IF splits."""
    X = np.random.RandomState(1).rand(200, 5)
    model = ExtendedIsolationForest(n_estimators=10, sample_size=64,
                                    extension_level=0, random_state=0).fit(X)
    for tree in model.trees_:
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                continue
            # Exactly one non-zero slope component == axis-parallel hyperplane.
            assert np.count_nonzero(node.normal) == 1
            stack.extend([node.left, node.right])
    print("extension_level=0: all splits axis-parallel")


def test_score_orientation():
    """Scores must rise as points move away from the bulk of the data."""
    rng = np.random.RandomState(2)
    X = rng.normal(0, 1, (600, 4))
    model = ExtendedIsolationForest(n_estimators=100, sample_size=256,
                                    extension_level=3, random_state=0).fit(X)
    near = np.zeros((1, 4))
    far = np.full((1, 4), 6.0)
    s_near = model.score_samples(near)[0]
    s_far = model.score_samples(far)[0]
    print(f"score at centre={s_near:.4f}, at corner={s_far:.4f}")
    assert s_far > s_near


if __name__ == "__main__":
    test_bitwise_equivalence()
    test_extension_level_zero_is_axis_parallel()
    test_score_orientation()
    print("\nall EIF tests passed")
