#!/usr/bin/env python3
"""
Extended Isolation Forest (EIF) — dependency-free NumPy implementation.

Reference
---------
Hariri S, Carrasco Kind M, Brunner R-J. Extended Isolation Forest.
IEEE Transactions on Knowledge and Data Engineering, 2021, 33(4): 1479-1489.
(arXiv:1811.02141)

Why this file exists
--------------------
The authors' reference implementation is published as the PyPI package ``eif``.
Release 2.0.2 ships only a Cython sdist, which fails to build on this machine
(``Failed building wheel for eif``; ``BackendUnavailable: Cannot import
'setuptools.build_meta'``). Release 1.0.2 is pure Python but is not
import-clean as an installed package. So the algorithm is restated here
against an explicit ``numpy.random.Generator``, which additionally makes every
number reproducible from a seed.

Fidelity is *verified*, not asserted: ``legacy_global=True`` switches all
randomness to the same sources the reference uses (Python's ``random.sample``
for subsampling, numpy's legacy global RNG for the hyperplane draws, consumed
in the same order and with the same arguments). Under that mode the scores
below match ``eif.iForest.compute_paths`` exactly — see
``tests/test_eif_equivalence.py``.

Splitting rule (per internal node), exactly as in the reference:
    n_zero = dim - extension_level - 1
    idxs   = random choice of n_zero feature indices            (drawn 1st)
    n      = normal(0, 1, dim), with n[idxs] = 0                (drawn 2nd)
    p      = uniform(mins, maxs) over the node's bounding box   (drawn 3rd)
    left   = {(x - p) . n < 0}

extension_level = 0      -> one component of n survives -> axis-parallel
                            splits, i.e. standard Isolation Forest
extension_level = dim-1  -> all components survive      -> fully extended
                            random-slope hyperplanes (used in this paper)

Path length and score are those of Isolation Forest:
    h(x) = depth(x) + c(size of the leaf that x reaches)
    s(x) = 2 ** (-mean_trees(h(x)) / c(sample_size))

Note the practical difference from scikit-learn's ``IsolationForest``:
``max_features`` there only *subsamples features per tree*; every split is
still axis-parallel, so it is NOT an Extended Isolation Forest.
"""
import random as _pyrandom

import numpy as np

EULER_GAMMA = 0.5772156649  # same constant as the reference implementation


def c_factor(n):
    """Average path length of an unsuccessful BST search (same as IF).

    Matches the reference implementation's formula literally for n >= 2.
    Note that this is *not* scikit-learn's ``_average_path_length``: sklearn
    hard-codes 1.0 at n == 2, whereas the closed form gives 0.1544. Trees here
    are built and scored with the closed form throughout, so the two must not
    be mixed.
    """
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + EULER_GAMMA) - (2.0 * (n - 1.0) / n)


class _GeneratorRNG:
    """Adapter: numpy Generator with the call signatures used in the tree."""

    def __init__(self, seed):
        self._g = np.random.default_rng(seed)

    def sample_rows(self, n_population, size):
        return self._g.choice(n_population, size=size, replace=False)

    def choice(self, n, size, replace=True):
        return self._g.choice(n, size=size, replace=replace)

    def normal(self, loc, scale, size):
        return self._g.normal(loc, scale, size)

    def uniform(self, low, high):
        return self._g.uniform(low, high)


class _LegacyGlobalRNG:
    """Adapter over the *global* legacy RNGs the reference implementation uses.

    Present only to verify equivalence with ``eif 1.0.2``; not used for the
    paper's numbers, which go through a seeded Generator.
    """

    def sample_rows(self, n_population, size):
        return _pyrandom.sample(range(n_population), size)

    def choice(self, n, size, replace=True):
        return np.random.choice(n, size, replace=replace)

    def normal(self, loc, scale, size):
        return np.random.normal(loc, scale, size)

    def uniform(self, low, high):
        return np.random.uniform(low, high)


class _Node:
    """One tree node: hyperplane (normal, intercept) or a leaf holding size."""

    __slots__ = ("normal", "intercept", "left", "right", "size")

    def __init__(self, size):
        self.normal = None
        self.intercept = None
        self.left = None
        self.right = None
        self.size = size

    @property
    def is_leaf(self):
        return self.left is None


class _EIFTree:
    def __init__(self, rng, extension_level, max_depth):
        self.rng = rng
        self.exlevel = extension_level
        self.max_depth = max_depth

    def build(self, X, depth=0):
        node = _Node(len(X))
        if depth >= self.max_depth or len(X) <= 1:
            return node

        # Draw order and arguments mirror the reference implementation so that
        # legacy_global mode reproduces its trees exactly. ``choice`` is called
        # even when it selects zero indices, because the reference does and the
        # call may advance the RNG stream.
        d = X.shape[1]
        n_zero = d - self.exlevel - 1
        idxs = np.asarray(self.rng.choice(d, max(n_zero, 0), replace=False),
                          dtype=int)
        normal = self.rng.normal(0.0, 1.0, d)
        if idxs.size:
            normal[idxs] = 0.0
        intercept = self.rng.uniform(X.min(axis=0), X.max(axis=0))

        mask = (X - intercept).dot(normal) < 0.0
        node.normal = normal
        node.intercept = intercept
        # No degenerate-split guard: when a hyperplane happens to leave one
        # side empty, the reference keeps recursing, producing a size-0 leaf on
        # the empty side and continuing to split the full set on the other.
        # Depth still terminates at max_depth, and a size-0 leaf contributes
        # c_factor(0) = 0 to the path length. Stopping early here instead would
        # change both the tree shape and the scores.
        node.left = self.build(X[mask], depth + 1)
        node.right = self.build(X[~mask], depth + 1)
        return node


def _walk(node, X, idx, out, depth):
    """Route rows ``idx`` of X down ``node``, writing path lengths into ``out``."""
    if node.is_leaf:
        if len(idx):
            out[idx] = depth + c_factor(node.size)
        return
    w = (X[idx] - node.intercept).dot(node.normal) < 0.0
    if w.any():
        _walk(node.left, X, idx[w], out, depth + 1)
    if (~w).any():
        _walk(node.right, X, idx[~w], out, depth + 1)


class ExtendedIsolationForest:
    """Extended Isolation Forest with random-slope hyperplane splits.

    Parameters
    ----------
    n_estimators : int
        Number of trees (200 in the paper's baseline, matching IF).
    sample_size : int or None
        Subsample size per tree; None means min(256, n_samples).
    extension_level : int or None
        Degree of freedom of the hyperplane slope. 0 gives axis-parallel
        splits (standard IF); d-1 gives fully extended hyperplanes. None
        selects d-1, the fully extended setting the authors recommend.
    random_state : int or None
        Seed for subsampling and for every normal vector / intercept.
    legacy_global : bool
        Route all randomness through Python's ``random`` module and numpy's
        legacy global RNG, matching ``eif 1.0.2`` exactly. Used only by the
        equivalence test; leave False for reproducible runs.
    """

    def __init__(self, n_estimators=200, sample_size=None,
                 extension_level=None, random_state=None,
                 legacy_global=False):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.random_state = random_state
        self.legacy_global = legacy_global

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.sample_size_ = int(self.sample_size or min(256, n))

        ext = self.extension_level
        if ext is None:
            ext = d - 1
        # The reference raises when 0 <= exlevel <= dim-1 is violated.
        self.extension_level_ = int(np.clip(ext, 0, d - 1))

        self.max_depth_ = int(np.ceil(np.log2(max(self.sample_size_, 2))))
        self._c_n = c_factor(self.sample_size_)

        if self.legacy_global:
            rng = _LegacyGlobalRNG()
        else:
            rng = _GeneratorRNG(self.random_state)

        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.sample_rows(n, self.sample_size_)
            tree = _EIFTree(rng, self.extension_level_, self.max_depth_)
            self.trees_.append(tree.build(X[idx]))
        return self

    def path_lengths(self, X):
        """Mean path length h(x) across trees, for each row of X."""
        X = np.ascontiguousarray(X, dtype=float)
        totals = np.zeros(len(X))
        for tree in self.trees_:
            out = np.zeros(len(X))
            _walk(tree, X, np.arange(len(X)), out, 0)
            totals += out
        return totals / len(self.trees_)

    def score_samples(self, X):
        """Anomaly score in [0, 1]; larger means more anomalous."""
        h = self.path_lengths(X)
        return 2.0 ** (-h / max(self._c_n, 1e-12))


def score_frame(df, random_state=None):
    """EIF anomaly scores for a DataFrame, oriented so larger = more anomalous.

    Convenience wrapper for the experiment scripts: object columns are
    label-encoded first (the encoding the baselines have always used), then
    the fully extended forest scores every row.

    This intentionally returns the *raw* score, not a [0,1] rescaling, to
    match what ``-IsolationForest.score_samples`` returned before. AP is
    rank-based, so the two are equivalent for every use in this repository;
    keeping the raw scale avoids surprising callers that combine scores from
    several methods.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    enc = df.copy()
    for col in enc.select_dtypes(include=['object']).columns:
        enc[col] = LabelEncoder().fit_transform(enc[col].astype(str))

    X = np.ascontiguousarray(enc.values, dtype=float)
    model = ExtendedIsolationForest(
        n_estimators=200,
        sample_size=min(256, len(X)),
        extension_level=None,          # fully extended (d-1)
        random_state=random_state,
    ).fit(X)
    return model.score_samples(X)

