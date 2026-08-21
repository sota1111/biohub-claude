"""Regime-conditional detection operating point (SOT-2923, default-off).

Why this exists — the *family-mix / density-mix wall*. Every attempt to move the
detection operating point (the re-detection threshold ``mad_k`` and the post-link
short-track prune ``min_track_length``) as a **single global** value has been
rejected on the mandatory 4/4 per-dataset non-regression gate: the dense,
over-detected sequences want a **harder** operating point (prune the node-budget
overshoot — SOT-2789/2912) while the sparse/clean sequences need the champion's
**recall-preserving** operating point (a harder prune deletes their real short
tracks). SOT-2912 found ``min_track_length=4`` is the *only* global point that
clears 4/4, so no global lever helps.

SOT-2921 (the cycle-5 foundation) showed the split that matters is **observable**:
a GT-free per-sequence density covariate (``median_knn_um`` / density / crowding,
computed from the detection point cloud alone) stratifies performance — but it
does **not** reproduce the *family* label (the worst sequence ``6bba_05b6850b`` is
observably the sparsest yet massively over-detected). So the actionable regime is
the **observable density label per sequence**, not the family prefix.

This module is the thin, default-off layer that turns that observation into a
policy: given a per-sequence density covariate, pick the detection operating point
``(mad_k, min_track_length)`` **conditional on the observed regime**. It is:

* **pure / deterministic** — no disk, no GT, no RNG; a function of the covariate
  value and a fitted policy only;
* **leak-free by construction** — the covariate is the observable one from
  :mod:`biohub_tracking.eval.regime` (GT-free, test-computable); the policy
  threshold and per-regime operating points are **fit on training families only**
  (leave-one-family-out) by the A/B harness, never on the held-out family;
* **default-off** — the champion config carries no ``regime_conditional_detect``
  block, so :func:`biohub_tracking.champion.regime_conditional_policy` returns
  ``None`` and the champion path is byte-for-byte unchanged. Only a promoted
  config would carry the block and invoke this layer.

The module holds the *policy representation* and its *leak-free fit* as pure
functions of already-scored per-``(family, operating-point)`` results, so unit
tests pin the behaviour without touching the heavy detection stack. The A/B
harness (``experiments/sot2923/ab_regime_conditional_detect.py``) supplies the
scored grid and the LOFO protocol.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .regime import assign_regime


class RegimeOpPoint(NamedTuple):
    """A detection operating point: adaptive threshold + short-track prune.

    ``mad_k`` is the adaptive re-detection threshold ``median + mad_k·1.4826·MAD``
    (``None`` reproduces the percentile threshold — never used here, the champion
    sets ``mad_k=3.0``). ``min_track_length`` is the post-link weakly-connected
    component prune gate. A *harder* operating point raises both.
    """

    mad_k: float
    min_track_length: int


@dataclasses.dataclass(frozen=True)
class ConditionalPolicy:
    """A per-sequence detection operating-point policy keyed on one covariate.

    A sequence whose ``covariate_key`` covariate lands in the *dense* regime gets
    ``dense_op``; the *sparse* regime gets ``sparse_op``. ``dense_is_low`` follows
    the covariate orientation (``True`` for ``median_knn_um``: tighter spacing ⇒
    denser). A sequence whose covariate is missing (NaN ⇒ ``"unknown"`` regime)
    falls back to ``sparse_op`` — the recall-preserving side — so an unclassifiable
    sequence is never pruned harder than the champion.
    """

    covariate_key: str
    threshold: float
    dense_is_low: bool
    dense_op: RegimeOpPoint
    sparse_op: RegimeOpPoint

    def regime_of(self, covariate_value: float) -> str:
        return assign_regime(
            covariate_value, self.threshold, dense_is_low=self.dense_is_low
        )

    def op_for(self, covariate_value: float) -> RegimeOpPoint:
        """Operating point for a sequence given its covariate value."""
        regime = self.regime_of(covariate_value)
        if regime == "dense":
            return self.dense_op
        # "sparse" and "unknown" both take the recall-preserving side.
        return self.sparse_op

    def to_dict(self) -> dict:
        return {
            "covariate_key": self.covariate_key,
            "threshold": self.threshold,
            "dense_is_low": self.dense_is_low,
            "dense_op": {
                "mad_k": self.dense_op.mad_k,
                "min_track_length": self.dense_op.min_track_length,
            },
            "sparse_op": {
                "mad_k": self.sparse_op.mad_k,
                "min_track_length": self.sparse_op.min_track_length,
            },
        }

    @classmethod
    def from_dict(cls, block: Mapping) -> ConditionalPolicy:
        """Build a policy from a champion-config ``regime_conditional_detect`` block.

        The block mirrors :meth:`to_dict`. Missing operating-point knobs fall back
        to the champion values (``mad_k=3.0``, ``min_track_length=4``) so a partial
        block is still safe.
        """
        d = block.get("dense_op", {}) or {}
        s = block.get("sparse_op", {}) or {}
        return cls(
            covariate_key=str(block.get("covariate_key", "median_knn_um")),
            threshold=float(block["threshold"]),
            dense_is_low=bool(block.get("dense_is_low", True)),
            dense_op=RegimeOpPoint(
                mad_k=float(d.get("mad_k", 3.0)),
                min_track_length=int(d.get("min_track_length", 4)),
            ),
            sparse_op=RegimeOpPoint(
                mad_k=float(s.get("mad_k", 3.0)),
                min_track_length=int(s.get("min_track_length", 4)),
            ),
        )


def threshold_candidates(train_values: Sequence[float]) -> list[float]:
    """Leak-free split thresholds from *training* covariate values only.

    Returns the midpoints between adjacent sorted training values (each realises a
    distinct binary dense/sparse split of the training families) plus two
    degenerate edges (``-inf`` ⇒ all-sparse, ``+inf`` ⇒ all-dense) so the fit can
    also choose "one global operating point everywhere". NaN values are dropped.
    """
    vals = sorted(v for v in train_values if not math.isnan(v))
    mids = [(a + b) / 2.0 for a, b in itertools.pairwise(vals) if b > a]
    # -inf: threshold below every value ⇒ (dense_is_low) nothing is dense ⇒ all
    # sparse. +inf: threshold above every value ⇒ everything dense. Both collapse
    # the policy to a single global operating point (champion-safe fallback).
    return [float("-inf"), *mids, float("inf")]


class FoldFit(NamedTuple):
    """The policy fitted on the training families of one LOFO fold."""

    policy: ConditionalPolicy
    train_micro_adj: float
    train_no_regression: bool
    fell_back_to_champion: bool


def _aggregate_micro_adj(rows) -> float:
    """Weighted micro-average of adjusted edge Jaccard over already-scored rows.

    Mirrors :func:`biohub_tracking.eval.cv.aggregate` micro arithmetic (weight =
    ``tp+fp+fn``) without importing the heavy scorer, so this stays pure/testable.
    """
    total_w = sum(r.weight for r in rows)
    if total_w <= 0:
        return float("nan")
    return sum(r.adj_edge_jaccard * r.weight for r in rows) / total_w


def fit_fold_policy(
    train_families: Sequence[str],
    covariate_by_family: Mapping[str, float],
    scored: Mapping[tuple[str, RegimeOpPoint], object],
    *,
    op_grid: Sequence[RegimeOpPoint],
    champion_op: RegimeOpPoint,
    champion_adj_by_family: Mapping[str, float],
    covariate_key: str = "median_knn_um",
    dense_is_low: bool = True,
    eps: float = 1e-9,
) -> FoldFit:
    """Fit the conditional policy on the training families (leak-free).

    Searches ``threshold × dense_op × sparse_op`` and picks the policy that
    **maximises the training micro adjusted Jaccard subject to per-family
    non-regression** vs the champion operating point (the exact promotion gate,
    applied on the training split). If no split-based policy clears the training
    non-regression gate, falls back to the champion operating point on both
    regimes (``fell_back_to_champion=True``) — which is trivially non-regressing
    and reproduces the champion on this fold.

    ``scored[(family, op)]`` is a pre-computed ``FamilyResult`` (adjusted/raw
    Jaccard + weight) for that family under that operating point. Only training
    families are read here; the held-out family is never touched.
    """
    train = list(train_families)
    train_vals = [covariate_by_family[f] for f in train]
    champ_policy = ConditionalPolicy(
        covariate_key=covariate_key,
        threshold=float("inf"),
        dense_is_low=dense_is_low,
        dense_op=champion_op,
        sparse_op=champion_op,
    )

    def train_rows_for(policy: ConditionalPolicy):
        return [scored[(f, policy.op_for(covariate_by_family[f]))] for f in train]

    def no_regression(rows) -> bool:
        return all(
            r.adj_edge_jaccard >= champion_adj_by_family[r.name] - eps for r in rows
        )

    best: FoldFit | None = None
    for thr in threshold_candidates(train_vals):
        for dense_op in op_grid:
            for sparse_op in op_grid:
                policy = ConditionalPolicy(
                    covariate_key=covariate_key,
                    threshold=thr,
                    dense_is_low=dense_is_low,
                    dense_op=dense_op,
                    sparse_op=sparse_op,
                )
                rows = train_rows_for(policy)
                if not no_regression(rows):
                    continue
                micro = _aggregate_micro_adj(rows)
                cand = FoldFit(
                    policy=policy,
                    train_micro_adj=micro,
                    train_no_regression=True,
                    fell_back_to_champion=(
                        dense_op == champion_op and sparse_op == champion_op
                    ),
                )
                if best is None or _is_better(cand, best, champion_op):
                    best = cand

    if best is None:
        # No split cleared the training gate — champion everywhere (safe).
        champ_rows = train_rows_for(champ_policy)
        return FoldFit(
            policy=champ_policy,
            train_micro_adj=_aggregate_micro_adj(champ_rows),
            train_no_regression=no_regression(champ_rows),
            fell_back_to_champion=True,
        )
    return best


def _is_better(cand: FoldFit, incumbent: FoldFit, champion_op: RegimeOpPoint) -> bool:
    """Higher training micro wins; ties prefer the champion-op (fewest changes)."""
    if cand.train_micro_adj > incumbent.train_micro_adj + 1e-12:
        return True
    if cand.train_micro_adj < incumbent.train_micro_adj - 1e-12:
        return False
    # Tie-break: prefer the policy that stays closest to the champion operating
    # point (deterministic, avoids gratuitous divergence from the frozen champion).
    return _n_nonchampion(cand.policy, champion_op) < _n_nonchampion(
        incumbent.policy, champion_op
    )


def _n_nonchampion(policy: ConditionalPolicy, champion_op: RegimeOpPoint) -> int:
    return int(policy.dense_op != champion_op) + int(policy.sparse_op != champion_op)
