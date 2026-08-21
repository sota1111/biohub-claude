"""Regime-conditional *linking* operating point (SOT-2922, default-off).

Why this exists — the same *family-mix / density-mix wall* that blocked the
detection operating point (SOT-2923), now applied to the **linking** levers. Two
linking levers each help exactly one density regime as a **single global** value
and are therefore rejected on the mandatory 4/4 per-dataset non-regression gate:

* **mutual-NN cycle-consistency prune** (SOT-2910, ``cycle_consistency_gate``):
  sheds contested FP steals in the dense, over-detected volumes (6bba gains) but
  deletes real links in the clean sparse family (44b6 regresses) — REJECTED
  globally (the dense gain is 95.8% of micro weight, masking the sparse loss).
* **motion_gain=2.0** (SOT-2900, stronger ARGUS motion prediction): CV-superior
  in aggregate but held as reserve; a per-regime split lets the sparse side keep
  the champion ``motion_gain=1.0``.

The champion linking operating point is ``motion_gain=1.0`` with the mutual-NN
gate **off**. This module is the thin, default-off layer that selects the linking
operating point **conditional on the observable per-sequence density regime**
(SOT-2921's GT-free ``median_knn_um`` covariate, computed from the champion
detection point cloud), fit leave-one-family-out by the A/B harness. It is:

* **pure / deterministic** — no disk, no GT, no RNG; a function of the covariate
  value and a fitted policy only;
* **leak-free by construction** — the covariate is the observable one from
  :mod:`biohub_tracking.eval.regime` (GT-free, test-computable); the policy
  threshold and per-regime operating points are **fit on training families only**
  (leave-one-family-out), never on the held-out family;
* **default-off** — the champion config carries no ``regime_conditional_link``
  block, so :func:`biohub_tracking.champion.regime_conditional_link_policy`
  returns ``None`` and the champion linking path is byte-for-byte unchanged. Only
  a promoted config would carry the block and invoke this layer.

Mirrors the detection-side :mod:`biohub_tracking.eval.regime_op` (SOT-2923); the
two are independent so a child can ablate the linking axis with detection frozen.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .regime import assign_regime


class RegimeLinkPoint(NamedTuple):
    """A linking operating point: motion prediction gain + mutual-NN prune.

    ``motion_gain`` scales SOT-2864's ARGUS motion-model predicted position
    (champion ``1.0``; the SOT-2900 reserve is ``2.0``). ``cycle_consistency_gate``
    turns on the SOT-2910 bidirectional mutual-NN edge prune, and
    ``cycle_consistency_margin`` is its runner-up drop margin (``0.0`` = the pure
    mutual-NN prune, SOT-2910's best point). A *more aggressive* operating point
    raises ``motion_gain`` and/or enables the prune.
    """

    motion_gain: float
    cycle_consistency_gate: bool
    cycle_consistency_margin: float = 0.0


@dataclasses.dataclass(frozen=True)
class ConditionalLinkPolicy:
    """A per-sequence *linking* operating-point policy keyed on one covariate.

    A sequence whose ``covariate_key`` covariate lands in the *dense* regime gets
    ``dense_op``; the *sparse* regime gets ``sparse_op``. ``dense_is_low`` follows
    the covariate orientation (``True`` for ``median_knn_um``: tighter spacing ⇒
    denser). A sequence whose covariate is missing (NaN ⇒ ``"unknown"`` regime)
    falls back to ``sparse_op`` — the champion / recall-preserving side — so an
    unclassifiable sequence never gets the aggressive dense operating point.
    """

    covariate_key: str
    threshold: float
    dense_is_low: bool
    dense_op: RegimeLinkPoint
    sparse_op: RegimeLinkPoint

    def regime_of(self, covariate_value: float) -> str:
        return assign_regime(
            covariate_value, self.threshold, dense_is_low=self.dense_is_low
        )

    def op_for(self, covariate_value: float) -> RegimeLinkPoint:
        """Linking operating point for a sequence given its covariate value."""
        regime = self.regime_of(covariate_value)
        if regime == "dense":
            return self.dense_op
        # "sparse" and "unknown" both take the champion / recall-preserving side.
        return self.sparse_op

    def to_dict(self) -> dict:
        def _op(op: RegimeLinkPoint) -> dict:
            return {
                "motion_gain": op.motion_gain,
                "cycle_consistency_gate": op.cycle_consistency_gate,
                "cycle_consistency_margin": op.cycle_consistency_margin,
            }

        return {
            "covariate_key": self.covariate_key,
            "threshold": self.threshold,
            "dense_is_low": self.dense_is_low,
            "dense_op": _op(self.dense_op),
            "sparse_op": _op(self.sparse_op),
        }

    @classmethod
    def from_dict(cls, block: Mapping) -> ConditionalLinkPolicy:
        """Build a policy from a champion-config ``regime_conditional_link`` block.

        The block mirrors :meth:`to_dict`. Missing operating-point knobs fall back
        to the champion linking values (``motion_gain=1.0``, gate off,
        ``margin=0.0``) so a partial block is still safe.
        """
        d = block.get("dense_op", {}) or {}
        s = block.get("sparse_op", {}) or {}

        def _op(o: Mapping) -> RegimeLinkPoint:
            return RegimeLinkPoint(
                motion_gain=float(o.get("motion_gain", 1.0)),
                cycle_consistency_gate=bool(o.get("cycle_consistency_gate", False)),
                cycle_consistency_margin=float(o.get("cycle_consistency_margin", 0.0)),
            )

        return cls(
            covariate_key=str(block.get("covariate_key", "median_knn_um")),
            threshold=float(block["threshold"]),
            dense_is_low=bool(block.get("dense_is_low", True)),
            dense_op=_op(d),
            sparse_op=_op(s),
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
    """The linking policy fitted on the training families of one LOFO fold."""

    policy: ConditionalLinkPolicy
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
    scored: Mapping[tuple[str, RegimeLinkPoint], object],
    *,
    op_grid: Sequence[RegimeLinkPoint],
    champion_op: RegimeLinkPoint,
    champion_adj_by_family: Mapping[str, float],
    covariate_key: str = "median_knn_um",
    dense_is_low: bool = True,
    eps: float = 1e-9,
) -> FoldFit:
    """Fit the conditional linking policy on the training families (leak-free).

    Searches ``threshold × dense_op × sparse_op`` and picks the policy that
    **maximises the training micro adjusted Jaccard subject to per-family
    non-regression** vs the champion operating point (the exact promotion gate,
    applied on the training split). If no split-based policy clears the training
    non-regression gate, falls back to the champion operating point on both
    regimes (``fell_back_to_champion=True``) — which is trivially non-regressing
    and reproduces the champion on this fold.

    ``scored[(family, op)]`` is a pre-computed ``FamilyResult`` (adjusted/raw
    Jaccard + weight) for that family under that linking operating point. Only
    training families are read here; the held-out family is never touched.
    """
    train = list(train_families)
    train_vals = [covariate_by_family[f] for f in train]
    champ_policy = ConditionalLinkPolicy(
        covariate_key=covariate_key,
        threshold=float("inf"),
        dense_is_low=dense_is_low,
        dense_op=champion_op,
        sparse_op=champion_op,
    )

    def train_rows_for(policy: ConditionalLinkPolicy):
        return [scored[(f, policy.op_for(covariate_by_family[f]))] for f in train]

    def no_regression(rows) -> bool:
        return all(
            r.adj_edge_jaccard >= champion_adj_by_family[r.name] - eps for r in rows
        )

    best: FoldFit | None = None
    for thr in threshold_candidates(train_vals):
        for dense_op in op_grid:
            for sparse_op in op_grid:
                policy = ConditionalLinkPolicy(
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


def _is_better(
    cand: FoldFit, incumbent: FoldFit, champion_op: RegimeLinkPoint
) -> bool:
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


def _n_nonchampion(
    policy: ConditionalLinkPolicy, champion_op: RegimeLinkPoint
) -> int:
    return int(policy.dense_op != champion_op) + int(policy.sparse_op != champion_op)
