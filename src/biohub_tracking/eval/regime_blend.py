"""Soft per-sequence operating-point *mixture* (SOT-2931, default-off).

Why this exists — the *family-mix / density-mix wall*, attacked one rung higher on
the escalation ladder (step-4: reformulate the problem, not micro-diff the
champion). SOT-2922 (linking) and SOT-2923 (detection) both tried to beat the wall
with a **hard regime label**: threshold the observable density covariate
(``median_knn_um``, SOT-2921) into ``dense`` / ``sparse`` and give each half its own
*discrete* operating point. Both were REJECTED on the mandatory 4/4 per-dataset
non-regression gate — a hard partition **crosscuts** the family boundary (the one
family that wants the aggressive prune, ``6bba_05b6850b``, is observably the
*sparsest*, so any single threshold puts it on the wrong side).

This module changes the **formulation**, not the champion: instead of a hard
switch that assigns each sequence to exactly one of two operating points, it forms
a **continuous weighted average** of a *conservative* and an *aggressive* operating
point, weighted by a smooth function of the same leak-free observable. A sequence
does not jump between two configs at a threshold; it sits on a **continuum** — its
motion-prediction gain slides continuously from champion (``1.0``) toward the
aggressive reserve (``2.0``) as the observable moves, and the mutual-NN prune
switches in only once the aggressive weight crosses a fitted activation.

Structural distinction from SOT-2922/2923 (recorded here per the acceptance
criterion). Those select ``op ∈ {dense_op, sparse_op}`` by ``assign_regime`` — a
**step function** of the covariate (the ``dense_is_low`` hard label). This selects
a **blended op** whose continuous knob is ``w = σ((center − x)/scale)`` — a
**logistic** function of the covariate. The hard switch is the *degenerate limit*
of this policy as ``scale → 0`` (the sigmoid becomes a step at ``center``), so the
soft mixture strictly generalises SOT-2922; any ``scale > 0`` is a genuinely new,
non-hard-partition operating regime that SOT-2922/2923 could not represent. The
covariate and its leak-free provenance are shared; the *decision rule over it* is
what differs (continuous mixture vs. hard label).

Leak-free / pure by construction, exactly like :mod:`regime_link`:

* **pure / deterministic** — no disk, no GT, no RNG; a function of the covariate
  value and a fitted policy only;
* **leak-free** — the covariate is the observable GT-free ``median_knn_um`` from
  :mod:`biohub_tracking.eval.regime`; the mixture parameters (``center``, ``scale``,
  ``gate_activation``, endpoints) are **fit on training families only**
  (leave-one-family-out) by the A/B harness, never on the held-out family;
* **default-off** — the champion config carries no ``operating_point_mixture``
  block, so :func:`biohub_tracking.champion.operating_point_mixture_policy` returns
  ``None`` and the champion linking path is byte-for-byte unchanged. Only a
  promoted config would carry the block and invoke this layer.

The module holds the *policy representation* and its *leak-free fit* as pure
functions of already-scored per-``(family, operating-point)`` results (the
conservative→aggressive **gain** axis is evaluated on a fine discrete grid the
harness pre-scores, so the continuous blend snaps to the nearest grid gain — a
disclosed numerical detail, not a hard partition). The A/B harness
(``experiments/sot2931/ab_soft_oppoint_mixture.py``) supplies the scored grid and
the LOFO protocol. Mirrors :mod:`regime_link` (SOT-2922) so the two are directly
comparable — same covariate, same gate, hard-switch vs. soft-mixture.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple


class BlendEndpoint(NamedTuple):
    """One *corner* operating point of the soft mixture.

    ``motion_gain`` scales SOT-2864's ARGUS motion-model predicted position
    (champion ``1.0``; the SOT-2900 reserve is ``2.0``). ``cycle_consistency_gate``
    turns on the SOT-2910 bidirectional mutual-NN edge prune and
    ``cycle_consistency_margin`` is its runner-up drop margin (``0.0`` = the pure
    mutual-NN prune). The *conservative* endpoint is the champion
    ``(1.0, gate off, 0.0)``; the *aggressive* endpoint raises the gain and/or
    enables the prune.
    """

    motion_gain: float
    cycle_consistency_gate: bool
    cycle_consistency_margin: float = 0.0


CHAMPION_ENDPOINT = BlendEndpoint(
    motion_gain=1.0, cycle_consistency_gate=False, cycle_consistency_margin=0.0
)


def _sigmoid(z: float) -> float:
    # Numerically-stable logistic; saturates rather than overflowing for large |z|.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclasses.dataclass(frozen=True)
class SoftBlendPolicy:
    """A per-sequence *soft mixture* of two linking operating points.

    Given a sequence's covariate value ``x`` the policy computes a continuous
    **aggressive weight** ``w = σ(sign·(center − x)/scale) ∈ (0, 1)`` (``sign`` from
    ``dense_is_low``: for ``median_knn_um`` a *smaller* value is denser, so
    ``dense_is_low=True`` makes small ``x`` map to large ``w``). The blended
    operating point is the ``w``-weighted average of the two endpoints:

    * ``motion_gain = (1 − w)·conservative.motion_gain + w·aggressive.motion_gain``
      — a genuinely continuous gain between the champion and the reserve;
    * ``cycle_consistency_margin`` is blended the same way;
    * the boolean ``cycle_consistency_gate`` is the aggressive endpoint's gate once
      ``w ≥ gate_activation`` (else the conservative endpoint's) — the one hard knob
      of the prune, thresholded on the *continuous* weight rather than on a hard
      regime label. ``gate_activation = +inf`` never enables the prune (pure gain
      blend); ``0.0`` enables it whenever ``w`` favours the aggressive side.

    ``scale → 0`` degenerates to a step at ``center`` (the SOT-2922 hard switch);
    ``scale > 0`` is the soft mixture. A missing / NaN covariate falls back to the
    conservative endpoint (champion / recall-preserving), so an unclassifiable
    sequence never gets the aggressive operating point.
    """

    covariate_key: str
    center: float
    scale: float
    dense_is_low: bool
    conservative: BlendEndpoint
    aggressive: BlendEndpoint
    gate_activation: float = 0.5

    def weight_of(self, covariate_value: float) -> float:
        """Continuous aggressive weight ``w ∈ (0, 1)`` for a sequence.

        NaN covariate ⇒ ``0.0`` (fully conservative / champion side). A
        non-positive ``scale`` is treated as the hard-switch limit (step at
        ``center``): ``w`` is ``1.0`` on the aggressive side of ``center`` and
        ``0.0`` on the conservative side.
        """
        x = covariate_value
        if math.isnan(x):
            return 0.0
        signed = (self.center - x) if self.dense_is_low else (x - self.center)
        if self.scale <= 0.0:
            # Hard-switch limit (the SOT-2922 step); ties resolve conservative.
            return 1.0 if signed > 0.0 else 0.0
        return _sigmoid(signed / self.scale)

    def op_for(self, covariate_value: float) -> BlendEndpoint:
        """Blended linking operating point for a sequence given its covariate."""
        w = self.weight_of(covariate_value)
        c, a = self.conservative, self.aggressive
        gain = (1.0 - w) * c.motion_gain + w * a.motion_gain
        margin = (1.0 - w) * c.cycle_consistency_margin + w * a.cycle_consistency_margin
        gate = a.cycle_consistency_gate if w >= self.gate_activation else c.cycle_consistency_gate
        if not gate:
            # Gate off ⇒ margin is inert; keep the conservative margin for clarity.
            margin = c.cycle_consistency_margin
        return BlendEndpoint(
            motion_gain=gain, cycle_consistency_gate=gate, cycle_consistency_margin=margin
        )

    def to_dict(self) -> dict:
        def _ep(ep: BlendEndpoint) -> dict:
            return {
                "motion_gain": ep.motion_gain,
                "cycle_consistency_gate": ep.cycle_consistency_gate,
                "cycle_consistency_margin": ep.cycle_consistency_margin,
            }

        return {
            "covariate_key": self.covariate_key,
            "center": self.center,
            "scale": self.scale,
            "dense_is_low": self.dense_is_low,
            "gate_activation": self.gate_activation,
            "conservative": _ep(self.conservative),
            "aggressive": _ep(self.aggressive),
        }

    @classmethod
    def from_dict(cls, block: Mapping) -> SoftBlendPolicy:
        """Build a policy from a champion-config ``operating_point_mixture`` block.

        Missing endpoint knobs fall back to the champion linking values
        (``motion_gain=1.0``, gate off, ``margin=0.0``) so a partial block is still
        safe; a missing ``conservative`` block defaults to the champion endpoint.
        """
        def _ep(o: Mapping | None, default: BlendEndpoint) -> BlendEndpoint:
            o = o or {}
            return BlendEndpoint(
                motion_gain=float(o.get("motion_gain", default.motion_gain)),
                cycle_consistency_gate=bool(
                    o.get("cycle_consistency_gate", default.cycle_consistency_gate)
                ),
                cycle_consistency_margin=float(
                    o.get("cycle_consistency_margin", default.cycle_consistency_margin)
                ),
            )

        aggr_default = BlendEndpoint(
            motion_gain=2.0, cycle_consistency_gate=True, cycle_consistency_margin=0.0
        )
        return cls(
            covariate_key=str(block.get("covariate_key", "median_knn_um")),
            center=float(block["center"]),
            scale=float(block.get("scale", 0.0)),
            dense_is_low=bool(block.get("dense_is_low", True)),
            gate_activation=float(block.get("gate_activation", 0.5)),
            conservative=_ep(block.get("conservative"), CHAMPION_ENDPOINT),
            aggressive=_ep(block.get("aggressive"), aggr_default),
        )


def center_candidates(train_values: Sequence[float]) -> list[float]:
    """Leak-free sigmoid centers from *training* covariate values only.

    Returns the midpoints between adjacent sorted training values (each is a
    plausible crossover of the aggressive weight through 0.5 on the training
    families) plus two degenerate edges (``-inf`` / ``+inf``) that collapse the
    mixture to a single global endpoint (champion-safe fallback). NaN dropped.
    """
    vals = sorted(v for v in train_values if not math.isnan(v))
    mids = [(a + b) / 2.0 for a, b in itertools.pairwise(vals) if b > a]
    return [float("-inf"), *mids, float("inf")]


def snap_gain(gain: float, gain_grid: Sequence[float]) -> float:
    """Snap a continuous blended gain to the nearest pre-scored grid gain.

    The conservative→aggressive **gain** axis is evaluated on the fine discrete
    grid the A/B harness pre-scores; the continuous blend is realised by mapping
    each sequence's ``motion_gain`` to its nearest grid value (a numerical
    discretisation of the continuum, not a two-way hard partition). Deterministic
    tie-break: the lower gain (closer to champion) wins.
    """
    return min(gain_grid, key=lambda g: (abs(g - gain), g))


def op_key(op: BlendEndpoint, gain_grid: Sequence[float]) -> tuple[float, bool]:
    """Pre-scored-grid key ``(snapped_gain, gate)`` for a blended op.

    ``cycle_consistency_margin`` is held at ``0.0`` on both endpoints in the A/B
    grid (the prune is the pure mutual-NN rule), so the scored grid is
    ``gain_grid × {gate off, on}``; the key drops the (constant) margin.
    """
    return (snap_gain(op.motion_gain, gain_grid), bool(op.cycle_consistency_gate))


class FoldFit(NamedTuple):
    """The soft-mixture policy fitted on the training families of one LOFO fold."""

    policy: SoftBlendPolicy
    train_micro_adj: float
    train_no_regression: bool
    fell_back_to_champion: bool


def _aggregate_micro_adj(rows) -> float:
    """Weighted micro-average of adjusted edge Jaccard (weight = tp+fp+fn).

    Mirrors :func:`biohub_tracking.eval.cv.aggregate` micro arithmetic without
    importing the heavy scorer, so this stays pure/testable.
    """
    total_w = sum(r.weight for r in rows)
    if total_w <= 0:
        return float("nan")
    return sum(r.adj_edge_jaccard * r.weight for r in rows) / total_w


def fit_fold_policy(
    train_families: Sequence[str],
    covariate_by_family: Mapping[str, float],
    scored: Mapping[tuple[float, bool], object] | Mapping[tuple[str, tuple[float, bool]], object],
    *,
    gain_grid: Sequence[float],
    scale_grid: Sequence[float],
    gate_activation_grid: Sequence[float],
    aggressive_gain_grid: Sequence[float],
    champion_adj_by_family: Mapping[str, float],
    covariate_key: str = "median_knn_um",
    dense_is_low: bool = True,
    eps: float = 1e-9,
) -> FoldFit:
    """Fit the soft-mixture policy on the training families (leak-free).

    Searches ``center × scale × gate_activation × aggressive_gain`` and picks the
    policy that **maximises the training micro adjusted Jaccard subject to
    per-family non-regression** vs the champion (the exact promotion gate on the
    training split). The conservative endpoint is fixed at the champion; the
    aggressive endpoint pairs a swept gain with the mutual-NN gate on. If no soft
    policy clears the training non-regression gate, falls back to the champion
    everywhere (``fell_back_to_champion=True``), which reproduces the champion.

    ``scored[(family, (gain, gate))]`` is a pre-computed ``FamilyResult`` for that
    family under the snapped ``(gain, gate)`` grid operating point. Only training
    families are read; the held-out family is never touched.
    """
    train = list(train_families)
    train_vals = [covariate_by_family[f] for f in train]
    grid = list(gain_grid)

    def op_for_family(policy: SoftBlendPolicy, fam: str) -> tuple[float, bool]:
        return op_key(policy.op_for(covariate_by_family[fam]), grid)

    def train_rows_for(policy: SoftBlendPolicy):
        return [scored[(f, op_for_family(policy, f))] for f in train]

    def no_regression(rows) -> bool:
        return all(
            r.adj_edge_jaccard >= champion_adj_by_family[r.name] - eps for r in rows
        )

    champ_policy = SoftBlendPolicy(
        covariate_key=covariate_key,
        center=float("inf"),
        scale=0.0,
        dense_is_low=dense_is_low,
        conservative=CHAMPION_ENDPOINT,
        aggressive=CHAMPION_ENDPOINT,
        gate_activation=float("inf"),
    )

    best: FoldFit | None = None
    for center in center_candidates(train_vals):
        for scale in scale_grid:
            for gate_act in gate_activation_grid:
                for aggr_gain in aggressive_gain_grid:
                    policy = SoftBlendPolicy(
                        covariate_key=covariate_key,
                        center=center,
                        scale=scale,
                        dense_is_low=dense_is_low,
                        conservative=CHAMPION_ENDPOINT,
                        aggressive=BlendEndpoint(
                            motion_gain=aggr_gain,
                            cycle_consistency_gate=True,
                            cycle_consistency_margin=0.0,
                        ),
                        gate_activation=gate_act,
                    )
                    rows = train_rows_for(policy)
                    if not no_regression(rows):
                        continue
                    micro = _aggregate_micro_adj(rows)
                    fell_back = all(
                        op_for_family(policy, f) == op_key(CHAMPION_ENDPOINT, grid)
                        for f in train
                    )
                    cand = FoldFit(
                        policy=policy,
                        train_micro_adj=micro,
                        train_no_regression=True,
                        fell_back_to_champion=fell_back,
                    )
                    if best is None or _is_better(cand, best, grid):
                        best = cand

    if best is None:
        champ_rows = [scored[(f, op_key(CHAMPION_ENDPOINT, grid))] for f in train]
        return FoldFit(
            policy=champ_policy,
            train_micro_adj=_aggregate_micro_adj(champ_rows),
            train_no_regression=no_regression(champ_rows),
            fell_back_to_champion=True,
        )
    return best


def _is_better(cand: FoldFit, incumbent: FoldFit, gain_grid: Sequence[float]) -> bool:
    """Higher training micro wins; ties prefer the least-aggressive (softest) policy.

    The tie-break keeps the fit deterministic and biased toward the champion: a
    larger ``scale`` (softer mixture) and a smaller aggressive gain are preferred
    at equal training micro, so gratuitous divergence from the frozen champion is
    avoided.
    """
    if cand.train_micro_adj > incumbent.train_micro_adj + 1e-12:
        return True
    if cand.train_micro_adj < incumbent.train_micro_adj - 1e-12:
        return False
    return _softness_key(cand.policy) < _softness_key(incumbent.policy)


def _softness_key(policy: SoftBlendPolicy) -> tuple[float, float]:
    # Prefer a larger scale (softer) and a smaller aggressive gain (closer to
    # champion). Negate scale so ``min`` picks the softest.
    return (-policy.scale, policy.aggressive.motion_gain)
