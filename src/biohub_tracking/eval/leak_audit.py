"""Learned-candidate leak audit — hardening the leak-free CV for judging the
learned detection/linking candidates of directions 1-2 (SOT-3015).

Why this exists — the leak-free guarantee of the primary CV
(:mod:`biohub_tracking.eval.cv`) is **conditional on nothing being fit on the
holdout**. That module's docstring says it outright: *"The pipeline is
deterministic and causal … There is nothing learned across time to leak."* That
is true for the **classical** DoG+NMS champion, which learns no parameters from
the four holdout videos. It is **false** for a *learned* candidate (a trained
`TemporalUNet3D` detector, a learned edge re-ranker, released organiser weights):
the moment a model is *fit* on data, the scoring set is only leak-free if that
fit never saw the scored entity. The entity/temporal holdout of :mod:`cv`
protects the **scorer**; it does not by itself protect a **learner**.

So directions 1-2 (SOT-2993 self-trained UNet, SOT-3011 released weights, learned
linkers) need a *training-side* holdout discipline on top of the scoring-side
holdout. This module ports the external validation design and known-leak
checklist from the official baseline + public notebooks and turns it into a
**pure, data-free, unit-tested** audit the parent-resume two-signal gate can run
on any learned candidate before trusting its CV number.

External sources (fetched/read for SOT-3015; scorer fidelity itself was already
confirmed byte-exact in SOT-2995, so this is about SPLIT design, not the scorer):

* **Official baseline** ``github.com/royerlab/kaggle-cell-tracking-competition``
  and the released artifacts ``thibautgoldsborough/cellmot-baseline-artifacts``.
  The organiser trains the learned pipeline **per cross-validation split**
  (the released detector/linker weights are literally namespaced
  ``weights/unet_transformer/split_0/…``): a model is fit on the training split
  and validated on **held-out videos it never trained on**. That is the external
  validation protocol we port — a learned candidate must be trained on a split
  that **excludes the video it is scored on**.
* **Public CV/EDA notebooks** (pilkwang EDA/baseline, harshitsama scoring,
  xiaoleilian classical) — the recurring known-leak warning is **same-embryo /
  same-lineage** contamination: two videos of the *same developing embryo* share
  appearance, illumination and developmental lineage, so holding out one video
  while training on its sibling is a *soft* train↔val leak even though no single
  frame or cell straddles the boundary.

The central finding for our four-family holdout (two ``44b6`` embryo videos + two
``6bba`` embryo videos): SOT-2993's learned-detector A/B trained **leave-one-
*video*-out** (4 folds, each on the other three videos). For the fold that holds
out ``44b6_0113de3b`` the training set still contains ``44b6_0b24845f`` — the
**same-embryo-lineage sibling**. That is exactly the same-lineage leak the
external notebooks warn about. The truly leak-free training discipline on this
holdout is **leave-one-*lineage*-out** (2 folds: hold out *all* ``44b6``, or hold
out *all* ``6bba``), so no video of the scored embryo lineage is ever in the
learner's training set. Released-weights candidates (SOT-3011) sidestep training
entirely but carry the opposite problem — their ``split_0`` weights were trained
on organiser videos of unconfirmed membership that may **include** our holdout
families, so their CV is an *optimistic, train-contaminated upper bound* (the
0.62→0.81 gap between SOT-2993's leak-free retrain and SOT-3011's released
weights is precisely that contamination + convergence gap).

Everything here is a pure function of the holdout family metadata
(:data:`biohub_tracking.eval.cv.CV_HOLDOUT`), so it unit-tests without the
(gitignored) competition data. It changes **no scorer, no champion, no live
gate** — it is the enabling audit SOT-3015 was opened to produce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .cv import CV_HOLDOUT, HoldoutFamily

# --------------------------------------------------------------------------- #
# Ported known-leak checklist (external validation design → our holdout).      #
# --------------------------------------------------------------------------- #


class LeakCheck(NamedTuple):
    """One ported known-leak item and how our leak-free CV handles it.

    ``severity`` is ``"hard"`` when a single entity (frame/cell/track) would
    straddle the train↔score boundary, ``"soft"`` for statistical (same-embryo)
    contamination that no single entity crosses but that still inflates a learner.
    ``applies_to`` records whether the item bites the ``"scorer"`` (any
    candidate), the ``"learner"`` (only trained candidates), or ``"both"``.
    """

    id: str
    name: str
    external_source: str
    severity: str  # "hard" | "soft"
    applies_to: str  # "scorer" | "learner" | "both"
    our_status: str  # "covered" | "covered_for_scorer_only" | "action_required"
    cure: str


# The checklist is ordered scorer-first (already solved) → learner-specific (the
# SOT-3015 additions). "covered_for_scorer_only" is the crux: the item is closed
# for the classical champion but OPEN for a learned candidate unless the cure is
# applied on the training side.
LEARNED_CANDIDATE_LEAK_CHECKLIST: tuple[LeakCheck, ...] = (
    LeakCheck(
        id="entity-holdout",
        name="Frame/cell/track straddles the train↔score boundary",
        external_source="official evaluate.py per-video scoring; whole-.geff units",
        severity="hard",
        applies_to="both",
        our_status="covered",
        cure="Score whole embryo videos (cv.CV_HOLDOUT is a strict per-video "
        "partition); never split a cell/track/frame across a bucket.",
    ),
    LeakCheck(
        id="temporal-causality",
        name="A future frame informs an earlier prediction",
        external_source="official forward-only linking; causal detection",
        severity="hard",
        applies_to="both",
        our_status="covered",
        cure="Detection is per-timepoint; linking is forward-only (t→t+1); no "
        "parameter is fit across time in the classical pipeline.",
    ),
    LeakCheck(
        id="scorer-fidelity",
        name="Local scorer diverges from the organiser metric",
        external_source="official metrics.md (a=0.1, w=0.1, 7µm one-to-one, micro-avg)",
        severity="hard",
        applies_to="scorer",
        our_status="covered",
        cure="SOT-2995 confirmed eval/official.py reproduces the genuine "
        "tracksdata scorer byte-exact (divergence 0 on 8 golden + 8 real).",
    ),
    LeakCheck(
        id="train-on-scored-video",
        name="A learner is trained on the very video it is then scored on",
        external_source="official split_0 weights: fit on train split, validate on held-out videos",
        severity="hard",
        applies_to="learner",
        our_status="covered_for_scorer_only",
        cure="Train each learned candidate on a fold that EXCLUDES the scored "
        "video (leave-one-out); the classical champion needs no fit, so cv.py "
        "did not have to enforce this — a learned candidate does.",
    ),
    LeakCheck(
        id="same-lineage-sibling",
        name="Learner trains on the same-embryo sibling of the scored video",
        external_source="public EDA notebooks: same-embryo videos share appearance/illumination/lineage",
        severity="soft",
        applies_to="learner",
        our_status="action_required",
        cure="Train leave-one-LINEAGE-out (hold out ALL 44b6 or ALL 6bba), not "
        "leave-one-video-out, so no video of the scored embryo lineage is in the "
        "training set. SOT-2993's per-video LOFO leaves the sibling in → soft leak.",
    ),
    LeakCheck(
        id="released-weights-provenance",
        name="Released weights trained on unconfirmed data that may include our holdout",
        external_source="thibautgoldsborough/cellmot-baseline-artifacts split_0 (membership not independently confirmable)",
        severity="soft",
        applies_to="learner",
        our_status="action_required",
        cure="Treat released-weights CV as an OPTIMISTIC train-contaminated upper "
        "bound, never a leak-free estimate (SOT-3011 0.81 vs leak-free retrain "
        "SOT-2993 0.62). A promotable learned number needs leak-free retraining.",
    ),
)


# --------------------------------------------------------------------------- #
# Training-side fold discipline (leave-one-lineage-out vs leave-one-video-out). #
# --------------------------------------------------------------------------- #


class LearnedFold(NamedTuple):
    """One training-side fold for a learned candidate.

    ``held_out`` are the videos scored by this fold (the model must NOT have
    trained on them); ``train`` are the videos the model may be fit on.
    ``residual_lineage_leak`` is True when any ``train`` video shares an embryo
    lineage with any ``held_out`` video — the soft same-embryo leak.
    """

    key: str
    held_out: tuple[str, ...]
    train: tuple[str, ...]
    residual_lineage_leak: bool


def _lineage_of(families: Sequence[HoldoutFamily]) -> dict[str, str]:
    return {f.name: f.lineage for f in families}


def learned_candidate_folds(
    families: Sequence[HoldoutFamily] = CV_HOLDOUT,
    *,
    by: str = "lineage",
) -> tuple[LearnedFold, ...]:
    """Build the leave-one-*group*-out training folds for a learned candidate.

    ``by="lineage"`` (the leak-free recommendation) holds out a whole embryo
    lineage at a time, so no video of the scored embryo is in the training set.
    ``by="video"`` holds out a single video (SOT-2993's scheme) and keeps the
    same-embryo sibling in training, which :attr:`LearnedFold.residual_lineage_leak`
    flags. Each fold's ``held_out`` set is the leak-free scoring unit for that
    fold; concatenating every fold's held-out predictions reconstructs a
    leak-free score over all four videos.
    """
    lineage_of = _lineage_of(families)
    names = tuple(f.name for f in families)
    if by == "video":
        groups = {n: (n,) for n in names}
    elif by == "lineage":
        groups = {}
        for n in names:
            groups.setdefault(lineage_of[n], ())
            groups[lineage_of[n]] += (n,)
    else:  # pragma: no cover - guarded by the caller / tests
        raise ValueError(f"by must be 'lineage' or 'video', got {by!r}")

    folds: list[LearnedFold] = []
    for key, held in sorted(groups.items()):
        train = tuple(n for n in names if n not in held)
        held_lineages = {lineage_of[n] for n in held}
        residual = any(lineage_of[t] in held_lineages for t in train)
        folds.append(
            LearnedFold(
                key=key, held_out=held, train=train, residual_lineage_leak=residual
            )
        )
    return tuple(folds)


class TrainValLeak(NamedTuple):
    """Result of auditing one declared train/val split of a learned candidate."""

    hard_leak_shared_video: tuple[str, ...]  # videos in BOTH train and val (hard)
    soft_leak_shared_lineage: tuple[str, ...]  # val lineages also present in train
    entity_holdout_ok: bool  # every val video is a whole holdout entity
    leak_free: bool  # neither hard nor soft leak


def train_val_leak_audit(
    train_videos: Sequence[str],
    val_videos: Sequence[str],
    families: Sequence[HoldoutFamily] = CV_HOLDOUT,
) -> TrainValLeak:
    """Audit a learned candidate's declared train/val split for train↔val leak.

    Reports the **hard** leak (a video scored is also trained on) and the
    **soft** same-lineage leak (a held-out video's embryo lineage also appears in
    training), and whether the split is leak-free by both criteria. Pure — takes
    only the video names and the holdout metadata.
    """
    lineage_of = _lineage_of(families)
    known = set(lineage_of)
    train = [v for v in train_videos if v in known]
    val = [v for v in val_videos if v in known]

    hard = tuple(sorted(set(train) & set(val)))
    train_lineages = {lineage_of[v] for v in train}
    soft = tuple(
        sorted({lineage_of[v] for v in val if lineage_of[v] in train_lineages})
    )
    entity_ok = all(v in known for v in val_videos) and bool(val_videos)
    leak_free = not hard and not soft
    return TrainValLeak(
        hard_leak_shared_video=hard,
        soft_leak_shared_lineage=soft,
        entity_holdout_ok=entity_ok,
        leak_free=leak_free,
    )


# --------------------------------------------------------------------------- #
# Provenance verdict for a learned candidate's CV number.                      #
# --------------------------------------------------------------------------- #

# How much CV-trust to place in a learned candidate's number, by how it was fit.
PROVENANCE_LEAK_FREE = "leak_free_retrained"  # LOFO-by-lineage, holdout excluded
PROVENANCE_UPPER_BOUND = "optimistic_upper_bound"  # released/external weights
PROVENANCE_CONTAMINATED = "train_contaminated"  # trained on the scored video


def provenance_verdict(
    *,
    trained_locally: bool,
    holdout_excluded_from_training: bool,
    lineage_held_out: bool,
) -> dict:
    """Classify how trustworthy a learned candidate's CV number is.

    * A candidate not trained locally (released/external weights of unconfirmed
      split membership) is an **optimistic upper bound** — its CV may be inflated
      by having seen the holdout during the organiser's training.
    * A locally-trained candidate that trained on the scored video is
      **train-contaminated** (hard leak) — reject the number outright.
    * A locally-trained candidate that excluded the scored video is at best
      **leak-free** only when it also held out the whole lineage; if only the
      single video was held out it is a leave-one-video-out estimate carrying the
      soft same-lineage leak, still an upper bound (weaker than leave-one-lineage).
    """
    if not trained_locally:
        verdict = PROVENANCE_UPPER_BOUND
        rationale = (
            "External/released weights; training-data membership vs the holdout is "
            "not independently confirmable → treat CV as an optimistic upper bound."
        )
    elif not holdout_excluded_from_training:
        verdict = PROVENANCE_CONTAMINATED
        rationale = (
            "Trained on the scored video (hard train-on-test leak) → the CV number "
            "is not usable evidence; reject."
        )
    elif not lineage_held_out:
        verdict = PROVENANCE_UPPER_BOUND
        rationale = (
            "Leave-one-video-out: the same-embryo sibling stayed in training (soft "
            "same-lineage leak) → still an upper bound, weaker than leave-one-lineage."
        )
    else:
        verdict = PROVENANCE_LEAK_FREE
        rationale = (
            "Leave-one-lineage-out: no video of the scored embryo lineage was in "
            "training → leak-free CV, promotable evidence under the two-signal gate."
        )
    return {
        "verdict": verdict,
        "cv_trustworthy_for_promotion": verdict == PROVENANCE_LEAK_FREE,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------- #
# Full report (checklist + fold discipline + example provenance).             #
# --------------------------------------------------------------------------- #


def _checklist_json() -> list[dict]:
    return [c._asdict() for c in LEARNED_CANDIDATE_LEAK_CHECKLIST]


def _folds_json(folds: Sequence[LearnedFold]) -> list[dict]:
    return [
        {
            "key": f.key,
            "held_out": list(f.held_out),
            "train": list(f.train),
            "residual_lineage_leak": f.residual_lineage_leak,
        }
        for f in folds
    ]


def learned_leak_report(
    families: Sequence[HoldoutFamily] = CV_HOLDOUT,
    *,
    examples: Mapping[str, dict] | None = None,
) -> dict:
    """The full learned-candidate CV-trust audit (pure, data-free).

    Bundles (1) the ported known-leak checklist, (2) the two training-side fold
    disciplines side by side (leave-one-lineage-out = leak-free vs
    leave-one-video-out = residual same-lineage leak), and (3) provenance
    verdicts for the concrete learned candidates evaluated so far, so the
    parent-resume two-signal gate has one artifact to consult.
    """
    lineage_folds = learned_candidate_folds(families, by="lineage")
    video_folds = learned_candidate_folds(families, by="video")

    if examples is None:
        # The learned candidates measured this cycle, keyed by how they were fit.
        examples = {
            "SOT-3011 released royerlab weights (ILP 0.8081)": provenance_verdict(
                trained_locally=False,
                holdout_excluded_from_training=False,
                lineage_held_out=False,
            ),
            "SOT-2993 self-trained UNet, leave-one-VIDEO-out (0.6217)": provenance_verdict(
                trained_locally=True,
                holdout_excluded_from_training=True,
                lineage_held_out=False,
            ),
            "hypothetical leave-one-LINEAGE-out retrain (leak-free target)": provenance_verdict(
                trained_locally=True,
                holdout_excluded_from_training=True,
                lineage_held_out=True,
            ),
        }

    return {
        "purpose": (
            "Harden the leak-free CV for judging learned candidates (SOT-3010 "
            "directions 1-2): the cv.py leak-free guarantee protects the scorer, "
            "not a learner — a learned candidate needs training-side holdout too."
        ),
        "scorer_untouched": True,
        "known_leak_checklist": _checklist_json(),
        "recommended_learned_fold": "leave-one-lineage-out",
        "leave_one_lineage_out": _folds_json(lineage_folds),
        "leave_one_video_out": _folds_json(video_folds),
        "leave_one_video_out_has_residual_leak": any(
            f.residual_lineage_leak for f in video_folds
        ),
        "leave_one_lineage_out_has_residual_leak": any(
            f.residual_lineage_leak for f in lineage_folds
        ),
        "candidate_provenance_examples": examples,
    }


def _main(argv: list[str] | None = None) -> int:
    """CLI: print (optionally write) the learned-candidate CV-trust audit."""
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Learned-candidate leak audit / CV-trust report (SOT-3015)."
    )
    ap.add_argument("--out", type=str, default=None, help="Write the JSON report here.")
    args = ap.parse_args(argv)

    report = learned_leak_report()
    print(json.dumps(report, indent=2))
    if args.out:
        from pathlib import Path

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
