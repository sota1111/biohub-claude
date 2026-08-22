"""Learned-candidate leak audit (SOT-3015) — pure, data-free coverage.

Pins the training-side holdout discipline that hardens the leak-free CV for
judging learned candidates (directions 1-2): the leave-one-lineage-out fold is
leak-free while the leave-one-video-out fold carries the soft same-lineage leak,
the train/val audit catches hard and soft leaks, and the provenance verdict
gates CV-trust. None of this needs the (gitignored) competition data.
"""

from __future__ import annotations

from biohub_tracking.eval.cv import CV_HOLDOUT
from biohub_tracking.eval.leak_audit import (
    LEARNED_CANDIDATE_LEAK_CHECKLIST,
    PROVENANCE_CONTAMINATED,
    PROVENANCE_LEAK_FREE,
    PROVENANCE_UPPER_BOUND,
    learned_candidate_folds,
    learned_leak_report,
    provenance_verdict,
    train_val_leak_audit,
)

# --------------------------------------------------------------------------- #
# Fold discipline: leave-one-lineage-out is leak-free, leave-one-video is not.  #
# --------------------------------------------------------------------------- #


def test_leave_one_lineage_out_is_leak_free() -> None:
    folds = learned_candidate_folds(by="lineage")
    # Two embryo lineages → two folds, each holding out both videos of one embryo.
    assert len(folds) == 2
    assert {f.key for f in folds} == {"44b6", "6bba"}
    for f in folds:
        assert len(f.held_out) == 2
        assert not f.residual_lineage_leak
        # No held-out video's lineage may appear in the training set.
        held_prefix = f.key
        assert all(not t.startswith(held_prefix) for t in f.train)


def test_leave_one_video_out_has_residual_same_lineage_leak() -> None:
    folds = learned_candidate_folds(by="video")
    # Four videos → four folds; every fold keeps the same-embryo sibling in train.
    assert len(folds) == 4
    assert all(f.residual_lineage_leak for f in folds)
    # Concretely: holding out 44b6_0113de3b still trains on 44b6_0b24845f.
    f0 = next(f for f in folds if f.held_out == ("44b6_0113de3b",))
    assert "44b6_0b24845f" in f0.train


def test_folds_cover_all_videos_exactly_once() -> None:
    for by in ("lineage", "video"):
        held = [v for f in learned_candidate_folds(by=by) for v in f.held_out]
        assert sorted(held) == sorted(f.name for f in CV_HOLDOUT)


# --------------------------------------------------------------------------- #
# Train/val split audit: hard (shared video) vs soft (shared lineage) leak.     #
# --------------------------------------------------------------------------- #


def test_hard_leak_when_scored_video_in_training() -> None:
    audit = train_val_leak_audit(
        train_videos=["44b6_0113de3b", "6bba_05b6850b"],
        val_videos=["44b6_0113de3b"],
    )
    assert audit.hard_leak_shared_video == ("44b6_0113de3b",)
    assert not audit.leak_free


def test_soft_leak_when_only_sibling_in_training() -> None:
    # SOT-2993's per-video LOFO fold: hold out 44b6_0113de3b, train on the rest.
    audit = train_val_leak_audit(
        train_videos=["44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"],
        val_videos=["44b6_0113de3b"],
    )
    assert audit.hard_leak_shared_video == ()  # no hard leak
    assert audit.soft_leak_shared_lineage == ("44b6",)  # but soft same-lineage leak
    assert audit.entity_holdout_ok
    assert not audit.leak_free


def test_leave_one_lineage_out_split_is_leak_free() -> None:
    audit = train_val_leak_audit(
        train_videos=["6bba_05b6850b", "6bba_05db0fb1"],
        val_videos=["44b6_0113de3b", "44b6_0b24845f"],
    )
    assert audit.hard_leak_shared_video == ()
    assert audit.soft_leak_shared_lineage == ()
    assert audit.leak_free


# --------------------------------------------------------------------------- #
# Provenance verdict gates CV-trust for promotion.                             #
# --------------------------------------------------------------------------- #


def test_released_weights_are_optimistic_upper_bound() -> None:
    v = provenance_verdict(
        trained_locally=False,
        holdout_excluded_from_training=False,
        lineage_held_out=False,
    )
    assert v["verdict"] == PROVENANCE_UPPER_BOUND
    assert v["cv_trustworthy_for_promotion"] is False


def test_train_on_scored_video_is_contaminated() -> None:
    v = provenance_verdict(
        trained_locally=True,
        holdout_excluded_from_training=False,
        lineage_held_out=False,
    )
    assert v["verdict"] == PROVENANCE_CONTAMINATED
    assert v["cv_trustworthy_for_promotion"] is False


def test_leave_one_video_out_still_upper_bound() -> None:
    v = provenance_verdict(
        trained_locally=True,
        holdout_excluded_from_training=True,
        lineage_held_out=False,
    )
    assert v["verdict"] == PROVENANCE_UPPER_BOUND
    assert v["cv_trustworthy_for_promotion"] is False


def test_leave_one_lineage_out_is_promotable() -> None:
    v = provenance_verdict(
        trained_locally=True,
        holdout_excluded_from_training=True,
        lineage_held_out=True,
    )
    assert v["verdict"] == PROVENANCE_LEAK_FREE
    assert v["cv_trustworthy_for_promotion"] is True


# --------------------------------------------------------------------------- #
# Checklist + full report shape.                                              #
# --------------------------------------------------------------------------- #


def test_checklist_flags_learner_items_as_not_yet_covered() -> None:
    by_id = {c.id: c for c in LEARNED_CANDIDATE_LEAK_CHECKLIST}
    # The two learner-specific items SOT-3015 adds must not read as fully covered.
    assert by_id["train-on-scored-video"].our_status == "covered_for_scorer_only"
    assert by_id["same-lineage-sibling"].our_status == "action_required"
    assert by_id["released-weights-provenance"].our_status == "action_required"
    # The scorer-side items remain covered (SOT-2995 etc.).
    assert by_id["scorer-fidelity"].our_status == "covered"
    assert by_id["entity-holdout"].our_status == "covered"


def test_report_is_serialisable_and_flags_the_leak() -> None:
    import json

    report = learned_leak_report()
    json.dumps(report)  # must be JSON-serialisable
    assert report["scorer_untouched"] is True
    assert report["recommended_learned_fold"] == "leave-one-lineage-out"
    assert report["leave_one_video_out_has_residual_leak"] is True
    assert report["leave_one_lineage_out_has_residual_leak"] is False
    # The released-weights example is not promotable; the leak-free retrain is.
    examples = report["candidate_provenance_examples"]
    assert any(
        v["cv_trustworthy_for_promotion"] for v in examples.values()
    )
    assert any(
        not v["cv_trustworthy_for_promotion"] for v in examples.values()
    )
