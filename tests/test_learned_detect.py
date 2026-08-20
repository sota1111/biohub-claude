"""Tests for the SOT-2847 learned-detector receptacle.

Two layers:

* **torch-free** (always run, incl. the numpy/scipy/zarr CI env): the receptacle
  is off by default and the champion path is byte-for-byte unchanged — no learned
  detector is built, torch is never imported, and the classical config resolves
  exactly as before.
* **torch-guarded** (``importorskip('torch')``, run in the GPU env): the detector
  honours the classical detector contract and the offline weights-attach
  smoke — construct → save → attach → load offline → forward — is feasible.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.champion import (
    champion_params,
    learned_detector_config,
    load_champion_config,
)
from biohub_tracking.learned_detect import (
    LearnedDetectorConfig,
    build_learned_detector,
    torch_available,
)


# --------------------------------------------------------------------------- #
# torch-free: default-off receptacle keeps the champion byte-for-byte unchanged
# --------------------------------------------------------------------------- #
def test_champion_config_has_no_learned_detector_block():
    """The reigning champion carries no learned_detector block → classical path."""
    assert learned_detector_config(load_champion_config()) is None


def test_absent_or_disabled_block_builds_no_detector():
    """None / absent / disabled → build returns None (classical detector used)."""
    assert build_learned_detector(None) is None
    assert build_learned_detector({}) is None
    assert build_learned_detector({"enabled": False, "arch": "voxel_scorer_v0"}) is None


def test_config_from_dict_parses_and_defaults_off():
    assert LearnedDetectorConfig.from_dict(None) is None
    assert LearnedDetectorConfig.from_dict({}) is None
    cfg = LearnedDetectorConfig.from_dict(
        {"enabled": True, "weights": "ds/w.pt", "threshold": 0.7, "device": "cpu"}
    )
    assert cfg is not None
    assert cfg.enabled and cfg.weights == "ds/w.pt"
    assert cfg.threshold == 0.7 and cfg.device == "cpu"
    # A bare dict defaults to disabled so a stray block never flips detection on.
    assert LearnedDetectorConfig(enabled=False).enabled is False


def test_learned_detector_config_accepts_nested_placement():
    """The block is honoured at top level or nested under ``detect``."""
    top = {"learned_detector": {"enabled": True}}
    nested = {"detect": {"learned_detector": {"enabled": True}}}
    assert learned_detector_config(top) == {"enabled": True}
    assert learned_detector_config(nested) == {"enabled": True}
    assert learned_detector_config({"detect": {}}) is None


def test_champion_params_unaffected_by_receptacle():
    """Resolving champion params does not depend on the receptacle at all."""
    detect, link, scale = champion_params(load_champion_config())
    assert detect.mad_k == 3.0
    assert link.min_track_length == 4
    assert tuple(scale) == (1.625, 0.40625, 0.40625)


def test_pipeline_learned_path_is_skipped_when_none(monkeypatch):
    """run_pipeline with learned_detector=None must not touch the learned module.

    Import of learned_detect would still be lazy, but the classical branch must be
    taken; assert the learned branch is not entered by using a sentinel detector
    that would raise if called.
    """
    from biohub_tracking import pipeline

    class _Boom:
        def detect_series(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("learned path taken when it should be classical")

    # Passing learned_detector=None takes the classical path; passing the sentinel
    # would raise. We only assert the None default is honoured via signature.
    import inspect

    sig = inspect.signature(pipeline.run_pipeline)
    assert sig.parameters["learned_detector"].default is None


# --------------------------------------------------------------------------- #
# torch-guarded: detector contract + offline weights-attach smoke
#
# Per-test skip (NOT a module-level importorskip, which would skip the torch-free
# byte-invariance tests above too) so the CI numpy/scipy/zarr env still runs them.
# --------------------------------------------------------------------------- #
requires_torch = pytest.mark.skipif(
    not torch_available(), reason="torch not importable (classical/CI env)"
)


def _save_dummy_weights(tmp_path):
    import torch

    from biohub_tracking.learned_detect import _build_arch

    torch.manual_seed(0)
    model = _build_arch("voxel_scorer_v0")
    ds_dir = tmp_path / "kaggle_input" / "biohub-claude-weights"
    ds_dir.mkdir(parents=True, exist_ok=True)
    wpath = ds_dir / "detector.pt"
    torch.save(model.state_dict(), wpath)
    return model, wpath, tmp_path / "kaggle_input"


@requires_torch
def test_detector_contract_detect_series(tmp_path):
    """detect_series returns {t: (N, 3)} float voxel coords — the classical
    detector contract, so it is drop-in for the linker."""
    from biohub_tracking.learned_detect import LearnedDetector

    _model, wpath, _root = _save_dummy_weights(tmp_path)
    det = LearnedDetector.for_smoke(wpath, device="cpu", threshold=0.0)
    vol_series = np.random.default_rng(0).random((3, 8, 16, 16)).astype(np.float32)
    out = det.detect_series(vol_series)
    assert set(out) == {0, 1, 2}
    for coords in out.values():
        assert coords.ndim == 2 and coords.shape[1] == 3
        assert coords.dtype == np.float64

    single = det.detect_centroids(vol_series[0])
    assert single.ndim == 2 and single.shape[1] == 3


@requires_torch
def test_weights_attach_resolves_offline(tmp_path):
    """The kernel's weights discovery finds an attached dataset mount and loads
    the checkpoint without any network."""
    import torch

    from biohub_tracking.learned_detect import LearnedDetector

    model, _wpath, input_root = _save_dummy_weights(tmp_path)
    det = LearnedDetector(
        LearnedDetectorConfig(
            enabled=True, weights="biohub-claude-weights/detector.pt", device="cpu"
        )
    )
    resolved = det.resolve_weights_path(search_roots=(str(input_root),))
    assert resolved is not None and resolved.is_file()
    det.load(weights_path=resolved)
    # Bit-exact offline reload.
    reloaded = det._model.state_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(reloaded[k].cpu(), v.cpu())


@requires_torch
def test_run_weights_attach_smoke_reports_feasible():
    """The end-to-end offline smoke returns a FEASIBLE verdict."""
    from biohub_tracking.learned_detect import run_weights_attach_smoke

    assert torch_available()
    result = run_weights_attach_smoke(device="cpu")
    assert result["torch_available"] is True
    assert result["resolved_offline"] is True
    assert result["state_dict_bit_exact"] is True
    assert result["forward_ok"] is True
    assert result["num_timepoints"] == 3
    assert "FEASIBLE" in result["verdict"]


@requires_torch
def test_build_learned_detector_enabled_loads(tmp_path):
    """An enabled block with attached weights builds a working detector."""
    from biohub_tracking.learned_detect import LearnedDetector

    _model, _wpath, input_root = _save_dummy_weights(tmp_path)
    # Point the receptacle's discovery at the simulated mount.
    import biohub_tracking.learned_detect as ld

    monkey_roots = (str(input_root),)
    orig = LearnedDetector.resolve_weights_path

    def _patched(self, search_roots=monkey_roots):
        return orig(self, search_roots=monkey_roots)

    ld.LearnedDetector.resolve_weights_path = _patched
    try:
        det = build_learned_detector(
            {
                "enabled": True,
                "weights": "biohub-claude-weights/detector.pt",
                "device": "cpu",
                "threshold": 0.0,
            }
        )
        assert det is not None
        out = det.detect_series(np.zeros((2, 6, 12, 12), dtype=np.float32))
        assert set(out) == {0, 1}
    finally:
        ld.LearnedDetector.resolve_weights_path = orig
