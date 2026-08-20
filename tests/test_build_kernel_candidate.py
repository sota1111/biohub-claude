"""Tests for the SOT-2840 candidate-artifact build path in ``submit/build_kernel.py``.

The candidate build must (a) leave the champion build byte-identical and deterministic,
(b) emit a valid, self-contained kernel that embeds the candidate config and installs it
as ``BIOHUB_CHAMPION_CONFIG`` at runtime, and (c) reproduce the candidate linking params
via the same resolver the kernel uses — all without touching the live champion config.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from biohub_tracking.champion import champion_params, load_champion_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = REPO_ROOT / "champion" / "candidates" / "sot2840-globalmcf-theta6p5-window2.json"
CHAMPION_CONFIG = REPO_ROOT / "champion" / "config.json"


def _load_build_kernel():
    path = REPO_ROOT / "submit" / "build_kernel.py"
    spec = importlib.util.spec_from_file_location("build_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def bk():
    return _load_build_kernel()


def _read(path: Path) -> str:
    return path.read_text()


def test_champion_build_is_deterministic_and_has_no_candidate_marker(bk, tmp_path):
    """The champion path is unchanged by the candidate feature: two builds match and
    the output never carries candidate scaffolding."""
    sub = "_pytest_champ_tmp"
    try:
        f1 = bk.build(REPO_ROOT, out_subdir=sub, code_file="k.py")
        text1 = _read(f1)
        f2 = bk.build(REPO_ROOT, out_subdir=sub, code_file="k.py")
        assert _read(f2) == text1  # deterministic
        assert "EMBEDDED_CANDIDATE_CONFIG" not in text1
        assert "_install_candidate_config" not in text1
    finally:
        shutil.rmtree(REPO_ROOT / "submit" / sub, ignore_errors=True)


def test_candidate_build_embeds_config_and_installs_it(bk, tmp_path):
    sub = "_pytest_cand_tmp"
    candidate_cfg = json.loads(CANDIDATE_CONFIG.read_text())
    try:
        kernel = bk.build(
            REPO_ROOT, candidate_config=candidate_cfg, out_subdir=sub, code_file="k.py"
        )
        text = _read(kernel)
        # Embeds the candidate config and wires it in before the pipeline runs.
        assert "EMBEDDED_CANDIDATE_CONFIG" in text
        assert "_install_candidate_config()" in text
        assert 'os.environ["BIOHUB_CHAMPION_CONFIG"]' in text
        # Still a valid, self-contained module.
        compile(text, str(kernel), "exec")
        # Metadata points at a distinct (non-champion) kernel id.
        meta = json.loads((REPO_ROOT / "submit" / sub / "kernel-metadata.json").read_text())
        assert meta["id"] != bk.KERNEL_ID
        assert meta["code_file"] == "k.py"
    finally:
        shutil.rmtree(REPO_ROOT / "submit" / sub, ignore_errors=True)


def test_candidate_config_differs_from_champion_but_shares_detection():
    """The candidate is a linking-only change: detection knobs match the champion,
    only the global-MCF knobs are added."""
    champion = load_champion_config(CHAMPION_CONFIG)
    candidate = json.loads(CANDIDATE_CONFIG.read_text())
    assert candidate != champion
    assert candidate["detect"] == champion["detect"]
    assert candidate["scale"] == champion["scale"]
    assert candidate["link"]["global_window"] == 2
    assert candidate["link"]["birth_cost"] == 3.25
    assert candidate["link"]["death_cost"] == 3.25


def test_candidate_config_resolves_to_globalmcf_params(monkeypatch, tmp_path):
    """The runtime mechanism the candidate kernel uses (BIOHUB_CHAMPION_CONFIG ->
    load_champion_config -> champion_params) yields the global-MCF linker."""
    written = tmp_path / "candidate.json"
    written.write_text(CANDIDATE_CONFIG.read_text())
    monkeypatch.setenv("BIOHUB_CHAMPION_CONFIG", str(written))
    cfg = load_champion_config()
    detect, link, scale = champion_params(cfg)
    assert link.global_window == 2
    assert link.birth_cost == 3.25
    assert link.death_cost == 3.25
    # Detection identical to the champion (linking-only change).
    champ_detect, _l, _s = champion_params(load_champion_config(CHAMPION_CONFIG))
    assert detect.mad_k == champ_detect.mad_k
    assert detect.threshold_percentile == champ_detect.threshold_percentile
