"""Tests for the SOT-2847 learned-detector kernel build path in build_kernel.py.

The learned build must (a) leave the champion build byte-identical, (b) embed the
torch receptacle module, (c) declare GPU + attached offline weights in metadata
while keeping internet OFF (Code-competition rule), and (d) compose with the
candidate mechanism so BIOHUB_CHAMPION_CONFIG points at a learned_detector config.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_kernel():
    path = REPO_ROOT / "submit" / "build_kernel.py"
    spec = importlib.util.spec_from_file_location("build_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def bk():
    return _load_build_kernel()


def test_champion_build_unchanged_by_learned_feature(bk):
    """The champion build is byte-identical and carries no learned scaffolding."""
    sub = "_pytest_champ_learned_tmp"
    try:
        f1 = bk.build(REPO_ROOT, out_subdir=sub, code_file="k.py")
        text = f1.read_text()
        meta = json.loads((REPO_ROOT / "submit" / sub / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is False
        assert meta["model_sources"] == []
        assert meta["enable_internet"] is False
        assert "learned_detect" not in text  # torch module not embedded
    finally:
        shutil.rmtree(REPO_ROOT / "submit" / sub, ignore_errors=True)


def test_learned_build_embeds_module_and_enables_gpu(bk):
    sub = "_pytest_learned_tmp"
    try:
        kernel = bk.build(
            REPO_ROOT,
            learned=True,
            model_sources=["sota1111/biohub-claude-weights"],
            out_subdir=sub,
            code_file="k.py",
        )
        text = kernel.read_text()
        # Embeds the torch receptacle module (base64-embedded source mentions it).
        assert "learned_detect.py" in text
        compile(text, str(kernel), "exec")  # still a valid self-contained module
        meta = json.loads((REPO_ROOT / "submit" / sub / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is True
        assert meta["model_sources"] == ["sota1111/biohub-claude-weights"]
        # Internet stays OFF — weights come from the attached (offline) source.
        assert meta["enable_internet"] is False
        assert meta["id"] == bk.LEARNED_KERNEL_ID
    finally:
        shutil.rmtree(REPO_ROOT / "submit" / sub, ignore_errors=True)


def test_learned_build_composes_with_candidate_config(bk):
    """learned=True + candidate_config installs a learned_detector-enabled config."""
    sub = "_pytest_learned_cand_tmp"
    learned_cfg = {
        "name": "learned-receptacle-smoke",
        "scale": [1.625, 0.40625, 0.40625],
        "detect": {"sigma_zyx": [1.0, 2.0, 2.0]},
        "link": {"max_distance": 7.0, "min_track_length": 4},
        "learned_detector": {
            "enabled": True,
            "weights": "biohub-claude-weights/detector.pt",
        },
    }
    try:
        kernel = bk.build(
            REPO_ROOT,
            learned=True,
            candidate_config=learned_cfg,
            model_sources=["sota1111/biohub-claude-weights"],
            out_subdir=sub,
            code_file="k.py",
        )
        text = kernel.read_text()
        assert "EMBEDDED_CANDIDATE_CONFIG" in text
        assert "_install_candidate_config()" in text
        assert "learned_detector" in text
        compile(text, str(kernel), "exec")
    finally:
        shutil.rmtree(REPO_ROOT / "submit" / sub, ignore_errors=True)
