"""Exec-compatibility gate tests for the submission stage (SOT-1984).

Enforces that the champion submission code runs under the constraints of a Kaggle
submission runtime — no ``__file__``, undefined cwd, no filesystem access to the
repo — and that the embedded champion config never drifts from the checked-in
``champion/config.json``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from biohub_tracking import champion as champion_mod
from biohub_tracking.champion import (
    EMBEDDED_CHAMPION_CONFIG,
    champion_params,
    load_champion_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_CONFIG = REPO_ROOT / "champion" / "config.json"


def _load_gate_module():
    """Import ``submit/exec_compat_gate.py`` (not on the package path)."""
    path = REPO_ROOT / "submit" / "exec_compat_gate.py"
    spec = importlib.util.spec_from_file_location("exec_compat_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_config_matches_champion_file():
    """The embedded fallback must be byte-for-byte the checked-in champion."""
    on_disk = json.loads(CHAMPION_CONFIG.read_text())
    assert EMBEDDED_CHAMPION_CONFIG == on_disk


def test_load_champion_config_falls_back_to_embedded(monkeypatch, tmp_path):
    """With no config file anywhere, resolution returns the embedded copy."""
    monkeypatch.delenv("BIOHUB_CHAMPION_CONFIG", raising=False)
    # Neutralise every on-disk candidate so only the embedded fallback remains.
    monkeypatch.setattr(champion_mod, "_candidate_config_paths", lambda: [tmp_path / "nope.json"])
    cfg = load_champion_config()
    assert cfg == EMBEDDED_CHAMPION_CONFIG
    # And params still build from the fallback.
    detect, link, scale = champion_params(cfg)
    assert detect.threshold_percentile == 99.3
    assert link.allow_division is False
    assert scale == (1.625, 0.40625, 0.40625)


def test_load_champion_config_honours_env_var(monkeypatch, tmp_path):
    """``BIOHUB_CHAMPION_CONFIG`` overrides the resolution order."""
    custom = tmp_path / "custom.json"
    cfg = dict(EMBEDDED_CHAMPION_CONFIG)
    cfg["detect"] = dict(cfg["detect"], threshold_percentile=98.0)
    custom.write_text(json.dumps(cfg))
    monkeypatch.setenv("BIOHUB_CHAMPION_CONFIG", str(custom))
    detect, _link, _scale = champion_params(load_champion_config())
    assert detect.threshold_percentile == 98.0


def test_exec_compat_gate_passes():
    """The full gate runs the champion under exec()/no-__file__/undefined cwd."""
    gate = _load_gate_module()
    cwd_before = os.getcwd()
    summary = gate.run_gate()
    assert os.getcwd() == cwd_before, "gate must restore the working directory"
    assert summary["num_nodes"] > 0
    assert summary["datasets"] == ["smoke"]
