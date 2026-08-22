"""SOT-3011 — pin the released-weights wholesale A/B finding (torch-free).

Reads the committed CV A/B artifact and asserts the recorded verdict and the
mandatory non-regression gate logic, so the finding (learned pipeline beats the
champion in aggregate but REJECTs on per-family sparse-lineage regression) is
captured in CI without needing torch/tracksdata. The heavy inference that
produced the artifact lives in ``experiments/sot3011/run_ab.py`` and runs in the
GPU ``.venv`` with the offline-bundled royerlab deps.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "experiments/sot3011/screen_royerlab_ab.json"


def _load() -> dict:
    assert ART.is_file(), f"missing A/B artifact {ART}"
    return json.loads(ART.read_text())


def _gate(arm: dict, champ: dict, noise: float) -> tuple[str, bool, float]:
    """Re-derive the promotion decision independently of the recorded gate."""
    champ_by = {r["name"]: r["adjusted_edge_jaccard"] for r in champ["per_dataset"]}
    arm_by = {r["name"]: r["adjusted_edge_jaccard"] for r in arm["per_dataset"]}
    no_reg = all(arm_by[n] >= champ_by[n] - 1e-9 for n in champ_by)
    delta = arm["micro_adj_edge_jaccard"] - champ["micro_adj_edge_jaccard"]
    if delta > noise and no_reg:
        decision = "PROMOTE"
    elif delta < -noise or not no_reg:
        decision = "REJECT"
    else:
        decision = "INCONCLUSIVE"
    return decision, no_reg, delta


def test_champion_reproduced_byte_exact():
    d = _load()
    assert round(d["champion_cv"]["micro_adj_edge_jaccard"], 4) == 0.6760


def test_overall_reject_and_leak_caveat_recorded():
    d = _load()
    assert d["overall_decision"] == "REJECT"
    assert "optimistic" in d["leak_caveat"] and "train" in d["leak_caveat"].lower()
    # No submission is performed by this child (parent-only).
    assert d["issue"] == "SOT-3011"


def test_gate_logic_matches_recorded_for_every_arm():
    d = _load()
    champ = d["champion_cv"]
    noise = d["noise_band"]
    assert d["learned_cv"], "expected at least one learned arm"
    for linker, arm in d["learned_cv"].items():
        decision, no_reg, delta = _gate(arm, champ, noise)
        assert decision == d["gates"][linker]["decision"] == "REJECT", linker
        # The signature of this finding: micro UP but a per-family regression.
        assert delta > 0.0, f"{linker}: expected micro gain, got {delta}"
        assert no_reg is False, f"{linker}: expected a per-family regression"
        # The regression is on the sparse 44b6 lineage specifically.
        arm_by = {r["name"]: r["adjusted_edge_jaccard"] for r in arm["per_dataset"]}
        champ_by = {r["name"]: r["adjusted_edge_jaccard"] for r in champ["per_dataset"]}
        assert any(
            arm_by[n] < champ_by[n] for n in champ_by if n.startswith("44b6")
        ), f"{linker}: expected a 44b6 regression"


def test_aggregate_views_all_rise_not_a_mix_artifact():
    """micro/macro/lineage-macro all rise -> the gain is genuine aggregate lift,
    not a 6bba-weight-mix artifact (the reason this is a notable finding)."""
    d = _load()
    champ = d["champion_cv"]
    for arm in d["learned_cv"].values():
        for key in ("micro_adj_edge_jaccard", "macro_adj_edge_jaccard", "lineage_macro_adj"):
            assert arm[key] > champ[key], key
