"""SOT-2898 same-seed A/B: precision-first ``mutual-nn`` division overlay.

The champion runs ``allow_division=false`` and predicts zero forks, so the
official metric's ``0.1 · division_jaccard`` term is entirely unrealised
(division_tp = division_fp = 0, division_jaccard = 0). This harness A/Bs a
**precision-first** post-processing overlay (``division_overlay = ("mutual-nn",
...)``) that re-attaches a dropped second daughter ONLY on a high-confidence,
mutual-nearest-neighbour, both-daughters-persist fork — buying division TP
without spraying the fork FPs (which are also edge FPs) that sank SOT-2762 /
SOT-2818.

Single-variable, same-seed: detection + champion linking is run ONCE per family
(the overlay is a pure post-processing pass on the final graph), then every
candidate is scored by copying that base graph and applying the overlay in
memory. The champion baseline (overlay off) reproduces the byte-frozen 0.6649.

Component split (the issue's requirement): every row reports the **edge**
component (micro-adjusted edge Jaccard + per-dataset adjusted edge Jaccard +
summed edge FP) SEPARATELY from the **division** component (division tp/fp/fn +
division_jaccard + 0.1·division term), on the SOT-2817 re-anchored full metric.

NO Kaggle submission (this Issue is CV-only). Champion config is never mutated.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from biohub_tracking.champion import champion_params, load_champion_config  # noqa: E402
from biohub_tracking.division_overlay import apply_division_overlay  # noqa: E402
from biohub_tracking.eval.cv import (  # noqa: E402
    CV_HOLDOUT,
    aggregate,
    representativeness_report,
    score_family,
)
from biohub_tracking.io import (  # noqa: E402
    geff_estimated_num_nodes,
    geff_scale,
    load_geff,
)
from biohub_tracking.pipeline import run_pipeline  # noqa: E402

CHAMP_MICRO = 0.6649  # byte-frozen champion CV (== last-submitted public 0.509 config)

# Precision-first candidate grid. Tuple shape:
#  ("mutual-nn", max_distance, sibling_ratio, min_daughter_len,
#   require_parent_track, require_primary_persist, mutual_margin)
CANDIDATES: dict[str, tuple] = {
    "m0.00_d7_s2_l2": ("mutual-nn", 7.0, 2.0, 2, True, True, 0.0),
    "m0.25_d7_s2_l2": ("mutual-nn", 7.0, 2.0, 2, True, True, 0.25),
    "m0.50_d7_s2_l2": ("mutual-nn", 7.0, 2.0, 2, True, True, 0.50),
    "m1.00_d7_s2_l2": ("mutual-nn", 7.0, 2.0, 2, True, True, 1.00),
    "m0.50_d5_s1.5_l3": ("mutual-nn", 5.0, 1.5, 3, True, True, 0.50),
    "m0.50_d7_s1.2_l2": ("mutual-nn", 7.0, 1.2, 2, True, True, 0.50),
}


def _copy(graph):
    return copy.deepcopy(graph)


def _rows_to_summary(rows) -> dict:
    agg = aggregate(rows)
    rep = representativeness_report(agg)
    return {
        "micro_adj": round(agg.micro_adj_edge_jaccard, 4),
        "lineage_macro_adj": round(agg.lineage_macro_adj, 4),
        "division_jaccard": (
            None
            if agg.division_jaccard != agg.division_jaccard  # NaN
            else round(agg.division_jaccard, 4)
        ),
        "division_term": round(agg.division_term, 4),
        "score": round(agg.score, 4),
        "division_measurable": agg.division_measurable,
        "family_mix_sensitive": rep["family_mix_sensitive"],
        "per_dataset": {
            r.name: {
                "adj_edge_jaccard": round(r.adj_edge_jaccard, 4),
                "edge_tp": r.edge_tp,
                "edge_fp": r.edge_fp,
                "edge_fn": r.edge_fn,
                "division_tp": r.division_tp,
                "division_fp": r.division_fp,
                "division_fn": r.division_fn,
            }
            for r in rows
        },
        "edge_fp_total": sum(r.edge_fp for r in rows),
        "division_tp_total": sum(r.division_tp for r in rows),
        "division_fp_total": sum(r.division_fp for r in rows),
        "division_fn_total": sum(r.division_fn for r in rows),
    }, agg


def main() -> int:
    config = load_champion_config()
    assert config["link"].get("division_overlay") is None, "champion must be overlay-off"
    detect, link, _scale = champion_params(config)

    # One champion pipeline pass per family -> base graph (overlay off), plus GT.
    base: list[dict] = []
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        image = REPO / fam.image
        gt = load_geff(geff)
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)
        graph = run_pipeline(image, scale=scale, detect_params=detect, link_params=link)
        base.append(
            {"fam": fam, "graph": graph, "gt": gt, "scale": scale, "n_true": n_true}
        )
        print(f"[base] {fam.name} nodes={graph.num_nodes} forks={len(graph.dividing_nodes())}")

    def score_with_overlay(params) -> list:
        rows = []
        for b in base:
            g = _copy(b["graph"])
            if params is not None:
                apply_division_overlay(g, tuple(b["scale"]), params)
            rows.append(score_family(b["fam"], g, b["gt"], b["n_true"], scale=b["scale"]))
        return rows

    # Baseline (overlay OFF) must reproduce the byte-frozen champion.
    base_rows = score_with_overlay(None)
    base_summary, base_agg = _rows_to_summary(base_rows)
    champ_ok = abs(base_agg.micro_adj_edge_jaccard - CHAMP_MICRO) < 1e-4
    print(f"[champion] micro_adj={base_summary['micro_adj']} reproduces {CHAMP_MICRO}: {champ_ok}")
    base_per = {r.name: r.adj_edge_jaccard for r in base_rows}

    results = []
    for label, params in CANDIDATES.items():
        rows = score_with_overlay(params)
        summ, agg = _rows_to_summary(rows)
        # No-regression gate on the EDGE component (adjusted edge Jaccard) per
        # dataset: the overlay must not cost any edge TP / add edge FP anywhere.
        deltas = {r.name: round(r.adj_edge_jaccard - base_per[r.name], 4) for r in rows}
        no_edge_regression = all(d >= -1e-9 for d in deltas.values())
        edge_fp_up = summ["edge_fp_total"] > base_summary["edge_fp_total"]
        div_tp = summ["division_tp_total"]
        div_fp = summ["division_fp_total"]
        score_delta = round(agg.score - base_agg.score, 4)
        micro_delta = round(agg.micro_adj_edge_jaccard - base_agg.micro_adj_edge_jaccard, 4)
        # Promotion: earn division TP, no per-dataset edge regression, no edge-FP
        # increase, and a net full-metric score gain.
        promote = (
            div_tp > 0
            and no_edge_regression
            and not edge_fp_up
            and score_delta > 0
            and not summ["family_mix_sensitive"]
        )
        row = {
            "label": label,
            "params": list(params),
            **summ,
            "edge_adj_delta_per_dataset": deltas,
            "no_edge_regression": no_edge_regression,
            "edge_fp_increase": edge_fp_up,
            "micro_delta_vs_champ": micro_delta,
            "score_delta_vs_champ": score_delta,
            "promote": promote,
        }
        results.append(row)
        print(
            f"[cand] {label}: div_tp={div_tp} div_fp={div_fp} "
            f"score={summ['score']} (dScore {score_delta:+.4f}, dMicro {micro_delta:+.4f}) "
            f"edge_fp {base_summary['edge_fp_total']}->{summ['edge_fp_total']} "
            f"no_edge_reg={no_edge_regression} promote={promote}"
        )

    payload = {
        "champion_reproduced": champ_ok,
        "champion": base_summary,
        "candidates": results,
        "any_promote": any(r["promote"] for r in results),
    }
    out = REPO / "experiments" / "sot2898" / "ab_mutual_nn.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    verdict = "PROMOTE-CANDIDATE" if payload["any_promote"] else "REJECTED"
    print(f"VERDICT={verdict} champion_reproduced={champ_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
