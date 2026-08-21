"""SOT-2932 same-seed A/B: decoupled ``split-signature`` division-event overlay.

Explore direction D — harvest the forgone ``0.1 · division_jaccard`` term. The
champion runs ``allow_division=false`` and predicts zero forks, so the division
Jaccard is 0.0 (division_tp = division_fp = 0). This harness A/Bs a NEW-formulation
post-processing overlay (``division_overlay = ("split-signature", ...)``) that is
**structurally different** from every prior division attempt:

* SOT-2762 turned division ON inside the linking LAP (fork FP spray) — REJECTED.
* SOT-2818 nearest-head / SOT-2898 mutual-nn are graph-geometry-only overlays that
  re-attach a nearby persistent head by **position alone** — REJECTED (the picked
  head is usually not the true daughter, so the added edge is a division FP *and*
  an edge FP and the edge component regresses).

The ``split-signature`` overlay keeps the non-destructive additive contract but
proposes a fork ONLY on a **local strong split signature decoupled from the linking
cost**: image intensity condensation (parent + both daughters bright in their own
frame) plus bipolar straddle geometry (the two daughters separate to opposite sides
of the parent). ``node_response`` is the champion DoG response sampled at each
node's voxel, read from the image ONCE per family.

Single-variable, same-seed: detection + champion linking is run ONCE per family;
every candidate is scored by copying that base graph and applying the overlay in
memory. The baseline (overlay off) reproduces the live champion byte-for-byte.

Component split (the issue's requirement): every row reports the EDGE component
(micro-adjusted edge Jaccard + per-dataset adjusted edge Jaccard + summed edge FP)
SEPARATELY from the DIVISION component (division tp/fp/fn + division_jaccard +
0.1·division term), on the SOT-2817 re-anchored full metric.

NO Kaggle submission (this Issue is CV-only). Champion config is never mutated.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from biohub_tracking.champion import champion_params, load_champion_config  # noqa: E402
from biohub_tracking.detect import (  # noqa: E402
    _compute_response,
    _normalize_intensity,
)
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
from biohub_tracking.pipeline import _open_image_array, run_pipeline  # noqa: E402

# Split-signature candidate grid. Tuple shape:
#  ("split-signature", max_distance, sibling_ratio, min_daughter_len,
#   require_parent_track, require_primary_persist, mutual_margin,
#   straddle_max, parent_bright_pct, daughter_bright_pct)
#
# straddle_max in [0,2]: 0 = perfectly opposed daughters (bipolar), 2 = no gate.
# bright_pct in [0,1]: per-frame response percentile floor (0 = no intensity gate).
CANDIDATES: dict[str, tuple] = {
    # Geometry-only reference (straddle gate, no intensity) — isolates the new
    # bipolar shape signal against SOT-2898 mutual-nn.
    "geom_str0.8": ("split-signature", 7.0, 2.0, 2, True, True, 0.50, 0.8, 0.0, 0.0),
    "geom_str0.6": ("split-signature", 7.0, 2.0, 2, True, True, 0.50, 0.6, 0.0, 0.0),
    # Full split-signature: straddle + parent/daughter brightness condensation.
    "sig_str0.8_p0.6_d0.5": ("split-signature", 7.0, 2.0, 3, True, True, 0.50, 0.8, 0.6, 0.5),
    "sig_str0.7_p0.7_d0.6": ("split-signature", 7.0, 1.5, 3, True, True, 0.50, 0.7, 0.7, 0.6),
    "sig_str0.6_p0.8_d0.7": ("split-signature", 6.0, 1.5, 3, True, True, 0.75, 0.6, 0.8, 0.7),
    "sig_str0.5_p0.85_d0.8": ("split-signature", 6.0, 1.5, 3, True, True, 1.00, 0.5, 0.85, 0.8),
}


def _node_response(image_path: Path, graph, detect_params, scale) -> dict[int, float]:
    """Champion DoG response sampled at every node's voxel, keyed by node id.

    The response is the *same* map the champion detector thresholds
    (:func:`_compute_response` on the normalized volume), computed per timepoint
    only for frames that carry graph nodes. Sampling at the node's rounded
    ``(z, y, x)`` voxel gives a decoupled brightness feature for the overlay's
    intensity-condensation gates — never re-running the linker.
    """
    arr = _open_image_array(image_path)
    nz, ny, nx = arr.shape[1], arr.shape[2], arr.shape[3]
    nodes_by_time = graph.nodes_by_time()
    resp: dict[int, float] = {}
    for t, ids in nodes_by_time.items():
        vol = np.asarray(arr[t], dtype=np.float32)
        vol = _normalize_intensity(vol, detect_params.intensity_norm)
        response = _compute_response(vol, detect_params)
        for n in ids:
            _t, z, y, x = graph.coords[n]
            zi = int(min(max(round(z), 0), nz - 1))
            yi = int(min(max(round(y), 0), ny - 1))
            xi = int(min(max(round(x), 0), nx - 1))
            resp[n] = float(response[zi, yi, xi])
    return resp


def _rows_to_summary(rows) -> tuple[dict, object]:
    agg = aggregate(rows)
    rep = representativeness_report(agg)
    summary = {
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
    }
    return summary, agg


def main() -> int:
    config = load_champion_config()
    assert config["link"].get("division_overlay") is None, "champion must be overlay-off"
    detect, link, _scale = champion_params(config)

    base: list[dict] = []
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        image = REPO / fam.image
        gt = load_geff(geff)
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)
        graph = run_pipeline(image, scale=scale, detect_params=detect, link_params=link)
        resp = _node_response(image, graph, detect, scale)
        base.append(
            {
                "fam": fam,
                "graph": graph,
                "gt": gt,
                "scale": scale,
                "n_true": n_true,
                "resp": resp,
            }
        )
        print(
            f"[base] {fam.name} nodes={graph.num_nodes} "
            f"forks={len(graph.dividing_nodes())} resp_sampled={len(resp)}"
        )

    def score_with_overlay(params) -> list:
        rows = []
        for b in base:
            g = copy.deepcopy(b["graph"])
            if params is not None:
                apply_division_overlay(
                    g, tuple(b["scale"]), params, node_response=b["resp"]
                )
            rows.append(
                score_family(b["fam"], g, b["gt"], b["n_true"], scale=b["scale"])
            )
        return rows

    base_rows = score_with_overlay(None)
    base_summary, base_agg = _rows_to_summary(base_rows)
    champ_micro = base_summary["micro_adj"]
    print(f"[champion] micro_adj={champ_micro} score={base_summary['score']} "
          f"div_jaccard={base_summary['division_jaccard']}")
    base_per = {r.name: r.adj_edge_jaccard for r in base_rows}

    results = []
    for label, params in CANDIDATES.items():
        rows = score_with_overlay(params)
        summ, agg = _rows_to_summary(rows)
        deltas = {r.name: round(r.adj_edge_jaccard - base_per[r.name], 4) for r in rows}
        no_edge_regression = all(d >= -1e-9 for d in deltas.values())
        edge_fp_up = summ["edge_fp_total"] > base_summary["edge_fp_total"]
        div_tp = summ["division_tp_total"]
        div_fp = summ["division_fp_total"]
        score_delta = round(agg.score - base_agg.score, 4)
        micro_delta = round(agg.micro_adj_edge_jaccard - base_agg.micro_adj_edge_jaccard, 4)
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
        "champion": base_summary,
        "candidates": results,
        "any_promote": any(r["promote"] for r in results),
    }
    out = REPO / "experiments" / "sot2932" / "ab_split_signature.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    verdict = "PROMOTE-CANDIDATE" if payload["any_promote"] else "REJECTED"
    print(f"VERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
