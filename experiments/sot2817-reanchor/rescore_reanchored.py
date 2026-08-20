"""Re-score every historical champion through the RE-ANCHORED CV (SOT-2817).

Pure re-aggregation: reads the SOT-2816 audit's per-family CV counts
(``experiments/sot2816-oracle-audit/lineage_cv_rescore.json``, produced by the
single leak-free harness over the real ``data/`` volumes) and re-runs them
through the re-anchored :func:`biohub_tracking.eval.cv.aggregate`, so it needs no
competition data. It emits, for each historical config, the full competition
metric (edge + 0.1*division, division term now explicit) plus the new
representativeness views (macro, lineage-macro, 6bba weight share, family-mix
sensitivity), and checks the CV order against the real public-LB submission
points — the SOT-2817 acceptance evidence that promotion runs on the re-anchored
CV, not on public raw values.
"""

from __future__ import annotations

import json
from pathlib import Path

from biohub_tracking.eval.cv import (
    FamilyResult,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUDIT = REPO / "experiments/sot2816-oracle-audit/lineage_cv_rescore.json"

# Real submitted public-LB points (SOT-2816 §1); None = not independently
# submitted. Used only to check CV<->LB *order* — never to select a champion.
PUBLIC_LB = {
    "detect-link-v1": 0.509,
    "detect-link-dog-v2": None,
    "detect-link-dog-v3-adaptive": None,
    "detect-link-dog-v4-shorttrack": 0.624,
}


def _rows(entry: dict) -> list[FamilyResult]:
    rows = []
    for p in entry["per_dataset"]:
        rows.append(
            FamilyResult(
                name=p["name"], lineage=p["lineage"],
                edge_tp=p["edge_tp"], edge_fp=p["edge_fp"], edge_fn=p["edge_fn"],
                edge_jaccard=p["edge_jaccard"], adj_edge_jaccard=p["adjusted_edge_jaccard"],
                division_tp=p["division_tp"], division_fp=p["division_fp"],
                division_fn=p["division_fn"], num_pred_nodes=p["pred_nodes"],
                n_true=(float("nan") if p["n_true"] is None else p["n_true"]),
                weight=p["weight"],
            )
        )
    return rows


def main() -> int:
    audit = json.loads(AUDIT.read_text())
    out = {"issue": "SOT-2817", "source": str(AUDIT.relative_to(REPO)), "configs": []}
    ordered = []
    for entry in audit["lineage_cv"]:
        name = entry["name"]
        res = aggregate(_rows(entry))
        d = cv_result_to_dict(res)
        rep = representativeness_report(res)
        row = {
            "name": name,
            "reanchored_score": d["score"],
            "micro_adj": d["micro_adj_edge_jaccard"],
            "macro_adj": d["macro_adj_edge_jaccard"],
            "lineage_macro_adj": d["lineage_macro_adj"],
            "division_measurable": d["division_measurable"],
            "division_term": d["division_term"],
            "family_mix_sensitive": rep["family_mix_sensitive"],
            "public_lb": PUBLIC_LB.get(name),
        }
        out["configs"].append(row)
        ordered.append(row)

    # CV<->LB order consistency on the two independently submitted points.
    submitted = [r for r in ordered if r["public_lb"] is not None]
    submitted_cv_sorted = sorted(submitted, key=lambda r: r["reanchored_score"])
    submitted_lb_sorted = sorted(submitted, key=lambda r: r["public_lb"])
    order_consistent = [r["name"] for r in submitted_cv_sorted] == [
        r["name"] for r in submitted_lb_sorted
    ]
    out["cv_lb_order_consistent_on_submitted"] = order_consistent
    out["submitted_points"] = [
        {"name": r["name"], "reanchored_score": r["reanchored_score"], "public_lb": r["public_lb"]}
        for r in submitted
    ]

    dest = HERE / "reanchored_cv_rescore.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(REPO)}")
    print(f"CV<->LB order consistent on submitted points: {order_consistent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
