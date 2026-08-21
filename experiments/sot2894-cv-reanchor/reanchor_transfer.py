"""SOT-2894 — re-anchor the leak-free CV to the CV↔public negative-transfer.

Ties the diagnostic module :mod:`biohub_tracking.eval.transfer` to the live data:

1. Re-runs the reigning champion through the single leak-free CV and asserts it
   still reproduces the registry 0.6649 (champion byte-frozen regression guard).
2. Cross-checks every :data:`HISTORICAL_LINEAGE` config's hard-coded per-family
   counts against the SOT-2816 single-harness live re-score
   (``experiments/sot2816-oracle-audit/lineage_cv_rescore.json``) so the
   reconstructed lineage is not stale.
3. Emits the ranking table (criterion 1), the per-statistic order-consistency vs
   the same-metric public anchors (criterion 2/3), and a Markdown summary.

Read-only w.r.t. champion state. **No Kaggle submission.** Requires the
(gitignored) ``data/`` volumes only for step 1; steps 2-3 are pure.

Usage::

    .venv/bin/python experiments/sot2894-cv-reanchor/reanchor_transfer.py
    .venv/bin/python experiments/sot2894-cv-reanchor/reanchor_transfer.py --no-live  # skip step 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_tracking.eval.transfer import (
    HISTORICAL_LINEAGE,
    transfer_report,
    transfer_stats,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = REPO / "experiments/sot2816-oracle-audit/lineage_cv_rescore.json"


def _check_champion_live() -> dict:
    from biohub_tracking.eval.cv import (
        CHAMPION_REFERENCE_MICRO_ADJ,
        cv_result_to_dict,
        evaluate_cv,
    )

    res = evaluate_cv(repo_root=REPO)
    delta = abs(res.micro_adj_edge_jaccard - CHAMPION_REFERENCE_MICRO_ADJ)
    assert delta <= 1e-4, f"champion CV drift {res.micro_adj_edge_jaccard} != 0.6649"
    return {"champion_micro_adj": round(res.micro_adj_edge_jaccard, 4), "delta": round(delta, 6),
            "cv": cv_result_to_dict(res)}


def _crosscheck_cache() -> list[str]:
    """Assert the hard-coded lineage counts match the live SOT-2816 re-score."""
    notes: list[str] = []
    if not CACHE.exists():
        return ["cache absent — skipped cross-check (run rescore_lineage.py first)"]
    cache = {c["name"]: c for c in json.loads(CACHE.read_text())["lineage_cv"]}
    for cfg in HISTORICAL_LINEAGE:
        live = cache.get(cfg.name)
        if live is None:
            notes.append(f"{cfg.name}: NOT in cache")
            continue
        live_fam = {r["name"]: r for r in live["per_dataset"]}
        for row in cfg.rows:
            lr = live_fam[row.name]
            for field, got in (("edge_tp", row.edge_tp), ("edge_fp", row.edge_fp),
                               ("edge_fn", row.edge_fn), ("pred_nodes", row.num_pred_nodes)):
                assert lr[field] == got, f"{cfg.name}/{row.name}.{field}: {lr[field]} != {got}"
        assert round(live["micro_adj_edge_jaccard"], 4) == round(transfer_stats(cfg).micro_adj, 4)
        notes.append(f"{cfg.name}: per-family counts + micro_adj match live re-score ✓")
    return notes


def _markdown(report: dict) -> str:
    lines = ["| config | issue | micro_adj (legacy) | micro_raw | lineage_macro_raw (re-anchor) | node-penalty | public LB |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["ranking_table"]:
        lb = "—" if r["public_lb"] is None else r["public_lb"]
        lines.append(f"| {r['name']} | {r['issue']} | {r['micro_adj']} | {r['micro_raw']} | "
                     f"**{r['lineage_macro_raw']}** | {r['node_penalty_contribution']:+} | {lb} |")
    lines.append("")
    lines.append("Order-consistency vs same-metric public anchors (Spearman ρ; excludes unconfirmed v3):")
    lines.append("")
    lines.append("| statistic | ρ vs public | champion is CV-top? |")
    lines.append("| --- | ---: | :---: |")
    for stat, oc in report["order_consistency"].items():
        lines.append(f"| {stat} | {oc['spearman_vs_public']} | {oc['cv_top_matches_public_top']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-live", action="store_true", help="Skip the disk-backed champion re-run.")
    ap.add_argument("--out", type=str, default=str(HERE / "transfer_report.json"))
    args = ap.parse_args()

    report = transfer_report(list(HISTORICAL_LINEAGE))
    crosscheck = _crosscheck_cache()
    champion = None if args.no_live else _check_champion_live()

    report["champion_live_guard"] = champion
    report["lineage_crosscheck"] = crosscheck
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "ranking_table.md").write_text(_markdown(report))

    print("=== SOT-2894 CV↔public transfer diagnosis ===")
    print(_markdown(report))
    print("cross-check vs live SOT-2816 re-score:")
    for n in crosscheck:
        print("  -", n)
    if champion:
        print(f"champion live guard: micro_adj {champion['champion_micro_adj']} "
              f"(delta {champion['delta']}) == 0.6649 ✓")
    print(f"\nwrote {args.out} and {HERE / 'ranking_table.md'}")


if __name__ == "__main__":
    main()
