"""SOT-2884 A/B screen: Ultrack multi-hypothesis detection selection by temporal support.

Runs the leak-free 4-family LOFO CV for the byte-frozen champion (baseline) and for
each hypothesis-select variant, printing per-family edge TP/FP/FN + adjusted Jaccard
and the per-dataset non-regression verdict. Kaggle is never touched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from biohub_tracking.champion import EMBEDDED_CHAMPION_CONFIG
from biohub_tracking.eval.cv import evaluate_cv, cv_result_to_dict


def _variant_config(mad_k_low: float, min_track: int, pool_cap: int = 20000) -> dict:
    d = dict(EMBEDDED_CHAMPION_CONFIG["detect"])
    d["detect_hypothesis_select"] = True
    d["hypothesis_mad_k_low"] = mad_k_low
    d["hypothesis_min_track"] = min_track
    d["hypothesis_pool_cap"] = pool_cap
    return {
        "name": f"hypsel_low{mad_k_low}_L{min_track}",
        "detect": d,
        "link": dict(EMBEDDED_CHAMPION_CONFIG["link"]),
        "scale": EMBEDDED_CHAMPION_CONFIG["scale"],
    }


def _per_family(result_dict: dict) -> dict:
    return {
        r["name"]: (r["edge_tp"], r["edge_fp"], r["edge_fn"],
                    r["adjusted_edge_jaccard"], r["pred_nodes"])
        for r in result_dict["per_dataset"]
    }


def main() -> int:
    baseline = json.loads(Path("docs/ai/sot2884/champion_baseline.json").read_text())
    base_fam = _per_family(baseline)
    base_micro = baseline["micro_adj_edge_jaccard"]
    print(f"BASELINE micro_adj={base_micro:.4f}")
    for name, (tp, fp, fn, adj, pn) in base_fam.items():
        print(f"  {name:16} TP={tp:5} FP={fp:5} FN={fn:5} adj={adj:.4f} pred_nodes={pn}")

    # (mad_k_low, min_track) grid from the CLI, else a default screen grid.
    grid = []
    for arg in sys.argv[1:]:
        low, L = arg.split(",")
        grid.append((float(low), int(L)))
    if not grid:
        grid = [(2.5, 3), (2.0, 3), (2.5, 2), (2.0, 4)]

    out_all = {"baseline": baseline, "variants": []}
    for mad_k_low, L in grid:
        cfg = _variant_config(mad_k_low, L)
        t0 = time.time()
        res = cv_result_to_dict(evaluate_cv(cfg))
        dt = time.time() - t0
        fam = _per_family(res)
        micro = res["micro_adj_edge_jaccard"]
        regressed = []
        for name, (tp, fp, fn, adj, pn) in fam.items():
            b_adj = base_fam[name][3]
            if adj < b_adj - 1e-9:
                regressed.append(name)
        verdict = "NON-REGRESSING" if not regressed else f"REGRESSED:{regressed}"
        gained = micro > base_micro + 1e-9
        print(f"\nVARIANT low={mad_k_low} L={L}  micro_adj={micro:.4f} "
              f"(base {base_micro:.4f}, {'+' if gained else ''}{micro-base_micro:.4f})  "
              f"[{verdict}]  ({dt:.0f}s)")
        for name, (tp, fp, fn, adj, pn) in fam.items():
            b_tp, b_fp, b_fn, b_adj, b_pn = base_fam[name]
            flag = "" if adj >= b_adj - 1e-9 else "  <-- REGRESS"
            print(f"  {name:16} TP={tp:5}({tp-b_tp:+d}) FP={fp:5}({fp-b_fp:+d}) "
                  f"FN={fn:5}({fn-b_fn:+d}) adj={adj:.4f}({adj-b_adj:+.4f}) "
                  f"pred_nodes={pn}({pn-b_pn:+d}){flag}")
        out_all["variants"].append({
            "mad_k_low": mad_k_low, "min_track": L, "seconds": dt,
            "micro_adj": micro, "non_regressing": not regressed,
            "gained": gained, "result": res,
        })
        Path("docs/ai/sot2884/ab_results.json").write_text(
            json.dumps(out_all, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
