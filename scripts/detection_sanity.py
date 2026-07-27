"""Run the champion on all four test datasets and record detection sanity.

The output captures node count, linear track-length distribution, and
per-frame detection counts.  The command fails when a dataset has an empty
frame or a frame count greater than five times its median.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.graph import TrackingGraph
from biohub_tracking.io import load_submission_csv
from biohub_tracking.pipeline import run_pipeline

DATASETS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)


def _track_lengths(graph: TrackingGraph) -> list[int]:
    """Return lengths of maximal non-branching directed track segments."""
    starts = [
        node_id
        for node_id in graph.node_ids()
        if graph.in_degree(node_id) != 1
        or graph.out_degree(graph.predecessors(node_id)[0]) != 1
    ]
    lengths: list[int] = []
    for start in starts:
        length = 1
        current = start
        while graph.out_degree(current) == 1:
            nxt = graph.successors(current)[0]
            if graph.in_degree(nxt) != 1:
                break
            length += 1
            current = nxt
        lengths.append(length)
    return lengths


def summarize_graph(graph: TrackingGraph) -> dict:
    nodes_by_time = graph.nodes_by_time()
    if nodes_by_time:
        frames = range(min(nodes_by_time), max(nodes_by_time) + 1)
        frame_counts = {str(frame): len(nodes_by_time.get(frame, ())) for frame in frames}
    else:
        frame_counts = {}
    counts = list(frame_counts.values())
    lengths = sorted(_track_lengths(graph))
    median_count = statistics.median(counts) if counts else 0.0
    empty_frames = [frame for frame, count in frame_counts.items() if count == 0]
    explosion_limit = 5.0 * median_count
    explosive_frames = [
        frame for frame, count in frame_counts.items() if count > explosion_limit
    ]
    return {
        "node_count": graph.num_nodes,
        "edge_count": graph.num_edges,
        "track_lengths": {
            "count": len(lengths),
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths, default=0),
        },
        "detections_per_frame": frame_counts,
        "frame_count_summary": {
            "min": min(counts, default=0),
            "median": median_count,
            "max": max(counts, default=0),
        },
        "sanity": {
            "empty_frames": empty_frames,
            "explosive_frames": explosive_frames,
            "pass": bool(counts) and not empty_frames and not explosive_frames,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/detection-sanity-all-test.json")
    )
    parser.add_argument(
        "--submission",
        type=Path,
        help="Summarize an already-generated champion submission instead of rerunning images.",
    )
    parser.add_argument("--max-t", type=int, default=None)
    args = parser.parse_args()

    detect_params, link_params, scale = champion_params()
    submitted = load_submission_csv(args.submission) if args.submission else None
    results: dict[str, dict] = {}
    for dataset in DATASETS:
        if submitted is not None:
            if dataset not in submitted:
                raise KeyError(f"submission has no dataset: {dataset}")
            graph = submitted[dataset]
        else:
            image = args.data_dir / f"{dataset}.zarr"
            if not image.exists():
                raise FileNotFoundError(f"missing test dataset: {image}")
            graph = run_pipeline(
                image,
                scale=scale,
                detect_params=detect_params,
                link_params=link_params,
                max_t=args.max_t,
            )
        results[dataset] = summarize_graph(graph)

    report = {
        "champion": load_champion_config()["name"],
        "source": "champion_submission" if submitted is not None else "champion_pipeline",
        "datasets": results,
        "all_sanity_checks_pass": all(
            result["sanity"]["pass"] for result in results.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_sanity_checks_pass"]:
        raise SystemExit("detection sanity failed")


if __name__ == "__main__":
    main()
