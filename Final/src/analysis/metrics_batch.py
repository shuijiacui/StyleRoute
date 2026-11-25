"""Compute contours from the EXACT full-precision image returned by each run."""

import argparse
import csv
import json

from src.experiments.artifacts import read_result
from src.project import project_path, write_json
from src.search.metrics import MetricsConfig, contour_f1, provenance


def run_batch(args):
    root = project_path(args.out_root) / args.split
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for run in summary["runs"]:
        image, content, metadata = read_result(run)
        score = contour_f1(image, content)
        rows.append(
            {
                "run_id": run["run_id"],
                "pair_id": run["pair_id"],
                "method": run["method"],
                "seed": run["seed"],
                "used_default": run["used_default"],
                **score,
                "final_alphas": json.dumps(metadata["region_alphas"]),
            }
        )
    if rows:
        with (root / "contour_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pairs = {}
    for row in rows:
        pairs.setdefault((row["method"], row["pair_id"]), []).append(row["f1"])
    methods = {}
    for (method, _), values in pairs.items():
        methods.setdefault(method, []).append(sum(values) / len(values))
    write_json(
        root / "contour_summary.json",
        {
            "provenance": provenance(MetricsConfig()),
            "run_count": len(rows),
            "methods": {
                m: {"pairs": len(v), "mean_pair_f1": sum(v) / len(v)} for m, v in methods.items()
            },
        },
    )
    print(f"Scored exact saved tensors: {len(rows)} runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["pilot", "evaluation"], default="pilot")
    parser.add_argument("--out-root", default="outputs/experiments/integrated-v2")
    run_batch(parser.parse_args())


if __name__ == "__main__":
    main()
