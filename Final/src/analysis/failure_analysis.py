"""Report actual fallback and termination counts from saved run summaries."""

import argparse
import json
from collections import Counter, defaultdict

from src.project import project_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="pilot", choices=["pilot", "evaluation"])
    parser.add_argument("--out-root", default="outputs/experiments/integrated-v2")
    args = parser.parse_args()
    root = project_path(args.out_root) / args.split
    rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))["runs"]
    methods = defaultdict(list)
    for row in rows:
        methods[row["method"]].append(row)
    report = {
        m: {
            "runs": len(runs),
            "fallback_runs": sum(r["used_default"] for r in runs),
            "statuses": dict(Counter(r["status"] for r in runs)),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in runs)),
            "no_improvement_pairs": sorted({r["pair_id"] for r in runs if r["used_default"]}),
        }
        for m, runs in methods.items()
    }
    write_json(
        root / "failure_analysis.json",
        {
            "methods": report,
            "scope": "Observed status/termination counts; "
            "no automatic causal claim about image quality.",
        },
    )
    print(f"Analysed {len(rows)} saved runs: {root / 'failure_analysis.json'}")


if __name__ == "__main__":
    main()
