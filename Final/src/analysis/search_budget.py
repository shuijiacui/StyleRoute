"""Compare best-feasible CMA outcomes at evaluation prefixes, without re-searching."""

import argparse
import json

from src.project import project_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--budgets", nargs="+", type=int, default=[40, 80, 160])
    parser.add_argument("--out", default="outputs/validation/landscape/budget_comparison.json")
    args = parser.parse_args()
    path = project_path(args.trajectory)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not metadata["metadata"]["method"].startswith("cma_"):
        raise ValueError("Prefix comparison is for CMA runs only")
    threshold = (metadata["metadata"].get("landscape") or {}).get("minimum_improvement", 1e-9)
    results = []
    for budget in args.budgets:
        if budget < 1 or len(rows) < budget:
            raise ValueError(f"Need at least {budget} real evaluations")
        prefix = rows[:budget]
        eligible = [row for row in prefix if row["feasible"]]
        selected = min(eligible, key=lambda r: r["J"])
        if rows[0]["J"] - selected["J"] < threshold:
            selected = rows[0]
        results.append(
            {
                "budget": budget,
                "selected_evaluation": selected["evaluation_index"],
                "J": selected["J"],
                "A": selected.get("A"),
                "U": selected.get("U"),
                "T_ratio": selected["T"] / rows[0]["T"],
                "feasible_fraction": len(eligible) / budget,
                "improvement_over_baseline": rows[0]["J"] - selected["J"],
            }
        )
    write_json(
        project_path(args.out),
        {
            "source": str(path),
            "results": results,
            "scope": "Best feasible fixed-seed prefixes; not separate runtime measurements",
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
