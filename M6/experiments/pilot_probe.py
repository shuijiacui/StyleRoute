"""Pilot probe (Phase B): real ImageNet-weight scoring on one content/style pair.

Purpose: BEFORE freezing epsilon/lambda/budget on the real pilot split, measure
the ACTUAL distribution of Q/T across evaluated states and the feasibility
rate at several epsilon values. This script reuses the production path
(M4 engine -> grid regions -> adapter.solve) so probe numbers are comparable
to formal runs.

Usage (from the repo root):
    python -m M6.experiments.pilot_probe \
        --content M4/resources/images/content/pilot/_tmp/mountain.jpg \
        --style   M4/resources/images/styles/met_436535.jpg \
        --method random_states --budget 40 --seed 0 \
        --out M6/experiments/probe_mountain__met_436535.jsonl

Output: one JSON line per evaluated state (full trajectory) plus a printed
summary (q0/t0, feasibility rates at several eps, J range, runtime).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from src.integration import load_adapter  # noqa: E402
from src.models.adain import AdaINEngine  # noqa: E402
from src.project import load_config  # noqa: E402

from M6.regions_grid import build_regions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--method", default="random_states",
                        choices=["risk", "plain", "random_states", "global_scan"])
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    solve = load_adapter("M6.search.adapter:solve")
    engine = AdaINEngine(
        load_config(_REPO_ROOT / "M4" / "resources" / "configs" / "team.json")
    )
    pair = engine.prepare(args.content, args.style, pair_id="probe_pair")
    regions = build_regions(
        pair.content_rgb, feature_hw=pair.feature_hw, device=str(engine.device),
        grid=tuple(args.grid),
    )
    prepared = engine.prepare_regions(pair, regions)
    initial = engine.render_global(pair, 0.6)

    def generate(alphas):
        return engine.render_regions(pair, prepared, tuple(float(a) for a in alphas))

    timing = {"t0": time.perf_counter(), "evals": 0}

    def progress(event: dict) -> None:
        if event.get("event") == "candidate_scored":
            timing["evals"] += 1
            if timing["evals"] % 10 == 0:
                el = time.perf_counter() - timing["t0"]
                print(f"  [{timing['evals']:>3} evals, {el:5.1f}s]", flush=True)

    result = solve(
        generate=generate,
        pair=pair,
        regions=regions,
        initial_result=initial,
        search_config={
            "method": args.method,
            "budget": args.budget,
            "seed": args.seed,
            "lam": args.lam,
            "epsilon": args.epsilon,
            "weights": "IMAGENET1K_V1",
        },
        progress_callback=progress,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q0 = result.metadata["q0"]
    t0 = result.metadata["t0"]
    with out_path.open("w", encoding="utf-8") as fh:
        for record in result.trajectory:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    states = [r for r in result.trajectory if not r["cache_hit"]]
    feasible = [r for r in states if r["feasible"]]
    print("=== probe summary ===")
    print(f"pair           : {args.content} x {args.style}")
    print(f"method/budget  : {args.method} / {args.budget} (seed {args.seed})")
    print(f"q0, t0         : {q0:.6g}, {t0:.6g}")
    print(f"states (new)   : {len(states)}")
    for eps in (0.0, 0.05, 0.10, 0.20):
        n = sum(1 for r in states if r["T"] <= (1 + eps) * t0)
        print(f"feasible eps={eps:<5.0%}: {n:>3}/{len(states)}")
    if states:
        qs = [r["Q"] for r in states]
        ts = [r["T"] for r in states]
        js = [r["J"] for r in states]
        print(f"Q range        : {min(qs):.6g} .. {max(qs):.6g}")
        print(f"T range        : {min(ts):.6g} .. {max(ts):.6g}")
        print(f"J range        : {min(js):.6g} .. {max(js):.6g}")
    print(f"status         : {result.status} (used_default={result.metadata['used_default']})")
    print(f"best alphas    : {result.region_alphas}")
    print(f"decoder_calls  : {result.decoder_calls}, cache_hits: {result.state_cache_hits}")
    print(f"trajectory ->  : {out_path}")


if __name__ == "__main__":
    main()
