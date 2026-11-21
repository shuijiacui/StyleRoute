"""Batch experiment runner (Phase C): run the full M6 method x seed matrix over
the pilot or evaluation split defined by M4's image manifest.

Production path only: M4 engine -> region adapter -> M6.search.adapter:solve.
The engine/pair/regions/default result are prepared ONCE per pair and reused
across methods and seeds (encoder caching is the M4 contract, not an M6 trick).
Pairs whose image files are missing are skipped with a warning so the matrix
can start before the full dataset has landed.

Outputs (per README section 8 logging requirements):
    M6/outputs/<split>/<pair_id>/<method>__seed<S>.jsonl   full trajectory
    M6/outputs/<split>/summary.json                        one record per run
    M6/outputs/<split>/runs.csv                            flat table for analysis

Usage (from the repo root):
    python -m M6.experiments.run --split pilot
    python -m M6.experiments.run --split evaluation --methods risk plain --seeds 0 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.integration import load_adapter, validate_search_result  # noqa: E402
from src.models.adain import AdaINEngine  # noqa: E402
from src.project import load_config  # noqa: E402

_IMAGES_ROOT = _REPO_ROOT / "M4" / "resources" / "images"


def _id_path_map(manifest: dict, key: str) -> dict[str, str]:
    """Map content_id/style_id -> local file path from the manifest."""
    mapping = {}
    for entry in manifest.get(key, []):
        local = entry.get("local_path")
        if local:
            mapping[entry["id"]] = str(_IMAGES_ROOT.parent.parent / local.replace("/", "\\"))
    return mapping


def _config_hash(search_config: dict) -> str:
    payload = json.dumps(search_config, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def run_matrix(args) -> None:
    exp_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads((_IMAGES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    scoring = exp_cfg["scoring"]
    methods = args.methods or exp_cfg["methods"]
    seeds = args.seeds or exp_cfg["seeds"]
    budget = args.budget or exp_cfg["budget"]

    contents = _id_path_map(manifest, "content_images")
    styles = _id_path_map(manifest, "style_images")
    pairs = manifest.get(f"{args.split}_pairs", [])
    if args.content and args.style:
        # 开发单图对模式：图不在 manifest 里（如临时替代图），provenance 由 pair_id 标明
        dev_id = args.pair_id or "dev_pair"
        pairs = [{"pair_id": dev_id, "content_id": "__dev", "style_id": "__dev"}]
        contents = {"__dev": args.content}
        styles = {"__dev": args.style}
    if args.pairs:
        keep = set(args.pairs)
        pairs = [p for p in pairs if p["pair_id"] in keep]

    solve = load_adapter("M6.search.adapter:solve")
    engine = AdaINEngine(load_config(args.team_config or (_REPO_ROOT / "M4" / "resources" / "configs" / "team.json")))
    build_regions = load_adapter(engine.config["region_adapter"])

    out_root = Path(args.out_root) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    summary_records = []
    skipped = []

    for pair_info in pairs:
        pair_id = pair_info["pair_id"]
        content_path = contents.get(pair_info["content_id"])
        style_path = styles.get(pair_info["style_id"])
        if not content_path or not Path(content_path).exists() or not Path(style_path or "").exists():
            skipped.append({"pair_id": pair_id, "reason": "missing image file(s)"})
            print(f"[SKIP] {pair_id}: image file(s) missing", flush=True)
            continue

        print(f"[PAIR] {pair_id} ({content_path} x {style_path})", flush=True)
        pair = engine.prepare(content_path, style_path, pair_id=pair_id)
        regions = build_regions(
            pair.content_rgb, feature_hw=pair.feature_hw, device=str(engine.device)
        )
        prepared = engine.prepare_regions(pair, regions)
        initial = engine.render_global(pair, engine.config["default_alpha"])

        def generate(alphas):
            return engine.render_regions(pair, prepared, tuple(float(a) for a in alphas))

        for method in methods:
            for seed in seeds:
                search_config = {
                    "method": method,
                    "budget": budget,
                    "max_restarts": exp_cfg["max_restarts"],
                    "seed": seed,
                    "levels": exp_cfg["levels"],
                    "lam": scoring["lam"],
                    "epsilon": scoring["epsilon"],
                    "delta": scoring["delta"],
                    "delta_t": scoring["delta_t"],
                    "weights": scoring["weights"],
                }
                run_id = f"{args.split}/{pair_id}/{method}/seed{seed}"
                t0 = time.perf_counter()
                result = solve(
                    generate=generate,
                    pair=pair,
                    regions=regions,
                    initial_result=initial,
                    search_config=search_config,
                )
                elapsed = time.perf_counter() - t0
                validate_search_result(
                    result, pair, prepared, tuple(engine.config["allowed_region_alphas"])
                )

                cfg_hash = _config_hash(search_config)
                pair_dir = out_root / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                traj_path = pair_dir / f"{method}__seed{seed}.jsonl"
                with traj_path.open("w", encoding="utf-8") as fh:
                    for record in result.trajectory:
                        fh.write(json.dumps(
                            {
                                "run_id": run_id,
                                "pair_id": pair_id,
                                "method": method,
                                "seed": seed,
                                "config_hash": cfg_hash,
                                **record,
                            },
                            ensure_ascii=False,
                        ) + "\n")

                summary_records.append(
                    {
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "method": method,
                        "seed": seed,
                        "config_hash": cfg_hash,
                        "status": result.status,
                        "used_default": result.metadata["used_default"],
                        "stop_reason": result.metadata["stop_reason"],
                        "Q": result.metrics["Q"],
                        "T": result.metrics["T"],
                        "S": result.metrics["S"],
                        "J": result.metrics["J"],
                        "q0": result.metadata["q0"],
                        "t0": result.metadata["t0"],
                        "candidate_count": result.candidate_count,
                        "decoder_calls": result.decoder_calls,
                        "state_cache_hits": result.state_cache_hits,
                        "restarts_used": result.metadata["restarts_used"],
                        "elapsed_s": round(elapsed, 3),
                        "trajectory": str(traj_path.relative_to(_REPO_ROOT)),
                    }
                )
                print(
                    f"  [{method}/seed{seed}] {result.status} "
                    f"J={result.metrics['J']:.5f} calls={result.decoder_calls} "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

    # 断点续跑合并：按 run_id 去重，后跑的覆盖同 run_id 旧记录
    merged: dict[str, dict] = {}
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        seen_skip = {s["pair_id"] for s in old.get("skipped", [])}
        skipped = old.get("skipped", []) + [s for s in skipped if s["pair_id"] not in seen_skip]
        for record in old.get("runs", []):
            merged[record["run_id"]] = record
    for record in summary_records:
        merged[record["run_id"]] = record
    all_records = sorted(merged.values(), key=lambda r: r["run_id"])

    summary_path.write_text(
        json.dumps(
            {"split": args.split, "skipped": skipped, "runs": all_records},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if all_records:
        with (out_root / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
    summary_records = all_records
    print(
        f"=== done: {len(summary_records)} runs, {len(skipped)} skipped -> {out_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-config", default=str(_REPO_ROOT / "M4" / "resources" / "configs" / "team_m6dev.json"))
    parser.add_argument("--split", default="pilot", choices=["pilot", "evaluation"])
    parser.add_argument("--config", default=str(_REPO_ROOT / "M6" / "configs" / "experiment.json"))
    parser.add_argument("--out-root", default=str(_REPO_ROOT / "M6" / "outputs"))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--content", default=None, help="dev 模式：单图对内容图路径")
    parser.add_argument("--style", default=None, help="dev 模式：单图对风格图路径")
    parser.add_argument("--pair-id", default=None, help="dev 模式：图对标识（默认 dev_pair）")
    run_matrix(parser.parse_args())


if __name__ == "__main__":
    main()
