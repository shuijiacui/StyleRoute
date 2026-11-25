"""Run a reproducible matrix from the project root; save exact returned tensors and regions."""

import argparse
import csv
import inspect
import json
import re
import time

import numpy as np

from src.integration import load_adapter, validate_search_result
from src.models.adain import AdaINEngine
from src.models.preprocessing import png_bytes
from src.outputs import save_run
from src.project import ROOT, digest, file_sha256, load_config, project_path, write_json
from src.search.cma_search import CMA_METHODS


def _relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def _persist(out_root, records, skipped, experiment):
    rows = sorted(records.values(), key=lambda r: r["run_id"])
    write_json(
        out_root / "summary.json",
        {"version": "integrated-v2", "experiment": experiment, "skipped": skipped, "runs": rows},
    )
    if rows:
        with (out_root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_matrix(args):
    exp = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(
        project_path("resources/images/manifest.json").read_text(encoding="utf-8")
    )
    methods = exp["methods"] if args.methods is None else args.methods
    seeds = exp["seeds"] if args.seeds is None else args.seeds
    budget = exp["budget"] if args.budget is None else args.budget
    if not methods or not seeds:
        raise ValueError("Provide at least one method and seed")
    if bool(args.content) != bool(args.style):
        raise ValueError("--content and --style must be supplied together")
    paths = {
        e["id"]: project_path(e["local_path"])
        for key in ["content_images", "style_images"]
        for e in manifest[key]
    }
    pairs = manifest[f"{args.split}_pairs"]
    if args.content:
        pairs = [
            {
                "pair_id": args.pair_id or "development_pair",
                "content_id": "__content",
                "style_id": "__style",
            }
        ]
        paths.update({"__content": project_path(args.content), "__style": project_path(args.style)})
    if args.pairs:
        known = {p["pair_id"] for p in pairs}
        if set(args.pairs) - known:
            raise ValueError(f"Unknown pairs: {set(args.pairs) - known}")
        pairs = [p for p in pairs if p["pair_id"] in args.pairs]
    engine = AdaINEngine(load_config(args.team_config))
    if engine.config["default_alpha"] != 0.6:
        raise ValueError("Experiments require a global AdaIN baseline at strength 0.6")
    source_fingerprint = digest(
        {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for folder in ["models", "regions", "search"]
            for path in (ROOT / "src" / folder).glob("*.py")
        }
    )
    evaluator_path = project_path(
        exp["scoring"].get("checkpoint", "resources/weights/vgg19_features.pth")
    )
    evaluator_fingerprint = (
        file_sha256(evaluator_path)
        if exp["scoring"].get("weights", "IMAGENET1K_V1") != "RANDOM" and evaluator_path.is_file()
        else exp["scoring"].get("weights", "IMAGENET1K_V1")
    )
    build_regions = load_adapter(engine.config["region_adapter"])
    solve = load_adapter(engine.config["search_adapter"])
    out_root = project_path(args.out_root) / args.split
    out_root.resolve().relative_to(ROOT)  # Validate before writing artifacts.
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    records = {}
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("version") != "integrated-v2":
            raise ValueError("Choose a fresh output root for the corrected experiment version")
        records = {r["run_id"]: r for r in old["runs"]}
    skipped = []
    for pair_info in pairs:
        pair_id = pair_info["pair_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", pair_id):
            raise ValueError("pair_id must be a simple filename")
        content, style = paths[pair_info["content_id"]], paths[pair_info["style_id"]]
        if not content.is_file() or not style.is_file():
            skipped.append({"pair_id": pair_id, "reason": "missing images"})
            continue
        pair = engine.prepare(content, style, pair_id=pair_id)
        regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=engine.device)
        prepared = engine.prepare_regions(pair, regions)
        initial = engine.render_global(pair, 0.6)
        pair_dir = out_root / pair_id
        artifact_dir = pair_dir / "artifacts"
        if not (artifact_dir / "runtime.json").exists():
            save_run(artifact_dir, engine, pair, initial, content, style, regions=regions)
        else:
            old_config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
            old_regions = json.loads(
                (artifact_dir / "region_metadata.json").read_text(encoding="utf-8")
            )
            old_manifest = json.loads(
                (artifact_dir / "input_manifest.json").read_text(encoding="utf-8")
            )
            if old_config != engine.config or old_manifest["cache_key"] != pair.cache_key:
                raise ValueError("Saved pair configuration/input changed; choose a new output root")
            if old_regions.get("config_fingerprint") != regions.metadata.get("config_fingerprint"):
                raise ValueError("Region configuration changed; choose a new output root")
            with np.load(artifact_dir / "regions.npz", allow_pickle=False) as saved:
                if (
                    not np.array_equal(saved["labels"], regions.labels)
                    or not np.array_equal(saved["soft_masks"], regions.soft_masks.cpu().numpy())
                    or not np.array_equal(saved["edges"], np.asarray(regions.edges).reshape(-1, 3))
                ):
                    raise ValueError(
                        "Region implementation/configuration changed; choose a new output root"
                    )
        baseline_path = artifact_dir / "default.npz"
        if not baseline_path.exists():
            np.savez_compressed(baseline_path, image_rgb=initial.image_rgb.detach().cpu().numpy())

        def generate(alphas):
            if method not in CMA_METHODS and any(
                float(a) not in engine.config["allowed_region_alphas"] for a in alphas
            ):
                raise ValueError("Candidate outside the formal domain")
            return engine.render_regions(pair, prepared, tuple(alphas))

        for method in methods:
            for seed in seeds:
                search_config = {
                    "method": method,
                    "seed": seed,
                    "budget": budget,
                    "levels": exp["levels"],
                    "max_restarts": exp["max_restarts"],
                    "tolerance": exp.get("tolerance", 1e-9),
                    "scoring": dict(exp["scoring"]),
                    "global_scan_grid": exp["global_scan_grid"],
                    **{k: exp[k] for k in ("landscape", "sigma", "calibration") if k in exp},
                }
                identity = {
                    "search": search_config,
                    "pair_cache_key": pair.cache_key,
                    "regions_sha256": file_sha256(artifact_dir / "regions.npz"),
                    "baseline_sha256": file_sha256(baseline_path),
                    "version": "integrated-v2",
                    "source_fingerprint": source_fingerprint,
                    "evaluator_fingerprint": evaluator_fingerprint,
                    "region_config_fingerprint": regions.metadata.get("config_fingerprint"),
                }
                config_hash = digest(identity)
                run_id = f"{args.split}/{pair_id}/{method}/seed{seed}"
                if run_id in records:
                    if not args.resume:
                        raise FileExistsError(
                            f"Run exists: {run_id}; use --resume or a new output root"
                        )
                    old = records[run_id]
                    if old["config_hash"] != config_hash:
                        raise ValueError("Resumed experiment configuration changed")
                    from .artifacts import read_result

                    read_result(old)
                    print(f"[RESUME] {run_id}", flush=True)
                    continue
                started = time.perf_counter()
                extra = {}
                if "reconstruct" in inspect.signature(solve).parameters:
                    extra["reconstruct"] = lambda: engine.render_global(pair, 0.0)
                result = solve(
                    generate=generate,
                    pair=pair,
                    regions=regions,
                    initial_result=initial,
                    search_config=search_config,
                    **extra,
                )
                validate_search_result(
                    result,
                    pair,
                    prepared,
                    None if method in CMA_METHODS else engine.config["allowed_region_alphas"],
                )
                elapsed = time.perf_counter() - started
                prefix = pair_dir / f"{method}__seed{seed}"
                trajectory = prefix.with_suffix(".jsonl")
                trajectory.write_text(
                    "".join(
                        json.dumps(record, allow_nan=False) + "\n" for record in result.trajectory
                    ),
                    encoding="utf-8",
                )
                selected = prefix.with_suffix(".npz")
                np.savez_compressed(
                    selected, image_rgb=result.best_image_rgb.detach().cpu().numpy()
                )
                prefix.with_suffix(".png").write_bytes(png_bytes(result.best_image_rgb))
                metadata = prefix.with_suffix(".json")
                write_json(
                    metadata,
                    {
                        "format_version": "integrated-v2",
                        "run_id": run_id,
                        "region_alphas": result.region_alphas,
                        "metrics": result.metrics,
                        "status": result.status,
                        "search_config": search_config,
                        "source_fingerprint": source_fingerprint,
                        "evaluator_fingerprint": evaluator_fingerprint,
                        "metadata": result.metadata,
                        "selected_sha256": file_sha256(selected),
                        "regions_sha256": identity["regions_sha256"],
                        "baseline_sha256": identity["baseline_sha256"],
                    },
                )
                row = {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "method": method,
                    "seed": seed,
                    "config_hash": config_hash,
                    "status": result.status,
                    "used_default": result.metadata["used_default"],
                    "stop_reason": result.metadata["stop_reason"],
                    **result.metrics,
                    **{k: result.metrics.get(k) for k in ("A", "U", "U_L", "U_ab")},
                    "beta": result.metadata.get("beta", 0.0),
                    "q0": result.metadata["q0"],
                    "t0": result.metadata["t0"],
                    "candidate_count": result.candidate_count,
                    "decoder_calls": result.decoder_calls,
                    "state_cache_hits": result.state_cache_hits,
                    "restarts_used": result.metadata["restarts_used"],
                    "elapsed_s": round(elapsed, 3),
                    "trajectory": _relative(trajectory),
                    "artifact_dir": _relative(artifact_dir),
                    "region_artifact": _relative(artifact_dir / "regions.npz"),
                    "selected_tensor": _relative(selected),
                    "result_metadata": _relative(metadata),
                }
                records[run_id] = row
                _persist(out_root, records, skipped, exp)
                print(
                    f"[{run_id}] {result.status} J={result.metrics['J']:.6f} "
                    f"candidates={result.candidate_count} {elapsed:.2f}s",
                    flush=True,
                )
    _persist(out_root, records, skipped, exp)
    print(f"Saved {len(records)} runs, {len(skipped)} missing-image pairs: {out_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-config", default="resources/configs/default.json")
    parser.add_argument("--config", default="resources/configs/experiment.json")
    parser.add_argument("--split", choices=["pilot", "evaluation"], default="pilot")
    parser.add_argument("--out-root", default="outputs/experiments/integrated-v2")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["risk", "plain", "random_states", "global_scan", *CMA_METHODS],
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--content")
    parser.add_argument("--style")
    parser.add_argument("--pair-id")
    parser.add_argument("--resume", action="store_true")
    run_matrix(parser.parse_args())


if __name__ == "__main__":
    main()
