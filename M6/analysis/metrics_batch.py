"""Batch contour evaluation (README section 6, metrics axis).

Re-renders each run's final returned image from its trajectory
(the last feasible `is_new_best` record; or the default image when
`used_default=True`) and scores it against the content image with the
frozen contour pipeline in M6.search.metrics.

Outputs (per split):
    M6/outputs/<split>/contour_metrics.csv   one row per run
    M6/outputs/<split>/contour_summary.md    pair-level aggregates by method

Incremental: runs already present in the CSV are skipped, so the batch
can be interrupted and resumed. Images are NOT stored; only metric rows.

Usage (from the repo root):
    python -m M6.analysis.metrics_batch --split evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.adain import AdaINEngine  # noqa: E402
from src.project import load_config  # noqa: E402

from M6.regions_grid import build_regions  # noqa: E402
from M6.search.metrics import MetricsConfig, contour_f1, provenance  # noqa: E402

_IMAGES_ROOT = _REPO_ROOT / "M4" / "resources" / "images"

_FIELDS = [
    "run_id", "pair_id", "method", "seed", "used_default",
    "precision", "recall", "f1", "pred_px", "target_px", "both_no_edge",
    "final_alphas",
]


def _final_alphas(traj_path: Path, used_default: bool, default_alpha: float) -> list[float] | None:
    """Final returned state: last feasible is_new_best record, else default."""
    if used_default:
        return None  # caller renders the default image
    last_best = None
    with traj_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("is_new_best") and rec.get("feasible"):
                last_best = rec["alphas"]
    return last_best if last_best is not None else None


def run_batch(args) -> None:
    out_root = Path(args.out_root) / args.split
    csv_path = out_root / "contour_metrics.csv"
    done: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done[row["run_id"]] = row

    with (out_root / "runs.csv").open(encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))
    pending = [r for r in runs if r["run_id"] not in done]
    print(f"[METRICS] {len(done)} cached, {len(pending)} to evaluate", flush=True)

    manifest = json.loads((_IMAGES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    contents = {
        e["id"]: str((_IMAGES_ROOT.parent.parent / e["local_path"].replace("/", "\\")).resolve())
        for e in manifest.get("content_images", []) if e.get("local_path")
    }
    styles = {
        e["id"]: str((_IMAGES_ROOT.parent.parent / e["local_path"].replace("/", "\\")).resolve())
        for e in manifest.get("style_images", []) if e.get("local_path")
    }

    engine = AdaINEngine(load_config(_REPO_ROOT / "M4" / "resources" / "configs" / "team.json"))
    default_alpha = float(engine.config["default_alpha"])

    # group by pair so the engine prepares each pair once
    by_pair: dict[str, list[dict]] = {}
    for r in pending:
        by_pair.setdefault(r["pair_id"], []).append(r)

    cfg = MetricsConfig()
    rows: list[dict] = []
    n_done = 0
    for pair_id, group in by_pair.items():
        content_path = style_path = None
        for pair_info in manifest.get(f"{args.split}_pairs", []):
            if pair_info["pair_id"] == pair_id:
                content_path = contents.get(pair_info["content_id"])
                style_path = styles.get(pair_info["style_id"])
                break
        if not content_path or not Path(content_path).exists() or not Path(style_path or "").exists():
            print(f"[SKIP] {pair_id}: missing image file(s)", flush=True)
            continue
        pair = engine.prepare(str(content_path), str(style_path), pair_id=pair_id)
        regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=str(engine.device))
        prepared = engine.prepare_regions(pair, regions)
        target_edges = None  # computed lazily per pair

        for run in group:
            method, seed = run["method"], run["seed"]
            traj_path = Path(_REPO_ROOT) / run["trajectory"].replace("\\", "/")
            alphas = _final_alphas(traj_path, run["used_default"] == "True", default_alpha)
            t0 = time.perf_counter()
            if alphas is None:
                result = engine.render_global(pair, default_alpha)
            else:
                result = engine.render_regions(pair, prepared, tuple(float(a) for a in alphas))
            score = contour_f1(result.image_rgb, pair.content_tensor, cfg)
            elapsed = time.perf_counter() - t0
            rows.append({
                "run_id": run["run_id"],
                "pair_id": pair_id,
                "method": method,
                "seed": seed,
                "used_default": run["used_default"],
                "precision": round(score["precision"], 6),
                "recall": round(score["recall"], 6),
                "f1": round(score["f1"], 6),
                "pred_px": score["pred_px"],
                "target_px": score["target_px"],
                "both_no_edge": score["both_no_edge"],
                "final_alphas": json.dumps(alphas) if alphas is not None else "default",
            })
            n_done += 1
            if n_done % 25 == 0:
                print(f"[{n_done}/{len(pending)}] last: {pair_id} {method}/s{seed} "
                      f"F1={score['f1']:.4f} ({elapsed:.2f}s)", flush=True)

    if rows:
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    _summarize(out_root, csv_path, cfg)
    print("[METRICS] done", flush=True)


def _summarize(out_root: Path, csv_path: Path, cfg: MetricsConfig) -> None:
    """Pair-level aggregates (seed mean per pair, then stats over pairs)."""
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pair_mean: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        pair_mean.setdefault((r["method"], r["pair_id"]), []).append(float(r["f1"]))
    by_method: dict[str, list[float]] = {}
    for (method, _pair), vals in pair_mean.items():
        by_method.setdefault(method, []).append(sum(vals) / len(vals))

    lines = [
        "# 轮廓 P/R/F1 汇总（种子均值 -> 图对级统计）",
        "",
        f"- provenance: `{provenance(cfg)}`",
        f"- runs: {len(rows)}（图对级单位 {sum(len(v) for v in by_method.values())}）",
        "",
        "| 方法 | 图对数 | F1 均值 | F1 中位数 | F1 最小值 |",
        "|---|---|---|---|---|",
    ]
    for method, vals in sorted(by_method.items()):
        vals = sorted(vals)
        mid = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        lines.append(f"| {method} | {len(vals)} | {sum(vals)/len(vals):.4f} | {mid:.4f} | {vals[0]:.4f} |")
    (out_root / "contour_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[METRICS] summary -> {out_root / 'contour_summary.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["pilot", "evaluation"])
    parser.add_argument("--out-root", default=str(_REPO_ROOT / "M6" / "outputs"))
    run_batch(parser.parse_args())


if __name__ == "__main__":
    main()
