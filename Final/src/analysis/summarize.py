"""Analysis / summarization (README section 8): regenerate all figures and
tables from the raw JSONL trajectories written by src.experiments.run.

Nothing here is hand-copied: every number and curve comes from the per-run
JSONL files. Re-running this script after new experiments updates everything.

Outputs (under M6/analysis/figures/<split>/):
    qt_scatter.png       Q/Q0 vs T/T0 over all evaluated states, by method,
                         feasible vs infeasible markers
    search_curves.png    best feasible J / J0 vs evaluation index per run,
                         mean curve per method; checkpoints 10/20/40
    timing_table.csv     per-method cost summary (decoder calls, wall time)
    summary.md           checkpoint table + method comparison text

Usage (from the repo root):
    python -m src.analysis.summarize --split pilot
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

from src.project import ROOT as _REPO_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKPOINTS = (10, 20, 40)


def _load_runs(split_dir: Path) -> list[dict]:
    runs = []
    with (split_dir / "runs.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            runs.append(row)
    return runs


def _load_trajectory(row: dict) -> list[dict]:
    path = _REPO_ROOT / row["trajectory"].replace("\\", "/")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _archive_curve(records: list[dict]) -> tuple[list[int], list[float]]:
    """Best feasible J so far (relative to the default state = record #1)."""
    j0 = records[0]["J"]
    if j0 <= 0:
        return [], []  # Relative improvement is undefined for a zero baseline.
    xs, ys, best = [], [], None
    for record in records:
        if record["cache_hit"]:
            continue
        if record["feasible"] and (best is None or record["J"] < best):
            best = record["J"]
        if best is not None:
            xs.append(record["evaluation_index"])
            ys.append(best / j0)
    return xs, ys


def make_qt_scatter(split_dir: Path, out_dir: Path) -> None:
    method_points = defaultdict(lambda: {"Q": [], "T": [], "feasible": []})
    for row in _load_runs(split_dir):
        records = _load_trajectory(row)
        q0, t0 = float(row["q0"]), float(row["t0"])
        if q0 <= 0 or t0 <= 0:
            continue
        for record in records:
            if record["cache_hit"]:
                continue
            bucket = method_points[row["method"]]
            bucket["Q"].append(record["Q"] / q0)
            bucket["T"].append(record["T"] / t0)
            bucket["feasible"].append(bool(record["feasible"]))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {
        "risk": "#d62728",
        "plain": "#1f77b4",
        "random_states": "#2ca02c",
        "global_scan": "#9467bd",
    }
    for method, bucket in method_points.items():
        color = colors.get(method, "#7f7f7f")
        feas = [f for f in bucket["feasible"]]
        q_feas = [q for q, f in zip(bucket["Q"], feas) if f]
        t_feas = [t for t, f in zip(bucket["T"], feas) if f]
        q_inf = [q for q, f in zip(bucket["Q"], feas) if not f]
        t_inf = [t for t, f in zip(bucket["T"], feas) if not f]
        ax.scatter(
            t_feas, q_feas, s=14, color=color, alpha=0.55, marker="o", label=f"{method} (可行)"
        )
        if q_inf:
            ax.scatter(
                t_inf, q_inf, s=22, color=color, alpha=0.9, marker="x", label=f"{method} (违反)"
            )
    ax.axvline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("T / T0（风格统计，越右越超风格参考）")
    ax.set_ylabel("Q / Q0（结构代理，越小越保内容）")
    ax.set_title(f"{split_dir.name} split：全部已评价状态")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "qt_scatter.png", dpi=160)
    plt.close(fig)


def make_search_curves(split_dir: Path, out_dir: Path) -> dict:
    curves = defaultdict(list)  # method -> list of (xs, ys)
    checkpoints = defaultdict(lambda: defaultdict(list))  # method -> k -> values

    for row in _load_runs(split_dir):
        xs, ys = _archive_curve(_load_trajectory(row))
        if not xs:
            continue
        curves[row["method"]].append((xs, ys))
        for k in CHECKPOINTS:
            eligible = [(x, y) for x, y in zip(xs, ys) if x <= k]
            if eligible:
                checkpoints[row["method"]][k].append(eligible[-1][1])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "risk": "#d62728",
        "plain": "#1f77b4",
        "random_states": "#2ca02c",
        "global_scan": "#9467bd",
    }
    for method, runs in curves.items():
        color = colors.get(method, "#7f7f7f")
        for xs, ys in runs:
            ax.plot(xs, ys, color=color, alpha=0.25, lw=1)
        # mean curve on the union grid
        grid = sorted({x for xs, _ in runs for x in xs})
        means = []
        for g in grid:
            vals = [ys[bisect_right(xs, g) - 1] for xs, ys in runs if xs[0] <= g <= xs[-1]]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        ax.plot(grid, means, color=color, lw=2.2, label=f"{method}（均值, n={len(runs)}）")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("真实评价次数（默认状态计 1）")
    ax.set_ylabel("可行档案最优 J / J0（<1 为改善）")
    ax.set_title(f"{split_dir.name} split：搜索曲线（细线=单次运行）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "search_curves.png", dpi=160)
    plt.close(fig)

    table = {}
    for method, ks in checkpoints.items():
        table[method] = {k: (sum(v) / len(v), len(v)) for k, v in ks.items()}
    return table


def bisect_right(xs, g):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= g:
            lo = mid + 1
        else:
            hi = mid
    return lo


def make_timing_table(split_dir: Path, out_dir: Path) -> list[dict]:
    agg = defaultdict(list)
    for row in _load_runs(split_dir):
        agg[row["method"]].append(row)
    table = []
    for method, rows in agg.items():
        ratios = [
            float(r["J"]) / _load_trajectory(r)[0]["J"]
            for r in rows
            if _load_trajectory(r)[0]["J"] > 0
        ]
        elapsed = [float(r["elapsed_s"]) for r in rows if str(r.get("elapsed_s", "")).strip()]
        table.append(
            {
                "method": method,
                "runs": len(rows),
                "no_improvement_rate": sum(r["status"] == "no_improvement" for r in rows)
                / len(rows),
                "mean_decoder_calls": sum(int(r["decoder_calls"]) for r in rows) / len(rows),
                "mean_elapsed_s": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
                "mean_final_J_over_J0": sum(ratios) / len(ratios) if ratios else None,
            }
        )
    if not table:
        return []
    with (out_dir / "timing_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    return table


def write_summary(
    split_dir: Path, out_dir: Path, checkpoint_table: dict, timing: list[dict]
) -> None:
    lines = [f"# {split_dir.name} split 汇总（由原始 JSONL 重新生成）\n"]
    lines.append("## 搜索曲线检查点（可行档案最优 J/J0，越小越好；括号内为运行数）\n")
    header = "| 方法 | " + " | ".join(f"@{k}" for k in CHECKPOINTS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CHECKPOINTS) + 1))
    for method, ks in checkpoint_table.items():
        cells = []
        for k in CHECKPOINTS:
            if k in ks:
                mean, n = ks[k]
                cells.append(f"{mean:.5f} (n={n})")
            else:
                cells.append("—")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("\n## 成本对比\n")
    lines.append("| 方法 | 运行数 | 无改善比例 | 平均解码次数 | 平均耗时(s) | 平均最终 J/J0 |")
    lines.append("|---|---|---|---|---|---|")
    for row in timing:
        ratio = row["mean_final_J_over_J0"]
        ratio_text = f"{ratio:.5f}" if ratio is not None else "未定义"
        lines.append(
            f"| {row['method']} | {row['runs']} | {row['no_improvement_rate']:.0%} "
            f"| {row['mean_decoder_calls']:.1f} | {row['mean_elapsed_s']:.1f} "
            f"| {ratio_text} |"
        )
    lines.append(
        "\n说明：J0 来自默认 0.6 的实际轨迹；全局扫描最多评价三档，曲线按不同状态数绘制。"
        "检查点超过实际结束次数时，表内沿用最后已观测值，不表示新增评价；J0=0 时不定义比值。"
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="pilot")
    parser.add_argument(
        "--outputs-root", default=str(_REPO_ROOT / "outputs" / "experiments" / "integrated-v2")
    )
    parser.add_argument("--figures-dir", default=str(_REPO_ROOT / "outputs" / "analysis"))
    args = parser.parse_args()

    split_dir = Path(args.outputs_root) / args.split
    if not (split_dir / "runs.csv").exists():
        raise SystemExit(f"找不到 {split_dir / 'runs.csv'}，先跑 src.experiments.run")
    out_dir = Path(args.figures_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    make_qt_scatter(split_dir, out_dir)
    checkpoint_table = make_search_curves(split_dir, out_dir)
    timing = make_timing_table(split_dir, out_dir)
    write_summary(split_dir, out_dir, checkpoint_table, timing)
    print(f"figures + tables -> {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(" ", path.name)


if __name__ == "__main__":
    main()
