"""失败案例归因分析（README section 7 稳定性与失败 / 7.2 未改善案例）。

失败图对定义：风险引导与普通局部两种方法在 5 种子均值上都未把 J 推过
默认参考 J0 = Q0/(Q0+δ)（S0=0）。

对每个失败图对提取（lam=1.0 evaluation 矩阵的原始轨迹，不重新搜索）：
- T 约束绑定程度：非缓存评价中不可行状态占比、平均超限量
- S 惩罚阻挡：可行邻居里 Q 已低于 Q0 但 J 未低于 J0 的状态数
- 地形平坦度：可行邻居 Q/Q0 的极差
- 内容结构集中度：默认状态下区域结构贡献 q_i 的 top-1 / top-4 占比
- 不变量核查：失败 run 中 accepted 移动数应为 0（接受即产生优于默认的档案）

分类规则（冻结，按顺序取第一个命中）：
1. t_binding   不可行占比 >= 25%
2. s_blocking  存在 Q<Q0 但 J>=J0 的可行邻居，且不可行占比 < 25%
3. flat        可行邻居 Q/Q0 极差 < 0.005
4. missed      其余（地形有改善空间但搜索未找到）

输出：
    M6/outputs/evaluation/failure_analysis.csv
    M6/outputs/evaluation/failure_analysis.md

用法（仓库根目录）：
    python -m M6.analysis.failure_analysis
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.adain import AdaINEngine  # noqa: E402
from src.project import load_config  # noqa: E402

from M6.regions_grid import build_regions  # noqa: E402
from M6.search.objective import ScorerConfig, VGG19Evaluator, apportion  # noqa: E402

_DELTA = 0.001
_OUT = _REPO_ROOT / "M6" / "outputs" / "evaluation"


def _load_runs() -> list[dict]:
    with (_OUT / "runs.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _trajectory_stats(pair_dir: Path, method: str) -> dict:
    """Aggregate trajectory stats across 5 seeds for one method."""
    infeasible = feasible = 0
    violations: list[float] = []
    q_spread_samples: list[float] = []
    blocked = 0
    accepted = 0
    j0 = None
    for seed in range(5):
        path = pair_dir / f"{method}__seed{seed}.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(l) for l in path.open(encoding="utf-8")]
        if not recs:
            continue
        j0 = recs[0]["J"]
        feasible_qs = []
        for r in recs[1:]:
            if r.get("cache_hit"):
                continue
            if r["accepted"]:
                accepted += 1
            if r["feasible"]:
                feasible += 1
                feasible_qs.append(r["Q"] / (recs[0]["Q"] + 1e-12))
                if r["Q"] < recs[0]["Q"] and r["J"] >= j0 - 1e-9:
                    blocked += 1
            else:
                infeasible += 1
                violations.append(r["violation"])
        if feasible_qs:
            q_spread_samples.append(max(feasible_qs) - min(feasible_qs))
    return {
        "j0": j0,
        "infeasible_frac": infeasible / max(1, infeasible + feasible),
        "mean_violation": statistics.mean(violations) if violations else 0.0,
        "blocked": blocked,
        "q_spread": statistics.mean(q_spread_samples) if q_spread_samples else 0.0,
        "accepted": accepted,
    }


def _style_tensor(pair):
    """Same canonical conversion as M6.search.adapter (kept in sync manually)."""
    from src.models.preprocessing import image_tensor

    return image_tensor(pair.style_rgb).to(pair.content_tensor.device)


def main() -> None:
    runs = _load_runs()
    # pair-level: both risk and plain failed to beat J0 on seed mean
    by_pair: dict[str, dict[str, list[float]]] = {}
    j0s: dict[str, float] = {}
    for r in runs:
        if r["method"] not in ("risk", "plain"):
            continue
        q0 = float(r["q0"])
        j0 = q0 / (q0 + _DELTA)
        j0s[r["pair_id"]] = j0
        by_pair.setdefault(r["pair_id"], {}).setdefault(r["method"], []).append(float(r["J"]))
    failing = sorted(
        p for p, m in by_pair.items()
        if len(m.get("risk", [])) == 5 and len(m.get("plain", [])) == 5
        and statistics.mean(m["risk"]) >= j0s[p] - 1e-6
        and statistics.mean(m["plain"]) >= j0s[p] - 1e-6
    )
    print(f"[FAIL] {len(failing)} failing pairs / {len(by_pair)}", flush=True)

    # content-structure concentration per failing pair (engine reused)
    manifest = json.loads((_REPO_ROOT / "M4/resources/images/manifest.json").read_text(encoding="utf-8"))
    paths = {e["id"]: _REPO_ROOT / "M4" / e["local_path"].replace("/", "\\") for e in manifest["content_images"]}
    spaths = {e["id"]: _REPO_ROOT / "M4" / e["local_path"].replace("/", "\\") for e in manifest["style_images"]}
    pair_meta = {p["pair_id"]: p for p in manifest["evaluation_pairs"]}
    engine = AdaINEngine(load_config(_REPO_ROOT / "M4/resources/configs/team.json"))
    cfg = ScorerConfig()

    rows = []
    accepted_violations = 0
    for pair_id in failing:
        meta = pair_meta[pair_id]
        stats = _trajectory_stats(_OUT / pair_id, "risk")
        stats_p = _trajectory_stats(_OUT / pair_id, "plain")
        accepted_total = stats["accepted"] + stats_p["accepted"]
        if accepted_total != 0:
            accepted_violations += 1

        pair = engine.prepare(str(paths[meta["content_id"]]), str(spaths[meta["style_id"]]), pair_id=pair_id)
        regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=str(engine.device))
        default = engine.render_global(pair, engine.config["default_alpha"])
        evaluator = VGG19Evaluator(
            content_tensor=pair.content_tensor,
            style_tensor=_style_tensor(pair),
            soft_masks=regions.soft_masks,
            edges=regions.edges,
            config=cfg,
            device=str(engine.device),
        )
        _, q_map = evaluator.structural_loss(default.image_rgb)
        q_i = apportion(q_map, evaluator._masks)
        total = sum(q_i) or 1.0
        ordered = sorted(q_i, reverse=True)
        top1 = ordered[0] / total
        top4 = sum(ordered[:4]) / total

        infeasible_frac = (stats["infeasible_frac"] + stats_p["infeasible_frac"]) / 2
        blocked = stats["blocked"] + stats_p["blocked"]
        q_spread = (stats["q_spread"] + stats_p["q_spread"]) / 2
        if infeasible_frac >= 0.25:
            cause = "t_binding"
        elif blocked > 0:
            cause = "s_blocking"
        elif q_spread < 0.005:
            cause = "flat"
        else:
            cause = "missed"
        rows.append({
            "pair_id": pair_id, "class": cause,
            "infeasible_frac": round(infeasible_frac, 4),
            "mean_violation": round(stats["mean_violation"], 4),
            "blocked": blocked, "q_spread": round(q_spread, 5),
            "top1_q_share": round(top1, 4), "top4_q_share": round(top4, 4),
            "accepted_moves": accepted_total,
            "content_class": meta["content_id"].rsplit("_", 1)[0],
        })
        print(f"[{pair_id}] {cause} inf={infeasible_frac:.2f} blocked={blocked} "
              f"spread={q_spread:.4f} top4q={top4:.2f}", flush=True)

    with (_OUT / "failure_analysis.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, list[str]] = {}
    for r in rows:
        counts.setdefault(r["class"], []).append(r["pair_id"])
    lines = [
        "# 失败图对归因（evaluation，lam=1.0）",
        "",
        f"- 失败定义：风险与普通两种方法 5 种子均值均未超过 J0 = Q0/(Q0+δ)。",
        f"- 失败图对：{len(rows)}/{len(by_pair)}。",
        f"- accepted 移动核查：{len(rows) - accepted_violations}/{len(rows)} 图对的"
        f" accepted 移动数为 0。注意重启阶段的接受是相对重启起点而非默认状态，"
        f"因此失败 run 出现 accepted>0 属正常（该移动未超过默认档案）。",
        "",
        "| 类别 | 规则 | 图对数 |",
        "|---|---|---|",
        f"| t_binding（T 约束顶死） | 不可行评价占比 ≥ 25% | {len(counts.get('t_binding', []))} |",
        f"| s_blocking（S 惩罚阻挡） | 存在 Q<Q0 但 J≥J0 的可行邻居 | {len(counts.get('s_blocking', []))} |",
        f"| flat（地形平坦） | 可行邻居 Q/Q0 极差 < 0.005 | {len(counts.get('flat', []))} |",
        f"| missed（其余） | 有改善空间但搜索未命中 | {len(counts.get('missed', []))} |",
        "",
        "## 明细",
        "",
        "| pair_id | 类别 | 不可行占比 | 平均超限量 | blocked | Q 极差 | top1 q | top4 q |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pair_id']} | {r['class']} | {r['infeasible_frac']} | "
            f"{r['mean_violation']} | {r['blocked']} | {r['q_spread']} | "
            f"{r['top1_q_share']} | {r['top4_q_share']} |"
        )
    (_OUT / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FAIL] -> {_OUT / 'failure_analysis.md'}", flush=True)


if __name__ == "__main__":
    main()
