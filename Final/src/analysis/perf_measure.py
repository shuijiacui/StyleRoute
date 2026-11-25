"""性能与资源测量（README section 7 运行资源口径）。

口径（与默认生成和搜索配置使用相同的输入尺寸）：
- 冷启动：import torch -> 引擎构造（载入生成器权重）的墙钟时间；
- 稳态：预热 1 次完整搜索后，正式测量 5 次重复的 单次解码 / 单次评分 /
  内容-风格特征准备，报 mean±sd；
- 完整流程：同一图对上 5 次完整风险引导搜索（seed 固定为 0，
  性能重复不使用质量种子）；
- 内存：进程当前 RSS 快照（不代表峰值）在
  准备后、搜索后各采样一次。

输出：M6/outputs/evaluation/perf_report.md（表格 + 机器指纹）

用法（仓库根目录）：
    python -m src.analysis.perf_measure
"""

from __future__ import annotations

import json
import statistics
import time

from src.models.adain import AdaINEngine  # noqa: E402
from src.project import ROOT as _REPO_ROOT
from src.project import load_config  # noqa: E402
from src.regions.adapter import build_regions
from src.search import objective  # noqa: E402

_IMAGES_ROOT = _REPO_ROOT / "resources" / "images"
_REPEATS = 5


def _rss_mb() -> float:
    """Current resident memory snapshot in MiB, not a sampled process peak."""
    import psutil

    return psutil.Process().memory_info().rss / 1024 / 1024


def _timed(fn, n: int) -> tuple[float, float]:
    vals = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def main() -> None:
    manifest = json.loads((_IMAGES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    first = manifest["evaluation_pairs"][0]
    paths = {
        e["id"]: _REPO_ROOT / e["local_path"].replace("/", "\\") for e in manifest["content_images"]
    }
    spaths = {
        e["id"]: _REPO_ROOT / e["local_path"].replace("/", "\\") for e in manifest["style_images"]
    }
    content = str(paths[first["content_id"]])
    style = str(spaths[first["style_id"]])

    # 冷启动：import 完成之后的引擎构造
    t0 = time.perf_counter()
    engine = AdaINEngine(load_config())
    cold_s = time.perf_counter() - t0

    rss_after_init = _rss_mb()

    # 稳态准备（该图对 prepare 一次，后续重复测量不再重新 prepare）
    t0 = time.perf_counter()
    pair = engine.prepare(content, style, pair_id=first["pair_id"])
    regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=str(engine.device))
    prepared = engine.prepare_regions(pair, regions)
    default = engine.render_global(pair, engine.config["default_alpha"])
    prep_s = time.perf_counter() - t0
    rss_after_prep = _rss_mb()

    from src.models.preprocessing import image_tensor

    scorer_cfg = objective.ScorerConfig()
    evaluator = objective.VGG19Evaluator(
        content_tensor=pair.content_tensor,
        style_tensor=image_tensor(pair.style_rgb).to(pair.content_tensor.device),
        soft_masks=regions.soft_masks,
        edges=regions.edges,
        config=scorer_cfg,
        device=str(engine.device),
    )
    scorer = objective.PairScorer(evaluator)
    scorer.calibrate(default.image_rgb)

    alphas_probe = (0.3,) + (0.6,) * (len(regions.region_ids) - 1)

    # 预热：1 次完整搜索
    from src.integration import load_adapter

    solve = load_adapter("src.search.adapter:solve")
    search_config = {
        "method": "risk",
        "budget": 40,
        "max_restarts": 2,
        "seed": 0,
        "levels": [0.3, 0.6, 0.9],
        "lam": 1.0,
        "epsilon": 0.05,
        "delta": 0.001,
        "delta_t": 1e-6,
        "weights": "IMAGENET1K_V1",
    }

    def generate(a):
        return engine.render_regions(pair, prepared, tuple(float(x) for x in a))

    solve(
        generate=generate,
        pair=pair,
        regions=regions,
        initial_result=default,
        search_config=dict(search_config),
    )

    decode_m, decode_sd = _timed(
        lambda: engine.render_regions(pair, prepared, alphas_probe), _REPEATS
    )
    probe_image = generate(alphas_probe).image_rgb
    score_m, score_sd = _timed(lambda: scorer(probe_image, alphas_probe), _REPEATS)
    full_m, full_sd = _timed(
        lambda: solve(
            generate=generate,
            pair=pair,
            regions=regions,
            initial_result=default,
            search_config=dict(search_config),
        ),
        _REPEATS,
    )
    rss_after_search = _rss_mb()

    import platform

    import torch

    lines = [
        "# 性能与资源测量（README §7 口径）",
        "",
        f"- 机器：{platform.processor() or 'unknown CPU'} / {platform.machine()}，"
        f"{round(_rss_mb())} MB 基线工作集；Python {platform.python_version()}，"
        f"torch {torch.__version__}（CPU）",
        f"- 图对：{first['pair_id']}，{pair.input_hw[1]}×{pair.input_hw[0]} RGB 输入",
        f"- 重复：预热 1 次完整搜索后正式 {_REPEATS} 次，报 mean±sd（秒）",
        "",
        "| 项目 | 均值 | SD |",
        "|---|---|---|",
        f"| 冷启动（引擎构造，含权重载入） | {cold_s:.2f} | — |",
        f"| 图对准备（编码内容/风格 + 区域准备 + 默认渲染） | {prep_s:.2f} | — |",
        f"| 单次解码（render_regions） | {decode_m:.3f} | {decode_sd:.3f} |",
        f"| 单次评分（VGG19 Q/T/S/J） | {score_m:.3f} | {score_sd:.3f} |",
        f"| 完整风险搜索（预算 40，含解码与评分） | {full_m:.1f} | {full_sd:.1f} |",
        "",
        "| 内存采样（进程工作集） | MB |",
        "|---|---|",
        f"| 引擎构造后 | {rss_after_init:.0f} |",
        f"| 图对准备后 | {rss_after_prep:.0f} |",
        f"| 完整搜索后（含预热） | {rss_after_search:.0f} |",
        "",
        "注：质量实验的 5 个种子不作性能重复；上表 5 次为同配置性能重复。",
    ]
    out = _REPO_ROOT / "outputs" / "experiments" / "integrated-v2" / "evaluation" / "perf_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
