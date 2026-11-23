"""生成"内容 | 风格 | 默认 | 搜索最佳"四联对比图（展示素材）。

从 evaluation 矩阵的原始轨迹重建风险方法的最终状态并重新渲染；
无改善图对的最佳状态 = 默认图（如实渲染，不伪装）。

输出：M6/analysis/figures/evaluation/compare_<pair_id>.png

用法（仓库根目录）：
    python -m M6.analysis.make_comparison_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if Path(candidate).exists():
                _FONT = ImageFont.truetype(candidate, 16)
                break
        else:
            _FONT = ImageFont.load_default()
    return _FONT

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.adain import AdaINEngine  # noqa: E402
from src.project import load_config  # noqa: E402

from M6.regions_grid import build_regions  # noqa: E402

_IMAGES_ROOT = _REPO_ROOT / "M4" / "resources" / "images"
_OUT = _REPO_ROOT / "M6" / "outputs" / "evaluation"

# (pair_id, 说明) —— 选材：改善最深 + 一个典型改善 + 一个失败案例
PAIRS = [
    ("beach_04__met_436535", "risk J/J0=0.952（全矩阵最深改善）"),
    ("downtown_03__met_436535", "risk J/J0=0.969（典型改善）"),
    ("beach_03__met_437854", "无改善（S 惩罚阻挡案例，最佳=默认）"),
]

LABELS = ["内容", "风格", "默认 0.6", "搜索最佳"]


def _final_alphas(pair_dir: Path) -> list[float] | None:
    last_best = None
    for seed in range(5):
        path = pair_dir / f"risk__seed{seed}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("is_new_best") and rec.get("feasible"):
                    last_best = rec["alphas"]
    return last_best


def main() -> None:
    manifest = json.loads((_IMAGES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    paths = {e["id"]: _REPO_ROOT / "M4" / e["local_path"].replace("/", "\\") for e in manifest["content_images"]}
    spaths = {e["id"]: _REPO_ROOT / "M4" / e["local_path"].replace("/", "\\") for e in manifest["style_images"]}
    meta = {p["pair_id"]: p for p in manifest["evaluation_pairs"]}

    engine = AdaINEngine(load_config(_REPO_ROOT / "M4" / "resources" / "configs" / "team.json"))
    default_alpha = float(engine.config["default_alpha"])

    for pair_id, note in PAIRS:
        m = meta[pair_id]
        pair = engine.prepare(str(paths[m["content_id"]]), str(spaths[m["style_id"]]), pair_id=pair_id)
        regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=str(engine.device))
        prepared = engine.prepare_regions(pair, regions)
        default_img = engine.render_global(pair, default_alpha).image_rgb
        alphas = _final_alphas(_OUT / pair_id)
        best_img = (engine.render_regions(pair, prepared, tuple(float(a) for a in alphas)).image_rgb
                    if alphas is not None else default_img)

        def to_pil(t):
            import torch
            arr = (t[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(arr)

        panels = [pair.content_rgb, pair.style_rgb, to_pil(default_img), to_pil(best_img)]
        w, h = panels[0].size
        scale = 256 / max(w, h)
        tw, th = int(w * scale), int(h * scale)
        pad_top = 28
        canvas = Image.new("RGB", (tw * 4 + 30, th + pad_top + 8), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(zip(panels, LABELS)):
            p = panel.resize((tw, th))
            x = i * (tw + 10)
            canvas.paste(p, (x, pad_top))
            draw.text((x + 4, 6), label, fill="black", font=_font())
        out = _REPO_ROOT / "M6" / "analysis" / "figures" / "evaluation" / f"compare_{pair_id}.png"
        canvas.save(out)
        print(f"[OK] {out.name}  ({note})", flush=True)


if __name__ == "__main__":
    main()
