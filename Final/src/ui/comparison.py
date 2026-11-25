"""Aligned views of the unmodified images, with the full PNG available in every view."""

import numpy as np
import streamlit as st
from PIL import ImageChops

from src.models.preprocessing import png_bytes, tensor_to_pil

POSITIONS = {
    "左上": (0, 0),
    "中上": (0.5, 0),
    "右上": (1, 0),
    "左中": (0, 0.5),
    "中央": (0.5, 0.5),
    "右中": (1, 0.5),
    "左下": (0, 1),
    "中下": (0.5, 1),
    "右下": (1, 1),
}


def crop_box(left, right, zoom, position):
    width, height = left.size
    w, h = width // zoom, height // zoom
    if position == "变化最大处":
        # Score each possible crop by the sum of absolute RGB differences.
        delta = np.asarray(ImageChops.difference(left, right)).sum(axis=2, dtype=np.int64)
        summed = np.pad(delta.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        totals = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
        y, x = np.unravel_index(totals.argmax(), totals.shape)
    else:
        fx, fy = POSITIONS[position]
        x, y = round((width - w) * fx), round((height - h) * fy)
    return int(x), int(y), int(x + w), int(y + h)


def input_preview(content, style, content_label, style_label):
    with st.container(key="reference_view"):
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("#### 内容图")
            st.image(content, width="stretch", output_format="PNG")
            st.caption(f"{content_label} · 中心裁剪")
        with right:
            st.markdown("#### 风格参考")
            st.image(style, width="stretch", output_format="PNG")
            st.caption(f"{style_label} · 保留原比例")


def result_view(default, best, reference_alpha):
    baseline = tensor_to_pil(default.image_rgb)
    searched = tensor_to_pil(best.image_rgb) if best is not None else None
    width, height = baseline.size
    box = None
    if searched is not None:
        same_png = ImageChops.difference(baseline, searched).getbbox() is None
        with st.container(key="comparison_tools", horizontal=True, vertical_alignment="center"):
            with st.popover("对照范围", icon=":material/zoom_in:"):
                zoom = st.radio(
                    "放大倍率",
                    [1, 2, 4],
                    format_func=lambda n: "全图" if n == 1 else f"局部 {n}×",
                    horizontal=True,
                    key="compare_zoom",
                )
                if zoom != 1:
                    positions = list(POSITIONS) if same_png else ["变化最大处", *POSITIONS]
                    pos = st.selectbox("两图同步查看位置", positions, key="compare_position")
                    box = crop_box(baseline, searched, zoom, pos)
            st.caption(f"局部 {zoom}× · 起点 ({box[0]}, {box[1]})" if box else "全图")
    left, right = st.container(key="result_pair").columns(2, gap="medium")
    for column, label, detail, picture, button, filename in (
        (
            left,
            "AdaIN 直出",
            f"全图统一强度 α = {reference_alpha:g}",
            baseline,
            "下载 AdaIN PNG",
            "adain",
        ),
        (right, "区域搜索结果", "按区域选择风格强度", searched, "下载搜索结果 PNG", "search"),
    ):
        with column:
            st.markdown(f"#### {label}", help=detail)
            if picture is None:
                st.html('<div class="pending-result">点击「生成双图对照」查看搜索结果</div>')
                continue
            st.image(picture.crop(box) if box else picture, width="stretch", output_format="PNG")
            st.download_button(
                button,
                data=png_bytes(picture),
                file_name=f"{filename}_{width}x{height}.png",
                mime="image/png",
                on_click="ignore",
            )


def score_table(result):
    rows = []
    for key, name in (
        ("A", "美学预测 ↑"),
        ("U", "新增不均匀度 ↓"),
        ("Q", "结构误差"),
        ("T", "风格误差"),
        ("J", "搜索目标"),
    ):
        if result.metrics.get(key) is None:
            continue
        baseline, searched = result.trajectory[0][key], result.metrics[key]
        rows.append(
            {
                "指标": f"{key} · {name}",
                "AdaIN 直出": f"{baseline:.6g}",
                "区域搜索": f"{searched:.6g}",
                "差值（搜索 − 直出）": f"{searched - baseline:+.4g}",
            }
        )
    st.table(rows, hide_index=True)
    limit = 1 + result.metadata.get("scorer", {}).get("epsilon", 0.05)
    st.caption(f"A 越高越好，U、Q、J 越低越好。T 不超过直出的 {limit:g} 倍。")
