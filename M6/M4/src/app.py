"""Local landscape-letter studio. Run: python -m streamlit run app.py."""

import json
import time

import streamlit as st

from src.integration import integration_status, run_search
from src.models.adain import AdaINEngine
from src.models.preprocessing import png_bytes, prepare_content, prepare_style, tensor_to_pil
from src.outputs import alpha_image, dataset_pair_id, region_overlay
from src.project import digest, load_config, project_path
from src.ui.postcard import postcard_dialog
from src.ui.theme import apply_theme, empty_preview


@st.cache_resource(show_spinner="正在准备画笔与颜料……")
def get_engine(config_json: str, asset_signature: str):
    return AdaINEngine(json.loads(config_json))


def _clear_results():
    for name in (
        "default",
        "best",
        "regions",
        "search_result",
        "postcard",
        "postcard_key",
        "last_error",
        "generation_seconds",
        "saved_directory",
    ):
        st.session_state.pop(name, None)
    st.session_state.postcard_open = False


def reset_if_input_changed(key: str):
    if st.session_state.get("input_key") != key:
        _clear_results()
        st.session_state.pop("pair", None)
        st.session_state.input_key = key


def reset_if_strength_changed(alpha: float):
    """Alpha changes invalidate results while retaining the encoded pair."""
    if st.session_state.get("requested_alpha") != alpha:
        _clear_results()
        st.session_state.requested_alpha = alpha


def _generate(config, asset_stats, content, style, alpha, search=False):
    try:
        engine = get_engine(
            json.dumps(config, sort_keys=True),
            digest(
                {
                    "stats": asset_stats,
                    "manifest": project_path(config["weight_manifest"]).read_text(encoding="utf-8"),
                }
            ),
        )
        st.session_state.engine = engine
        start = time.perf_counter()
        with st.spinner("正在把风景绘成画……"):
            if "pair" not in st.session_state:
                st.session_state.pair = engine.prepare(
                    content, style, pair_id=dataset_pair_id(content, style)
                )
            if "default" not in st.session_state:
                st.session_state.default = engine.render_global(st.session_state.pair, alpha)
            if search:
                for name in ("best", "regions", "search_result", "postcard_key"):
                    st.session_state.pop(name, None)
                progress = st.empty()
                regions, best, result = run_search(
                    engine,
                    st.session_state.pair,
                    st.session_state.default,
                    progress_callback=lambda event: progress.write(event),
                )
                st.session_state.regions = regions
                st.session_state.best = best
                st.session_state.search_result = result
        st.session_state.generation_seconds = time.perf_counter() - start
        st.session_state.pop("last_error", None)
    except Exception as error:
        st.session_state.last_error = f"{type(error).__name__}: {error}"


def main():
    st.set_page_config(page_title="山海来信 · 艺术明信片", page_icon="✉️", layout="wide")
    apply_theme()
    manifest = json.loads(
        project_path("resources/images/manifest.json").read_text(encoding="utf-8")
    )
    controls, workspace = st.columns([1, 2.75], gap="large")
    with controls, st.container(key="letter_controls"):
        st.subheader("01 / 选一帧风景")
        mode = st.radio("内容图片来源", ["项目试运行图片", "上传自己的图片"], horizontal=True)
        if mode == "项目试运行图片":
            pilots = sorted(
                (item for item in manifest["content_images"] if item["usage"] == "pilot"),
                key=lambda item: item["class_name"] != "mountain",
            )
            choices = {item["id"]: item for item in pilots}
            selected = st.selectbox(
                "内容图", list(choices), format_func=lambda key: choices[key]["label_zh"]
            )
            content_bytes = project_path(choices[selected]["local_path"]).read_bytes()
        else:
            upload = st.file_uploader(
                "上传内容图", type=["jpg", "jpeg", "png", "webp"], key="content_upload"
            )
            content_bytes = upload.getvalue() if upload else None
        st.subheader("02 / 借一抹画意")
        style_mode = st.radio("风格图片来源", ["项目艺术参考", "上传风格图"], horizontal=True)
        if style_mode == "项目艺术参考":
            styles = {item["id"]: item for item in manifest["style_images"]}
            selected_style = st.selectbox(
                "风格参考", list(styles), format_func=lambda key: styles[key]["label_zh"]
            )
            style_bytes = project_path(styles[selected_style]["local_path"]).read_bytes()
            with st.container(key="style_thumb"):
                st.image(style_bytes, width=120)
            st.caption(f"[作品来源]({styles[selected_style]['object_url']}) · CC0")
        else:
            upload_style = st.file_uploader(
                "上传风格图", type=["jpg", "jpeg", "png", "webp"], key="style_upload"
            )
            style_bytes = upload_style.getvalue() if upload_style else None
        profile = st.selectbox(
            "生成模式",
            ["creative", "baseline"],
            format_func=lambda value: {
                "creative": "清晰创作 · 512×512",
                "baseline": "实验基线 · 256×256",
            }[value],
        )
        creative = profile == "creative"
        config = load_config(
            "resources/configs/creative.json" if creative else "resources/configs/default.json"
        )
        if creative:
            render_alpha = st.slider(
                "全局风格强度", 0.0, 1.0, 0.8, 0.05, help="越大画意越浓，也可能损失更多结构细节。"
            )
            st.caption("轻一些留住细节，浓一些更有画意。")
        else:
            render_alpha = config["default_alpha"]
            st.caption("固定 256×256 · 全局初始强度 0.6")

    asset_stats = {
        name: (
            project_path(config[name]).stat().st_size,
            project_path(config[name]).stat().st_mtime_ns,
        )
        if project_path(config[name]).exists()
        else None
        for name in ("encoder_checkpoint", "decoder_checkpoint")
    }
    key = digest(
        {
            "content": content_bytes.hex() if content_bytes else None,
            "style": style_bytes.hex() if style_bytes else None,
            "config": config,
            "asset_stats": asset_stats,
            "assets": project_path(config["weight_manifest"]).read_text(encoding="utf-8")
            if project_path(config["weight_manifest"]).exists()
            else None,
        }
    )
    reset_if_input_changed(key)
    reset_if_strength_changed(render_alpha)
    with workspace, st.container(key="letter_workspace"):
        st.subheader("让风景，慢慢成为一封信。")
        st.caption("先生成喜欢的画面，再打开信笺，写下想寄出的那句话。")
        if content_bytes is None or style_bytes is None:
            st.info("请先选择或上传内容图和风格图。")
            return
        try:
            processed = prepare_content(content_bytes, tuple(config["content_hw"]))
            prepare_style(
                style_bytes, config["style_long_side"], config["minimum_style_short_side"]
            )
        except (ValueError, OSError) as error:
            st.error(f"图片无法处理：{error}")
            return
        ready, reasons = integration_status(config)
        generate_column, search_column = st.columns([1.6, 1])
        generate_clicked = generate_column.button("生成风景", type="primary", width="stretch")
        search_clicked = search_column.button(
            "运行区域搜索", disabled=not ready or creative, width="stretch"
        )
        if generate_clicked or search_clicked:
            _generate(config, asset_stats, content_bytes, style_bytes, render_alpha, search_clicked)
        if st.session_state.get("last_error"):
            st.error(st.session_state.last_error)
        original, rendered = st.columns(2, gap="medium")
        with original:
            st.caption("01  /  镜头里的风景")
            st.image(processed.rgb, width="stretch")
        default = st.session_state.get("default")
        selected_result = st.session_state.get("best", default)
        with rendered:
            st.caption("02  /  画意里的风景")
            if selected_result is None:
                empty_preview()
            else:
                st.image(tensor_to_pil(selected_result.image_rgb), width="stretch")
        st.caption("内容图按方形中心裁剪。生成后，可在独立信笺窗口中选择版式与下载。")
        create, download = st.columns([1.6, 1])
        if create.button("制作明信片 →", type="primary", disabled=default is None, width="stretch"):
            st.session_state.postcard_open = True
        if selected_result is not None:
            download.download_button(
                "下载风景原图",
                data=png_bytes(selected_result.image_rgb),
                file_name=f"stylized_{config['content_hw'][0]}.png",
                mime="image/png",
                on_click="ignore",
                width="stretch",
            )
        with st.expander("实验信息与区域搜索"):
            if not ready:
                st.caption("风景生成与明信片可使用。区域搜索等待接入 M5 区域模块与 M6 搜索模块。")
                for reason in reasons:
                    st.code(reason, language=None)
            elif creative:
                st.caption("区域搜索使用固定实验基线；切换“实验基线”模式后可运行。")
            if "search_result" in st.session_state:
                result = st.session_state.search_result
                st.image(tensor_to_pil(default.image_rgb), caption="搜索前的默认结果", width=256)
                st.write(
                    {
                        "状态": result.status,
                        "使用默认图": result.metadata["used_default"],
                        "已评价状态": result.candidate_count,
                        "评分": result.metrics,
                    }
                )
            if selected_result is not None:
                st.image(
                    alpha_image(selected_result.alpha_map), caption="特征层风格强度图", width=256
                )
                if "regions" in st.session_state:
                    st.image(
                        region_overlay(
                            st.session_state.pair.content_rgb, st.session_state.regions.labels
                        ),
                        caption="M5 区域划分叠加图",
                        width=360,
                    )
                st.json(
                    {
                        "actual_device": str(st.session_state.engine.device),
                        "feature_hw": st.session_state.pair.feature_hw,
                        "counters": dict(st.session_state.engine.counters),
                        "last_generation_wall_seconds": st.session_state.get("generation_seconds"),
                        "timing_note": "交互墙钟时间，排除模型加载；"
                        "正式同步计时使用 src.bench.profile",
                    }
                )
    st.html(
        '<footer class="letter-footer">山海自有归期，风景值得一封来信。'
        " / MADE WITH A LITTLE WONDER</footer>"
    )
    if st.session_state.get("postcard_open") and "default" in st.session_state:
        postcard_dialog(config, key, content_bytes, style_bytes)


if __name__ == "__main__":
    main()
