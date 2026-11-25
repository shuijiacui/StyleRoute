"""Local landscape-letter studio. Run: python -m streamlit run app.py."""

import hashlib
import json
import time

import streamlit as st

from src.integration import integration_status, run_search
from src.models.adain import AdaINEngine
from src.models.preprocessing import prepare_content, prepare_style
from src.outputs import alpha_image, dataset_pair_id, region_overlay
from src.project import digest, load_config, project_path
from src.ui.comparison import input_preview, result_view, score_table
from src.ui.postcard import postcard_dialog
from src.ui.theme import apply_theme


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


def _generate(config, asset_stats, content, style, search=False, alpha=None):
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
        with st.spinner("正在生成对照图……" if search else "正在生成 AdaIN 直出……"):
            if "pair" not in st.session_state:
                st.session_state.pair = engine.prepare(
                    content, style, pair_id=dataset_pair_id(content, style)
                )
            target_alpha = config["default_alpha"] if search or alpha is None else alpha
            previous = st.session_state.get("default")
            if previous is not None and previous.metadata["alpha"] != target_alpha:
                _clear_results()
            if "default" not in st.session_state:
                st.session_state.default = engine.render_global(st.session_state.pair, target_alpha)
            if search:
                for name in ("best", "regions", "search_result", "postcard_key"):
                    st.session_state.pop(name, None)
                progress = st.empty()

                def show_progress(event):
                    if event["event"] == "candidate_scored":
                        count = event.get("candidate_count", event.get("decoder_calls", 0))
                        progress.caption(f"已评价 {count} 个候选")

                try:
                    regions, best, result = run_search(
                        engine,
                        st.session_state.pair,
                        st.session_state.default,
                        progress_callback=show_progress,
                    )
                finally:
                    progress.empty()
                st.session_state.regions = regions
                st.session_state.best = best
                st.session_state.search_result = result
        st.session_state.generation_seconds = time.perf_counter() - start
        st.session_state.pop("last_error", None)
    except Exception as error:
        st.session_state.last_error = f"{type(error).__name__}: {error}"


def _select_image(label, items, upload_key):
    options = {item["id"]: item for item in items}
    selected = st.selectbox(
        label,
        [*options, "upload"],
        format_func=lambda key: options[key]["label_zh"] if key in options else "上传自己的图片…",
        key=f"{upload_key}_choice",
    )
    if selected == "upload":
        uploaded = st.file_uploader(
            f"上传{label}",
            type=["jpg", "jpeg", "png", "webp"],
            key=upload_key,
        )
        return (uploaded.getvalue(), uploaded.name, None) if uploaded else (None, "", None)
    record = options[selected]
    return project_path(record["local_path"]).read_bytes(), record["label_zh"], record


def _materials(contents, styles):
    st.subheader("测试素材")
    scenes = {"全部": None, "山地": "mountain", "海岸": "beach", "城市": "downtown"}
    scene = st.radio("场景", list(scenes), horizontal=True, key="scene_filter")
    filtered = [
        item for item in contents if not scenes[scene] or item["class_name"] == scenes[scene]
    ]
    content, name, content_record = _select_image("内容图", filtered, "content_upload")
    style, style_name, style_record = _select_image("风格参考", styles, "style_upload")
    with st.popover("素材来源", width="stretch"):
        st.caption(f"图库共 {len(contents)} 张内容图、{len(styles)} 张风格参考。")
        for item in (content_record, style_record):
            if item:
                usage = {"pilot": "试运行", "evaluation": "正式评价"}.get(item.get("usage"))
                st.caption(
                    f"[{item['label_zh']}]({item['object_url']}) · {item['artist']} · "
                    f"{item['license']}" + (f" · {usage}" if usage else "")
                )
    return content, name, style, style_name


def main():
    st.set_page_config(page_title="山海来信 · 风格迁移对照台", page_icon="✉️", layout="wide")
    apply_theme()
    config = load_config()
    manifest = json.loads(
        project_path("resources/images/manifest.json").read_text(encoding="utf-8")
    )
    contents = sorted(manifest["content_images"], key=lambda item: item["class_name"] != "mountain")
    styles = manifest["style_images"]
    with st.container(key="studio_toolbar"):
        content, content_label, style, style_label = _materials(contents, styles)
        ready, reasons = integration_status(config)
        missing_input = content is None or style is None
        direct_alpha = st.slider(
            "AdaIN 全局风格强度",
            0.0,
            1.0,
            float(config["default_alpha"]),
            0.05,
            key="direct_alpha",
            on_change=_clear_results,
            help="用于「仅生成 AdaIN」。0 为内容重建，1 为完整风格迁移。",
        )
        generate_clicked = st.button("仅生成 AdaIN", disabled=missing_input, width="stretch")
        search_clicked = st.button(
            "生成双图对照",
            type="primary",
            disabled=missing_input or not ready,
            width="stretch",
        )
        st.caption(f"双图对照基准 α = {config['default_alpha']:g}")
        card_action = st.empty()

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
            "content": hashlib.sha256(content).hexdigest() if content else None,
            "style": hashlib.sha256(style).hexdigest() if style else None,
            "config": config,
            "asset_stats": asset_stats,
            "assets": project_path(config["weight_manifest"]).read_text(encoding="utf-8")
            if project_path(config["weight_manifest"]).exists()
            else None,
        }
    )
    reset_if_input_changed(key)
    with st.container(key="compare_workspace"):
        if not ready:
            st.warning("区域搜索暂不可用。" + " ".join(map(str, reasons)))
        if missing_input:
            st.info("请选择或上传内容图和风格图。")
            return
        try:
            processed = prepare_content(content, tuple(config["content_hw"]))
            reference = prepare_style(
                style, config["style_long_side"], config["minimum_style_short_side"]
            )
        except (ValueError, OSError) as error:
            st.error(f"图片无法处理：{error}")
            return
        if search_clicked or generate_clicked:
            _generate(
                config, asset_stats, content, style, search=search_clicked, alpha=direct_alpha
            )
        if st.session_state.get("last_error"):
            st.error(st.session_state.last_error)
        default = st.session_state.get("default")
        best = st.session_state.get("best")
        result = st.session_state.get("search_result")
        if default is None:
            st.subheader("输入预览")
            input_preview(processed.rgb, reference.rgb, content_label, style_label)
        else:
            names = ["结果对照", "本次输入", "区域分析"]
            if result is not None:
                names.append("评分")
            tabs = st.tabs(names)
            with tabs[0]:
                result_view(default, best, default.metadata["alpha"])
            with tabs[1]:
                input_preview(processed.rgb, reference.rgb, content_label, style_label)
            with tabs[2], st.container(key="region_view"):
                strength, partition = st.columns(2, gap="medium")
                selected = best if best is not None else default
                strength.markdown("#### 风格强度")
                strength.image(alpha_image(selected.alpha_map), width="stretch")
                if "regions" in st.session_state:
                    partition.markdown("#### 区域划分")
                    partition.image(
                        region_overlay(
                            st.session_state.pair.content_rgb, st.session_state.regions.labels
                        ),
                        width="stretch",
                    )
            if result is not None:
                with tabs[3]:
                    score_table(result)
                    choice = "采用直出" if result.metadata["used_default"] else "采用搜索结果"
                    st.caption(f"{choice}　·　已评价 {result.candidate_count} 个候选状态")
                    if "beta" in result.metadata:
                        st.caption(
                            f"CMA-ES · 局部补偿 β = {result.metadata['beta']:.3f} · "
                            f"区域强度 {min(result.region_alphas):.3f}～"
                            f"{max(result.region_alphas):.3f}"
                        )
            st.caption(
                f"{config['content_hw'][1]} × {config['content_hw'][0]} px　·　"
                f"{st.session_state.engine.device}　·　"
                f"本次用时 {st.session_state.generation_seconds:.1f} 秒"
            )
    if card_action.button("制作明信片 →", disabled=default is None, width="stretch"):
        st.session_state.postcard_open = True
    if st.session_state.get("postcard_open") and default is not None:
        postcard_dialog(config, key, content, style)


if __name__ == "__main__":
    main()
