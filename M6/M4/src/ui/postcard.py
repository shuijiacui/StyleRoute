"""Postcard editing is an independent dialog; edits never invoke the model."""

from datetime import datetime

import streamlit as st

from src.outputs import save_run
from src.postcard.render import (
    CANVAS_WH,
    DESIGN_VERSION,
    TEMPLATE_LABELS,
    TEMPLATES,
    render_postcard,
)
from src.project import digest, file_sha256, project_path


def close_postcard():
    st.session_state.postcard_open = False


@st.dialog("写一封山海来信", width="large", on_dismiss=close_postcard)
def postcard_dialog(config, input_key, content_bytes, style_bytes):
    st.caption("选一张信笺，写几句心意。修改文字或版式后，预览会随之更新。")
    default = st.session_state.default
    selected = st.session_state.get("best", default)
    draft = st.session_state.get("postcard_draft", {})
    editor, preview = st.columns([1, 2], gap="large")
    with editor:
        template = st.selectbox(
            "明信片模板",
            TEMPLATES,
            index=TEMPLATES.index(draft.get("template", "minimal")),
            format_func=TEMPLATE_LABELS.get,
            key="card_template",
        )
        st.caption(
            {
                "minimal": "清透留白 · 一帧风景与一枚朱红邮戳",
                "border": "复古邮笺 · 航空信封边纹与书写横线",
                "journal": "旅行手记 · 贴纸相框与随行笔记",
            }[template]
        )
        title = st.text_input(
            "明信片标题", value=draft.get("title", "山海来信"), max_chars=100, key="card_title"
        )
        message = st.text_area(
            "明信片正文",
            value=draft.get("message", "把眼前的风景，\n寄给远方的你。\n愿山海可期，来日可盼。"),
            max_chars=1500,
            height=175,
            key="card_message",
        )
        st.caption("短短几句就很好。正文按实际字宽换行，最多容纳五行。")
    st.session_state.postcard_draft = {"title": title, "message": message, "template": template}
    try:
        font = project_path(config["font_path"])
        heading = project_path(config.get("postcard_heading_font_path", config["font_path"]))
        hashes = {"font_sha256": file_sha256(font), "heading_font_sha256": file_sha256(heading)}
        export_key = digest(
            {
                "input": input_key,
                "image": selected.metadata,
                **st.session_state.postcard_draft,
                **hashes,
                "design": DESIGN_VERSION,
            }
        )
        if st.session_state.get("postcard_key") != export_key:
            with preview, st.spinner("正在排好这封来信……"):
                postcard = render_postcard(
                    selected.image_rgb, title, message, template, font, title_font_path=heading
                )
            st.session_state.postcard = postcard
            st.session_state.postcard_key = export_key
            st.session_state.pop("saved_directory", None)
        postcard = st.session_state.postcard
        with preview, st.container(key="letter_preview"):
            # Keep the original PNG bytes; the browser scales the 1500px image to its column.
            st.image(postcard, width=CANVAS_WH[0], output_format="PNG")
            st.caption(f"{TEMPLATE_LABELS[template]}  /  1500 × 1000 px · 预览与下载一致")
        with editor:
            st.download_button(
                "下载这封来信 PNG",
                data=postcard,
                file_name=f"山海来信_{TEMPLATE_LABELS[template]}.png",
                mime="image/png",
                type="primary",
                width="stretch",
                on_click="ignore",
            )
            if st.button("保存本次运行记录到项目", width="stretch"):
                directory = project_path(config["run_root"]) / datetime.now().strftime(
                    "app_%Y%m%d_%H%M%S_%f"
                )
                save_run(
                    directory,
                    st.session_state.engine,
                    st.session_state.pair,
                    default,
                    content_bytes,
                    style_bytes,
                    best=st.session_state.get("best"),
                    regions=st.session_state.get("regions"),
                    search_result=st.session_state.get("search_result"),
                    postcard=postcard,
                    postcard_metadata={
                        **st.session_state.postcard_draft,
                        **hashes,
                        "canvas_wh": list(CANVAS_WH),
                        "design_version": DESIGN_VERSION,
                    },
                )
                st.session_state.saved_directory = str(directory)
        if st.session_state.get("saved_directory"):
            st.success(f"已保存：{st.session_state.saved_directory}")
    except (ValueError, OSError) as error:
        # Never leave a previous, valid download attached to an invalid new draft.
        for name in ("postcard", "postcard_key", "saved_directory"):
            st.session_state.pop(name, None)
        preview.error(f"这封信还需要调整：{error}")
    st.caption("明信片画布用于排版；图片本身保留生成时的清晰度。")
    if st.button("完成，返回风景", width="stretch"):
        close_postcard()
        st.rerun()
