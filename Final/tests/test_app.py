import io
import json

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.models.preprocessing import png_bytes
from src.project import load_config, project_path
from src.ui.comparison import crop_box


@pytest.mark.ui
def test_all_manifest_photos_available_and_scene_filters():
    contents = json.loads(
        project_path("resources/images/manifest.json").read_text(encoding="utf-8")
    )["content_images"]
    app = AppTest.from_file(str(project_path("app.py")), default_timeout=30).run()
    choices = app.selectbox(key="content_upload_choice")
    assert len(contents) == 21
    assert set(choices.options) == {item["label_zh"] for item in contents} | {"上传自己的图片…"}
    assert choices.value == "mountain_01"
    for scene, group in (("城市", "downtown"), ("海岸", "beach"), ("山地", "mountain")):
        app.radio(key="scene_filter").set_value(scene).run()
        choices = app.selectbox(key="content_upload_choice")
        assert set(choices.options) == {
            item["label_zh"] for item in contents if item["class_name"] == group
        } | {"上传自己的图片…"}
        choices.select(f"{group}_07").run()
        assert not app.error and not app.exception
    app.selectbox(key="content_upload_choice").select("upload").run()
    assert not app.exception and app.info
    assert next(b for b in app.button if b.label == "生成双图对照").disabled


def test_zoom_uses_same_coordinates_and_locates_largest_change():
    left = Image.new("RGB", (40, 24))
    right = left.copy()
    right.paste("white", (7, 5, 17, 11))
    assert crop_box(left, right, 4, "变化最大处") == (7, 5, 17, 11)
    assert crop_box(left, right, 2, "右下") == (20, 12, 40, 24)


@pytest.mark.ui
@pytest.mark.weights
def test_text_template_download_and_new_input_state():
    if not project_path("resources/weights/manifest.json").exists():
        pytest.skip("Actual weights required for UI inference checks")
    app = AppTest.from_file(str(project_path("app.py")), default_timeout=60).run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "制作明信片 →").disabled
    next(b for b in app.button if b.label == "仅生成 AdaIN").click().run()
    assert not app.exception and not app.error
    engine = app.session_state["engine"]
    counters = dict(engine.counters)
    assert len(app.text_input) == 0  # no editor below the generated images
    next(b for b in app.button if b.label == "制作明信片 →").click().run()
    assert not app.exception and not app.error
    assert app.session_state["postcard_open"]
    original = app.session_state["postcard"]
    app.text_input[0].set_value("来自群山的问候").run()
    assert not app.exception
    assert dict(engine.counters) == counters
    assert app.session_state["postcard"] != original
    next(box for box in app.selectbox if box.label == "明信片模板").select("border").run()
    assert not app.exception and dict(engine.counters) == counters
    download = app.get("download_button")[0]
    assert download.proto.ignore_rerun
    next(b for b in app.button if b.label == "完成，返回风景").click().run()
    assert not app.session_state["postcard_open"] and len(app.text_input) == 0
    next(b for b in app.button if b.label == "制作明信片 →").click().run()
    assert app.text_input[0].value == "来自群山的问候"
    assert dict(engine.counters) == counters
    app.text_area[0].set_value("这一行太多了\n" * 7).run()
    assert app.error and "postcard" not in app.session_state
    assert all(item.proto.label != "下载这封来信 PNG" for item in app.get("download_button"))
    app.text_area[0].set_value("把眼前的风景，寄给远方的你。").run()
    assert not app.error and "postcard" in app.session_state
    assert dict(engine.counters) == counters
    style = next(box for box in app.selectbox if box.label == "风格参考")
    style.select("met_45434").run()
    assert not app.exception
    assert "default" not in app.session_state and "pair" not in app.session_state
    assert not app.session_state["postcard_open"]
    next(b for b in app.button if b.label == "仅生成 AdaIN").click().run()
    assert not app.exception and not app.error
    assert engine.counters["prepare_calls"] == counters["prepare_calls"] + 1
    assert engine.counters["decoder_calls"] == counters["decoder_calls"] + 1


@pytest.mark.ui
@pytest.mark.weights
@pytest.mark.evaluator
def test_adain_and_search_share_preprocessing_and_export_resolution(monkeypatch):
    if not project_path("resources/weights/vgg19_features.pth").exists():
        pytest.skip("Actual evaluator weights required for search")
    config = load_config()
    config["search_config"]["budget"] = 4
    monkeypatch.setattr("src.app.load_config", lambda: config)
    app = AppTest.from_file(str(project_path("app.py")), default_timeout=90).run()
    next(b for b in app.button if b.label == "仅生成 AdaIN").click().run()
    assert not app.exception and not app.error
    pair = app.session_state["pair"]
    engine = app.session_state["engine"]
    before = dict(engine.counters)
    baseline = app.session_state["default"]
    assert pair.input_hw == (512, 512)
    app.slider(key="direct_alpha").set_value(0.3).run()
    assert "default" not in app.session_state
    next(b for b in app.button if b.label == "仅生成 AdaIN").click().run()
    assert not app.exception and not app.error
    custom = app.session_state["default"]
    assert custom.metadata["alpha"] == 0.3
    assert not custom.image_rgb.equal(baseline.image_rgb)
    assert app.session_state["pair"] is pair
    assert engine.counters["encoder_calls"] == before["encoder_calls"]
    search = next(b for b in app.button if b.label == "生成双图对照")
    assert not search.disabled
    search.click().run()
    assert not app.exception and not app.error
    assert app.session_state["pair"] is pair
    baseline = app.session_state["default"]
    assert baseline is not custom
    assert engine.counters["encoder_calls"] == before["encoder_calls"]
    assert baseline.metadata["alpha"] == 0.6
    best = app.session_state["best"]
    assert best.image_rgb.shape == baseline.image_rgb.shape == (1, 3, 512, 512)
    for output in (baseline, best):
        with Image.open(io.BytesIO(png_bytes(output.image_rgb))) as png:
            assert png.size == (512, 512)
    assert {button.proto.label for button in app.get("download_button")} >= {
        "下载 AdaIN PNG",
        "下载搜索结果 PNG",
    }
    before_zoom = dict(engine.counters)
    app.radio(key="compare_zoom").set_value(4).run()
    app.selectbox(key="compare_position").select("右下").run()
    assert any("(384, 384)" in caption.value for caption in app.caption)
    assert not app.exception and engine.counters == before_zoom
    # Switching directly to a held-out photograph clears the previous pair and its results.
    app.selectbox(key="content_upload_choice").select("downtown_06").run()
    assert not app.exception and "best" not in app.session_state
    assert all("搜索结果" not in button.proto.label for button in app.get("download_button"))
