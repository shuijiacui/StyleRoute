import pytest
from streamlit.testing.v1 import AppTest

from src.project import project_path


@pytest.mark.ui
@pytest.mark.weights
def test_text_template_download_and_new_input_state():
    if not project_path("resources/weights/manifest.json").exists():
        pytest.skip("Actual weights required for UI inference checks")
    app = AppTest.from_file(str(project_path("app.py")), default_timeout=60).run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "运行区域搜索").disabled
    assert next(b for b in app.button if b.label == "制作明信片 →").disabled
    next(b for b in app.button if b.label == "生成风景").click().run()
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
    next(b for b in app.button if b.label == "生成风景").click().run()
    assert not app.exception and not app.error
    assert engine.counters["prepare_calls"] == counters["prepare_calls"] + 1
    assert engine.counters["decoder_calls"] == counters["decoder_calls"] + 1


@pytest.mark.ui
@pytest.mark.weights
def test_strength_reuses_features_and_baseline_stays_256():
    if not project_path("resources/weights/manifest.json").exists():
        pytest.skip("Actual weights required for UI inference checks")
    app = AppTest.from_file(str(project_path("app.py")), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    pair = app.session_state["pair"]
    engine = app.session_state["engine"]
    before = dict(engine.counters)
    assert pair.input_hw == (512, 512)
    app.slider[0].set_value(0.6).run()
    assert "default" not in app.session_state
    assert app.session_state["pair"] is pair
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert engine.counters["encoder_calls"] == before["encoder_calls"]
    assert engine.counters["decoder_calls"] == before["decoder_calls"] + 1
    assert app.session_state["default"].metadata["alpha"] == 0.6
    next(box for box in app.selectbox if box.label == "生成模式").select("baseline").run()
    assert "pair" not in app.session_state and "default" not in app.session_state
    assert len(app.slider) == 0
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state["pair"].input_hw == (256, 256)
    assert app.session_state["default"].metadata["alpha"] == 0.6
