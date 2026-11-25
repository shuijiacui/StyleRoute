"""Relocation must preserve real assets and make historical configurations usable."""

import json

from src.project import ROOT, load_config, project_path


def test_resources_and_historical_paths_resolve(tmp_path):
    for old, current in (
        ("data/styles/met_436535.jpg", "resources/images/styles/met_436535.jpg"),
        ("weights/decoder.pth", "resources/weights/decoder.pth"),
        ("assets/fonts/NotoSansCJKsc-Regular.otf", "resources/fonts/NotoSansCJKsc-Regular.otf"),
    ):
        assert project_path(old) == ROOT / current
        assert project_path(old).is_file()
    assert project_path(ROOT / "README.md") == ROOT / "README.md"
    # A clean checkout must not depend on the author's untracked demo output.
    snapshot = {
        **load_config(),
        "encoder_checkpoint": "weights/vgg_normalized.pth",
        "font_path": "assets/fonts/NotoSansCJKsc-Regular.otf",
    }
    snapshot_path = tmp_path / "historical_config.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    historical = load_config(snapshot_path)
    assert project_path(historical["encoder_checkpoint"]).is_file()
    assert project_path(historical["font_path"]).is_file()
