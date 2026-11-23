from uuid import uuid4

import numpy as np
import pytest
import torch
from PIL import Image

from src.contracts import PreparedPair, RegionPack
from src.models.adain import AdaINEngine
from src.project import load_config, project_path


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Choose a fresh project-local base before pytest constructs TempPathFactory.

    Windows creates pytest temporary directories with owner-only permissions.
    User and sandbox sessions must never reuse or scan each other's private bases.
    An explicitly supplied --basetemp remains the caller's choice.
    """
    if config.option.basetemp is None:
        parent = project_path('outputs/test-temp')
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / f'run-{uuid4().hex}')


@pytest.fixture
def fake_pair():
    generator = torch.Generator().manual_seed(42)
    content = torch.randn((1, 4, 4, 4), generator=generator)
    return PreparedPair(
        "fixture",
        Image.new("RGB", (8, 8)),
        Image.new("RGB", (8, 8)),
        torch.zeros(1, 3, 8, 8),
        content,
        torch.randn(content.shape, generator=generator),
        (4, 4),
        (8, 8),
        "fixture",
        {},
    )


@pytest.fixture
def two_regions():
    # Synthetic regions are only test fixtures, never used as production M5 output.
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:, 4:] = 1
    masks = torch.zeros(2, 4, 4)
    masks[0, :, :2], masks[1, :, 2:] = 1, 1
    return RegionPack((0, 1), labels, masks, ((0, 1, 1.0),), {"test_only": True})


@pytest.fixture(scope="session")
def real_engine():
    if not project_path("resources/weights/manifest.json").is_file():
        pytest.skip("Actual downloaded weights not configured; this is not a passed inference test")
    return AdaINEngine(load_config())


@pytest.fixture(scope="session")
def real_pair(real_engine):
    return real_engine.prepare(
        project_path("resources/images/content/pilot/mountain/mountain_01.jpg"),
        project_path("resources/images/styles/met_436535.jpg"),
    )
