import numpy as np
import torch

def test_dummy_add():
    assert 1 + 1 == 2

def test_soft_mask_shape():
    from M5.regions.masks import generate_soft_masks
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[:16, :16] = 0
    labels[:16, 16:] = 1
    labels[16:, :16] = 2
    labels[16:, 16:] = 3
    region_ids = np.arange(4, dtype=np.int32)

    masks = generate_soft_masks(
        region_labels=labels,
        region_ids=region_ids,
        feature_hw=(16, 16),
        sigma=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert masks.shape == (4, 16, 16)
    assert torch.allclose(masks.sum(dim=0), torch.ones(16, 16), atol=1e-3)