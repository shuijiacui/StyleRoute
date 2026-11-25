"""Local, verified IQA-PyTorch NIMA VGG16 AVA checkpoint, MOS in [1,10].

Architecture/preprocessing follow pyiqa.archs.nima_arch.NIMA. The full AVA
checkpoint includes the backbone, so no redundant ImageNet download is needed.

NIMA adaptation: Copyright (c) 2022 Chaofeng Chen.
PolyForm Noncommercial 1.0.0, see resources/upstream/pyiqa/LICENSE.
Modified for offline checkpoint verification and no redundant backbone download.
"""

import json
from functools import lru_cache

import torch
from torch import nn
from torchvision.transforms import functional as TF

from src.project import file_sha256, project_path

CHECKPOINT = "resources/weights/NIMA_VGG16_ava-dc4e8265.pth"
URL = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/"


class NIMA(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        self.base_model = timm.create_model("vgg16", pretrained=False, features_only=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0), nn.Linear(512, 10), nn.Softmax(dim=-1)
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, image):
        image = TF.center_crop(TF.resize(image, 224, antialias=True), 224)
        features = self.base_model((image - self.mean) / self.std)[-1]
        distribution = self.classifier(self.global_pool(features))
        return (distribution * torch.arange(1, 11, device=image.device)).sum(dim=-1)


@lru_cache(maxsize=2)
def load_nima(device="cpu"):
    path = project_path(CHECKPOINT)
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        raise RuntimeError("NIMA weights missing. Run python -m src.tools.prepare_aesthetics")
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if info.get("sha256") != file_sha256(path):
        raise ValueError("NIMA checkpoint checksum mismatch")
    model = NIMA()
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights.get("params", weights), strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model, info


class AestheticMetric:
    def __init__(self, device):
        self.model, self.provenance = load_nima(str(device))

    @torch.inference_mode()
    def __call__(self, image):
        return float(self.model(image).item())
