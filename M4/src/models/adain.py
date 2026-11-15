"""Strict checkpoint loading, reusable pair features and whole-image AdaIN decoding."""

import json
import math
from copy import deepcopy

import torch
from torch import nn

from src.contracts import GenerationResult, PreparedPair, PreparedRegions, RegionPack
from src.models.fusion import fuse_region_features, prepare_regions, validate_alphas
from src.models.networks import build_decoder, build_encoder
from src.models.preprocessing import ImageInput, prepare_content, prepare_style
from src.project import digest, file_sha256, load_config, project_path, runtime_info


def calc_mean_std(features: torch.Tensor, eps=1e-5):
    """Match upstream: sample variance (correction=1), add epsilon, then sqrt."""
    flat = features.flatten(2)
    return flat.mean(2)[:, :, None, None], (flat.var(2, correction=1) + eps).sqrt()[
        :, :, None, None
    ]


def adaptive_instance_normalization(content, style, eps=1e-5):
    content_mean, content_std = calc_mean_std(content, eps)
    style_mean, style_std = calc_mean_std(style, eps)
    return (content - content_mean) / content_std * style_std + style_mean


class AdaINEngine:
    def __init__(self, config: dict | None = None):
        self.config = deepcopy(config or load_config())
        requested = self.config["device"]
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch build has no available CUDA device"
            )
        self.device = torch.device(
            "cuda"
            if requested == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested == "auto"
            else requested
        )
        if self.device.type == "cpu":
            torch.set_num_threads(int(self.config.get("cpu_threads", 4)))
        manifest_path = project_path(self.config["weight_manifest"])
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run: node src/tools/download_assets.mjs"
            )
        self.weight_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        encoder = build_encoder()
        decoder = build_decoder()
        for name, module in (("encoder", encoder), ("decoder", decoder)):
            checkpoint = project_path(self.config[f"{name}_checkpoint"])
            expected = self.weight_manifest["checkpoints"][name]["sha256"]
            if file_sha256(checkpoint) != expected:
                raise ValueError(f"{name} SHA256 mismatch: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            module.load_state_dict(state, strict=True)
        self.encoder = nn.Sequential(*list(encoder.children())[:31]).to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        for parameter in (*self.encoder.parameters(), *self.decoder.parameters()):
            parameter.requires_grad_(False)
        self.model_hash = digest(self.weight_manifest["checkpoints"])
        self.config_hash = digest(self.config)
        self.counters = {"prepare_calls": 0, "encoder_calls": 0, "decoder_calls": 0}
        self.runtime = runtime_info(self.device)
        self.runtime.update(
            {
                "model_hash": self.model_hash,
                "config_hash": self.config_hash,
                "upstream_ref": self.weight_manifest["upstream_ref"],
            }
        )

    @torch.inference_mode()
    def prepare(
        self, content_image: ImageInput, style_image: ImageInput, *, pair_id: str | None = None
    ) -> PreparedPair:
        content = prepare_content(content_image, tuple(self.config["content_hw"]))
        style = prepare_style(
            style_image, self.config["style_long_side"], self.config["minimum_style_short_side"]
        )
        return self.prepare_preprocessed(content, style, pair_id=pair_id)

    @torch.inference_mode()
    def prepare_preprocessed(self, content, style, *, pair_id: str | None = None) -> PreparedPair:
        """Encode already-preprocessed CPU images; useful for isolated phase timing."""
        if (
            tuple(content.tensor.shape) != (1, 3, *self.config["content_hw"])
            or content.tensor.dtype != torch.float32
            or style.tensor.dtype != torch.float32
            or max(style.tensor.shape[-2:]) != self.config["style_long_side"]
            or min(style.tensor.shape[-2:]) < self.config["minimum_style_short_side"]
        ):
            raise ValueError("Preprocessed inputs do not match the active configuration")
        content_tensor = content.tensor.to(self.device)
        content_features = self.encoder(content_tensor)
        style_features = self.encoder(style.tensor.to(self.device))
        adain_features = adaptive_instance_normalization(
            content_features, style_features, self.config["adain_eps"]
        )
        if not torch.isfinite(adain_features).all().item():
            raise ValueError("AdaIN produced nonfinite features")
        metadata = {
            "content": content.metadata,
            "style": style.metadata,
            "model_hash": self.model_hash,
            "config_hash": self.config_hash,
            "actual_device": str(self.device),
            "dtype": "float32",
            "style_hw": list(style.tensor.shape[-2:]),
        }
        key = digest(metadata)
        self.counters["prepare_calls"] += 1
        self.counters["encoder_calls"] += 2
        return PreparedPair(
            pair_id or key[:16],
            content.rgb,
            style.rgb,
            content_tensor,
            content_features,
            adain_features,
            tuple(content_features.shape[-2:]),
            tuple(content.tensor.shape[-2:]),
            key,
            metadata,
        )

    def _check_pair(self, pair: PreparedPair):
        if (
            pair.metadata["model_hash"] != self.model_hash
            or pair.metadata["config_hash"] != self.config_hash
        ):
            raise ValueError("Pair belongs to another model/configuration; prepare it again")
        if pair.content_features.device != self.device:
            raise ValueError("Pair is on a different device")

    def _decode(self, pair, mixed, alpha_map, metadata):
        image = self.decoder(mixed).clamp(0, 1)
        self.counters["decoder_calls"] += 1
        if image.shape != pair.content_tensor.shape or not torch.isfinite(image).all().item():
            raise ValueError(f"Invalid decoder result: {tuple(image.shape)}")
        return GenerationResult(image, alpha_map, {"pair_id": pair.pair_id, **metadata})

    @torch.inference_mode()
    def render_global(self, pair: PreparedPair, alpha: float = 0.6) -> GenerationResult:
        self._check_pair(pair)
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            raise ValueError("Global alpha must be finite and in [0,1]")
        alpha_map = torch.full((1, 1, *pair.feature_hw), alpha, device=self.device)
        mixed = alpha * pair.adain_features + (1 - alpha) * pair.content_features
        return self._decode(pair, mixed, alpha_map, {"kind": "global", "alpha": alpha})

    def prepare_regions(self, pair: PreparedPair, regions: RegionPack) -> PreparedRegions:
        self._check_pair(pair)
        return prepare_regions(pair, regions)

    @torch.inference_mode()
    def render_regions(
        self,
        pair: PreparedPair,
        regions: RegionPack | PreparedRegions,
        region_alphas: tuple[float, ...],
    ) -> GenerationResult:
        self._check_pair(pair)
        prepared = (
            self.prepare_regions(pair, regions) if isinstance(regions, RegionPack) else regions
        )
        if prepared.pair_cache_key != pair.cache_key:
            raise ValueError("Regions belong to a different image pair")
        alphas = validate_alphas(region_alphas, len(prepared.region_ids), self.device)
        mixed, alpha_map = fuse_region_features(
            pair.content_features, pair.adain_features, prepared.soft_masks, alphas
        )
        return self._decode(
            pair,
            mixed,
            alpha_map,
            {
                "kind": "regions",
                "region_ids": prepared.region_ids,
                "region_alphas": list(region_alphas),
                "region_cache_key": prepared.cache_key,
            },
        )

    def stylize_global(self, content_image, style_image, alpha=0.6):
        return self.render_global(self.prepare(content_image, style_image), alpha)
