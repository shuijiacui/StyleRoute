"""Objective, constraints and risk apportionment for M6 (README section 3).

Implements the frozen scoring contract:

* Q — structural proxy: per-position channel MSD between generated and content
  activations at relu3_1, weighted by bounded nonnegative weights b(p) built
  from the ORIGINAL content gradients (never from the candidate);
* T — style statistics: layer-mean MSE of normalised Gram matrices
  G(F) = F Fᵀ / (C·H·W) against ONE fixed style reference, layers relu1_1..relu4_1;
* S / J — edge-weighted coordination of alphas and the search objective
  J = Q / (Q0 + delta) + lambda * S;
* feasibility — T <= (1 + epsilon) * T0 with the recorded violation;
* per-region risks risk_i = q_i / (Q0 + delta) + lambda * s_i where the
  structural and coordination contributions are apportioned so that
  sum(q_i) ~= Q and sum(s_i) ~= S (verified by tests).

The evaluator uses its own frozen torchvision VGG19 with ImageNet
normalisation, kept strictly separate from the M4 AdaIN encoder. Every
hyperparameter lives in ScorerConfig and is recorded in the run metadata.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .solver import Score, ScoreFailure

try:  # torch is optional at import time: the pure scoring math (S/J assembly,
    # coordination, risk apportionment) stays testable without it.
    import torch
    import torch.nn.functional as F
    from torchvision import models
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch
    torch = None
    F = None
    models = None

# relu1_1, relu2_1, relu3_1, relu4_1 positions in torchvision VGG19 .features
VGG19_RELU_LAYERS = (1, 6, 11, 20)
Q_LAYER_NAME = "relu3_1"


@dataclass(frozen=True)
class ScorerConfig:
    """Frozen scoring settings; every value is recorded in run metadata."""

    # objective
    delta: float = 1e-3  # > 0, denominator guard for Q/Q0
    delta_t: float = 1e-6  # > 0, denominator guard for violation
    lam: float = 1.0  # lambda >= 0 coordination weight
    epsilon: float = 0.05  # style slack over T0 (freeze after pilot)
    tolerance: float = 1e-9  # comparator strict-improvement margin
    # Q construction
    q_layer_index: int = 11  # relu3_1 inside torchvision vgg19.features
    grad_smooth_kernel: int = 3  # box smoothing for gradient magnitudes
    b_min: float = 0.1  # lower bound of b(p)
    b_max: float = 1.0  # upper bound of b(p)
    gram_normalisation: str = "c*h*w"  # G = F Fᵀ / (C*H*W), frozen
    style_eval_long_side: int | None = None  # None: use pair.style_rgb as given
    # provenance
    weights: str = "IMAGENET1K_V1"  # or "RANDOM" for logic tests only
    fingerprint: str = ""
    checkpoint: str = "resources/weights/vgg19_features.pth"

    def __post_init__(self):
        if self.q_layer_index not in VGG19_RELU_LAYERS:
            raise ValueError("q_layer_index must name one of the recorded ReLU layers")
        if self.gram_normalisation != "c*h*w":
            raise ValueError("Only c*h*w Gram normalisation is implemented")
        if self.delta <= 0 or self.delta_t <= 0 or self.lam < 0 or self.epsilon < 0:
            raise ValueError("Invalid objective/constraint parameters")
        if self.b_min <= 0 or self.b_max < self.b_min:
            raise ValueError("Invalid gradient weight bounds")
        if self.grad_smooth_kernel < 1 or self.grad_smooth_kernel % 2 != 1:
            raise ValueError("Gradient smoothing kernel must be positive and odd")


def _fingerprint(state_dict: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


class VGG19Evaluator:
    """Frozen evaluator network + Q/T computation for one prepared pair.

    Prepared ONCE per pair outside the candidate loop: content activations for
    Q, the b(p) weights, and per-layer style Gram references. Candidate calls
    only run the frozen network forward, never re-encode content/style.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        content_tensor: "torch.Tensor",  # [1,3,H,W] FP32 [0,1], canonical
        style_tensor: "torch.Tensor",  # [1,3,Hs,Ws] FP32 [0,1]
        soft_masks: "torch.Tensor",  # [N,Hf,Wf] FP32
        edges: Sequence[tuple[int, int, float]],
        config: ScorerConfig | None = None,
        device: Any = None,
    ) -> None:
        if torch is None:
            raise ScoreFailure("torch is required for the real VGG19 evaluator")
        self.config = config or ScorerConfig()
        cfg = self.config
        self.device = torch.device(device) if device is not None else content_tensor.device
        self.edges = tuple((int(i), int(j), float(w)) for i, j, w in edges)

        from torchvision.models.vgg import cfgs, make_layers

        if cfg.weights == "RANDOM":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                features = make_layers(cfgs["E"])
        elif cfg.weights == "IMAGENET1K_V1":
            import json

            from src.project import file_sha256, project_path

            path = project_path(cfg.checkpoint)
            manifest_path = path.with_suffix(".json")
            if not path.is_file() or not manifest_path.is_file():
                raise ScoreFailure(
                    "Missing evaluator weights. Run: python -m src.tools.prepare_evaluator"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("weights") != cfg.weights or file_sha256(path) != manifest.get(
                "sha256"
            ):
                raise ScoreFailure("Evaluator checkpoint SHA256 or weight identity mismatch")
            features = make_layers(cfgs["E"])
            features.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
        else:
            raise ValueError(f"unknown weights enum {cfg.weights!r}")
        self._fingerprint = _fingerprint(features.state_dict())
        self._features = features[: max(VGG19_RELU_LAYERS) + 1].to(self.device).eval()
        for parameter in self._features.parameters():
            parameter.requires_grad_(False)
        self._hooks: list[Any] = []
        self._activations: dict[int, torch.Tensor] = {}
        for index in VGG19_RELU_LAYERS:
            self._hooks.append(self._features[index].register_forward_hook(self._make_hook(index)))

        mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

        style = style_tensor.to(self.device, torch.float32)
        if cfg.style_eval_long_side is not None:
            style = self._resize_long_side(style, cfg.style_eval_long_side)
        with torch.no_grad():
            self._style_grams = self._grams(style)
            content = content_tensor.to(self.device, torch.float32)
            content_acts = self._forward(content)
            q_act = content_acts[cfg.q_layer_index]
            self._content_q = q_act
            self._b = self._build_b_weights(content, q_act.shape[-2:])
            self._masks = self._align_masks(soft_masks, q_act.shape[-2:])

    # -- network helpers ---------------------------------------------------

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            self._activations[index] = output

        return hook

    def _forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        self._activations = {}
        self._features((image - self._mean) / self._std)
        return self._activations

    @staticmethod
    def _resize_long_side(image: torch.Tensor, long_side: int) -> torch.Tensor:
        _, _, h, w = image.shape
        scale = long_side / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        return F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)

    @staticmethod
    def _gram(activation: torch.Tensor) -> torch.Tensor:
        # Frozen normalisation G = F Fᵀ / (C*H*W); F is [1,C,H,W].
        c, h, w = activation.shape[1], activation.shape[2], activation.shape[3]
        flat = activation.reshape(1, c, h * w)
        return flat @ flat.transpose(1, 2) / float(c * h * w)

    def _grams(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        acts = self._forward(image)
        return {index: self._gram(acts[index]) for index in VGG19_RELU_LAYERS}

    def _build_b_weights(self, content: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Bounded nonnegative b(p) from ORIGINAL content gradients (README 3.2)."""
        cfg = self.config
        gray = content.mean(dim=1, keepdim=True)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]) * 0.5
        gy[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]) * 0.5
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        k = cfg.grad_smooth_kernel
        if k > 1:
            mag = F.avg_pool2d(
                mag, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False
            )
        b = F.interpolate(mag, size=hw, mode="bilinear", align_corners=False)
        b = b / b.amax().clamp_min(1e-12)  # normalise to [0,1]
        b = b.clamp(cfg.b_min, cfg.b_max)  # bounded, nonnegative
        return b

    def _align_masks(self, masks: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Resample region soft weights to the Q feature grid and renormalise."""
        masks = masks.to(self.device, torch.float32)
        if tuple(masks.shape[-2:]) != tuple(hw):
            masks = F.interpolate(
                masks.unsqueeze(0), size=hw, mode="bilinear", align_corners=False
            ).squeeze(0)
            masks = masks.clamp_min(0)
        total = masks.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return masks / total

    # -- public scoring ------------------------------------------------------

    def style_loss(self, image: torch.Tensor) -> float:
        """T: layer-mean MSE of normalised Grams vs the fixed reference."""
        acts = self._forward(image.to(self.device, torch.float32))
        return self._style_from_acts(acts)

    def _style_from_acts(self, acts):
        losses = [
            F.mse_loss(self._gram(acts[index]), self._style_grams[index])
            for index in VGG19_RELU_LAYERS
        ]
        return float(sum(losses) / len(losses))

    def structural_loss(self, image: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Q and the spatial map b(p)*d(p)/sum(b) used for risk apportionment."""
        acts = self._forward(image.to(self.device, torch.float32))
        return self._structure_from_acts(acts)

    def _structure_from_acts(self, acts):
        gen = acts[self.config.q_layer_index]
        d = ((gen - self._content_q) ** 2).mean(dim=1, keepdim=True)  # [1,1,H,W]
        b = self._b
        denom = float(b.sum())
        if denom <= 0:
            raise ScoreFailure("Q denominator (sum of b) is zero")
        q_map = b * d / denom
        return float(q_map.sum()), q_map

    def calibrate(self, default_image: "torch.Tensor") -> tuple[float, float]:
        """Measure q0/t0 on the REAL default 0.6 result; enables compute()."""
        acts = self._forward(default_image.to(self.device, torch.float32))
        Q, _ = self._structure_from_acts(acts)
        T = self._style_from_acts(acts)
        if not (math.isfinite(Q) and math.isfinite(T)):
            raise ScoreFailure("default calibration produced non-finite scores")
        self.q0, self.t0 = Q, T
        self._q0_den = Q + self.config.delta
        return Q, T

    def compute(self, image: "torch.Tensor", alphas: Sequence[float]) -> Score:
        """Full Q/T/S/J/violation/feasible/risk record for one candidate."""
        cfg = self.config
        alphas = tuple(float(a) for a in alphas)
        if self._masks.shape[0] != len(alphas):
            raise ScoreFailure("alphas length does not match region count")
        if not hasattr(self, "_q0_den"):
            raise ScoreFailure("evaluator not calibrated; score the default result first")
        acts = self._forward(image.to(self.device, torch.float32))
        Q, q_map = self._structure_from_acts(acts)
        T = self._style_from_acts(acts)
        S = coordination(alphas, self.edges)
        q_i = apportion(q_map, self._masks)  # sum(q_i) ~= Q
        s_i = coordination_contrib(alphas, self.edges)  # sum(s_i) ~= S
        risks = tuple(q_i[i] / (self._q0_den) + cfg.lam * s_i[i] for i in range(len(alphas)))
        return assemble_score(Q, T, S, risks, self.q0, self.t0, cfg)


def coordination(alphas: Sequence[float], edges: Sequence[tuple[int, int, float]]) -> float:
    """S = sum_ij w_ij (a_i - a_j)^2 / sum_ij w_ij; 0 when no edges (README 3.4)."""
    num = 0.0
    den = 0.0
    for i, j, w in edges:
        num += w * (float(alphas[i]) - float(alphas[j])) ** 2
        den += w
    return num / den if den > 0 else 0.0


def coordination_contrib(
    alphas: Sequence[float], edges: Sequence[tuple[int, int, float]]
) -> tuple[float, ...]:
    """Per-region coordination contribution; each edge splits half/half (README 4)."""
    n = len(alphas)
    acc = [0.0] * n
    den = sum(w for _, _, w in edges)
    if den <= 0:
        return tuple(acc)
    for i, j, w in edges:
        c = 0.5 * w * (float(alphas[i]) - float(alphas[j])) ** 2 / den
        acc[i] += c
        acc[j] += c
    return tuple(acc)


def apportion(q_map: torch.Tensor, masks: torch.Tensor) -> tuple[float, ...]:
    """Distribute the spatial Q map over regions via their soft weights.

    masks are [N,H,W] resampled to the q_map grid and renormalised so that
    sum_i q_i equals Q up to float tolerance (checked in tests).
    """
    q = q_map[0, 0]  # [H,W]
    per_region = (masks * q.unsqueeze(0)).sum(dim=(1, 2))
    return tuple(float(v) for v in per_region)


def assemble_score(
    Q: float,
    T: float,
    S: float,
    risks: Sequence[float],
    q0: float,
    t0: float,
    config: ScorerConfig,
    *,
    source: str = "vgg19",
) -> Score:
    """Frozen feasibility constraint + objective assembly (README 3.4/3.5)."""
    limit = (1.0 + config.epsilon) * t0
    violation = max(0.0, (T - limit) / (t0 + config.delta_t))
    feasible = violation <= config.tolerance
    J = Q / (q0 + config.delta) + config.lam * S
    return Score(
        Q=Q,
        T=T,
        S=S,
        J=J,
        violation=violation,
        feasible=feasible,
        risks=tuple(risks),
        source=source,
    )


class PairScorer:
    """Evaluator bound to one pair + regions, exposing the solver interface.

    q0/t0 come from the REAL default 0.6 result; the default image itself is
    scored through the same code path as every candidate.
    """

    def __init__(self, evaluator: VGG19Evaluator) -> None:
        self.evaluator = evaluator
        self._q0: float | None = None
        self._t0: float | None = None

    def calibrate(self, default_image: "torch.Tensor") -> None:
        self._q0, self._t0 = self.evaluator.calibrate(default_image)

    @property
    def q0(self) -> float:
        if self._q0 is None:
            raise ScoreFailure("scorer not calibrated; score the default result first")
        return self._q0

    @property
    def t0(self) -> float:
        if self._t0 is None:
            raise ScoreFailure("scorer not calibrated; score the default result first")
        return self._t0

    def __call__(self, image: torch.Tensor, alphas: Sequence[float]) -> Score:
        return self.evaluator.compute(image, alphas)

    def provenance(self) -> dict[str, Any]:
        cfg = self.evaluator.config
        return {
            "weights": cfg.weights,
            "checkpoint": cfg.checkpoint if cfg.weights != "RANDOM" else None,
            "score_version": "integrated-v2-relu-corrected",
            "weight_fingerprint": self.evaluator._fingerprint,
            "relu_layers": list(VGG19_RELU_LAYERS),
            "q_layer_index": cfg.q_layer_index,
            "gram_normalisation": cfg.gram_normalisation,
            "imagenet_mean": list(VGG19Evaluator.IMAGENET_MEAN),
            "imagenet_std": list(VGG19Evaluator.IMAGENET_STD),
            "delta": cfg.delta,
            "delta_t": cfg.delta_t,
            "lam": cfg.lam,
            "epsilon": cfg.epsilon,
            "tolerance": cfg.tolerance,
            "b_bounds": [cfg.b_min, cfg.b_max],
            "grad_smooth_kernel": cfg.grad_smooth_kernel,
            "style_eval_long_side": cfg.style_eval_long_side,
        }
