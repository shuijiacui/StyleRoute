"""Landscape objective with fixed content masks and fixed normalisation scales."""

from dataclasses import dataclass

import numpy as np

from src.models.compensation import image_lab


class UniformityMetric:
    """Excess multiscale Lab variation, excluding original edges and region borders.

    Original band-pass magnitudes provide a local allowance for natural gradients.
    Uniform global colour offsets have zero penalty. The support NEVER depends on output.
    """

    def __init__(self, geometry, scales=(3.0, 8.0, 16.0)):
        self.geometry = geometry
        self.scales = tuple(scales)
        self.references = []
        for box, mask, weight in geometry.parts:
            source = geometry.lab[box]
            for sigma in self.scales:
                band = source - geometry.blur(source, mask, sigma)
                self.references.append((box, mask, weight, sigma, np.abs(band)))

    def __call__(self, image):
        lab = image_lab(image)
        total_l = total_c = mass = 0.0
        for box, mask, weight, sigma, original in self.references:
            candidate = lab[box]
            band = np.abs(candidate - self.geometry.blur(candidate, mask, sigma))
            excess = np.maximum(band - 1.25 * original, 0)
            total_l += float((excess[..., 0] * weight).sum())
            total_c += float((np.linalg.norm(excess[..., 1:], axis=2) * weight).sum())
            mass += float(weight.sum())
        lightness = total_l / max(mass, 1e-6)
        chroma = total_c / max(mass, 1e-6)
        return {"U": lightness + 0.5 * chroma, "U_L": lightness, "U_ab": chroma}


@dataclass(frozen=True)
class LandscapeConfig:
    # Fixed across candidates/pairs, never population min-max normalised.
    aesthetic_weight: float = 0.4
    uniformity_weight: float = 0.4
    structure_weight: float = 0.2
    aesthetic_scale: float = 1.0  # NIMA MOS units, calibrated on pilot only
    uniformity_scale: float = 3.0  # Lab units
    structure_scale: float = 1.0  # Q / (baseline Q + delta)
    minimum_improvement: float = 0.002

    def __post_init__(self):
        values = list(self.__dict__.values())
        if not all(np.isfinite(x) and x >= 0 for x in values):
            raise ValueError("Invalid landscape objective settings")
        if min(self.aesthetic_scale, self.uniformity_scale, self.structure_scale) <= 0:
            raise ValueError("Normalisation scales must be positive")
        if self.aesthetic_weight + self.uniformity_weight + self.structure_weight <= 0:
            raise ValueError("At least one objective weight must be positive")

    def objective(self, a, u, q, q0, delta):
        return (
            -self.aesthetic_weight * a / self.aesthetic_scale
            + self.uniformity_weight * u / self.uniformity_scale
            + self.structure_weight * q / (q0 + delta) / self.structure_scale
        )
