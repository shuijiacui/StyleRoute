"""Content-guided, bounded luminance compensation after the frozen decoder.

No semantic class is special-cased. All masks come from the original photograph.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, find_objects, gaussian_filter
from skimage.color import lab2rgb, rgb2lab


def image_lab(image):
    return rgb2lab(image.detach().float().cpu().numpy()[0].transpose(1, 2, 0))


class ContentGeometry:
    """Cache region crops, smoothness weights and original Lab once per pair."""

    def __init__(self, content, labels):
        self.lab = image_lab(content)
        if labels.shape != self.lab.shape[:2]:
            raise ValueError("Content/region size mismatch")
        mean = gaussian_filter(self.lab, (3, 3, 0))
        variance = np.maximum(gaussian_filter(self.lab**2, (3, 3, 0)) - mean**2, 0)
        std = np.sqrt(variance.sum(axis=2))
        self.smooth = np.clip((5.0 - std) / 3.0, 0, 1)
        self.parts = []
        for index, box in enumerate(find_objects(labels.astype(np.int32) + 1)):
            if box is None:
                continue
            mask = labels[box] == index
            # Padding makes distances correct even when a region touches the frame.
            distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
            weight = self.smooth[box] * np.clip((distance - 4) / 8, 0, 1)
            if weight.sum() < 16:
                continue
            self.parts.append((box, mask.astype(np.float32), weight))

    @staticmethod
    def blur(values, mask, sigma):
        norm = gaussian_filter(mask, sigma)
        if values.ndim == 3:
            return gaussian_filter(values * mask[..., None], (sigma, sigma, 0)) / np.maximum(
                norm[..., None], 1e-6
            )
        return gaussian_filter(values * mask, sigma) / np.maximum(norm, 1e-6)


class LocalCompensator:
    def __init__(self, geometry, reconstruction, limit=12.0):
        self.geometry = geometry
        self.limit = float(limit)
        residual = geometry.lab[..., 0] - image_lab(reconstruction)[..., 0]
        self.reference = []
        for box, mask, weight in geometry.parts:
            low = geometry.blur(residual[box], mask, 3)
            low -= np.median(low[weight > 0])
            self.reference.append(low)

    def __call__(self, image, beta):
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("Compensation beta must be in [0,1]")
        if beta == 0:
            return image
        lab = image_lab(image)
        delta = np.zeros(lab.shape[:2], dtype=np.float32)
        for (box, mask, weight), reference in zip(self.geometry.parts, self.reference):
            residual = self.geometry.lab[box][..., 0] - lab[box][..., 0]
            low = self.geometry.blur(residual, mask, 3)
            low -= np.median(low[weight > 0])
            same_direction = reference * low > 0
            correction = np.sign(reference) * np.minimum(
                np.minimum(np.abs(reference), np.abs(low)), self.limit
            )
            delta[box] += beta * correction * same_direction * weight
        lab[..., 0] = np.clip(lab[..., 0] + delta, 0, 100)
        rgb = lab2rgb(lab).astype(np.float32)
        original = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
        rgb[delta == 0] = original[delta == 0]  # avoid roundtrip drift outside correction
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(image.device)
