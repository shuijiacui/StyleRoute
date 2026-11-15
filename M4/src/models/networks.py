"""Inference-only topology adapted from naoto0804/pytorch-AdaIN, v0.0.0 (MIT).

Original files and license are retained under resources/upstream/adain. Sequential
indices intentionally match the original checkpoint keys. Full VGG loads before
truncation to relu4_1; no classification VGG replacement is used.
"""

from torch import nn


def build_encoder() -> nn.Sequential:
    layers = [nn.Conv2d(3, 3, 1)]
    in_channels = 3
    for block, (channels, count) in enumerate(zip((64, 128, 256, 512, 512), (2, 2, 4, 4, 4))):
        for _ in range(count):
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, channels, 3), nn.ReLU()]
            in_channels = channels
        if block < 4:
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
    return nn.Sequential(*layers)


def build_decoder() -> nn.Sequential:
    layers = []
    channels = (512, 256, 256, 256, 256, 128, 128, 64, 64, 3)
    for index, (before, after) in enumerate(zip(channels, channels[1:])):
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(before, after, 3)]
        if index < 8:
            layers.append(nn.ReLU())
        if index in (0, 4, 6):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
    return nn.Sequential(*layers)
