import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv, autopad


class MorphologyOffsetAttention(nn.Module):
    """Coordinate-aware offset and mask predictor used by MGDE."""

    def __init__(self, in_channels, kernel_size=3, stride=1, deformable_groups=1):
        super().__init__()
        padding = autopad(kernel_size)
        out_channels = deformable_groups * 3 * kernel_size * kernel_size
        self.offset_mask = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)
        hidden = max(out_channels // 4, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _coords_like(x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return coords

    def forward(self, x):
        morphology_context = torch.cat([x, self._coords_like(x)], dim=1)
        offset_mask = self.offset_mask(morphology_context)
        return offset_mask * self.attention(offset_mask)


class MorphologyGuidedDeformableConv(nn.Module):
    """
    Morphology-Guided Deformable Extraction.

    This module uses coordinate-aware offset prediction and torchvision's
    deform_conv2d operator. A standard convolution fallback is used when the
    operator is unavailable.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        deformable_groups=1,
        act=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        pad = autopad(kernel_size, padding, dilation)
        self.padding = (pad, pad)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.offset_attention = MorphologyOffsetAttention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.offset_attention.offset_mask.weight.data.zero_()
        self.offset_attention.offset_mask.bias.data.zero_()

    def forward(self, x):
        offset_mask = self.offset_attention(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        if hasattr(torch.ops, "torchvision") and hasattr(torch.ops.torchvision, "deform_conv2d"):
            x = torch.ops.torchvision.deform_conv2d(
                x,
                self.weight,
                offset,
                mask,
                self.bias,
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.dilation[0],
                self.dilation[1],
                self.groups,
                self.deformable_groups,
                True,
            )
        else:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.act(self.bn(x))


class MGDEBlock(nn.Module):
    """A compact residual MGDE block for feature-stage insertion."""

    def __init__(self, channels, kernel_size=3, gamma_init=0.1):
        super().__init__()
        self.mgde = MorphologyGuidedDeformableConv(channels, channels, kernel_size, act=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.gamma * self.mgde(x))
