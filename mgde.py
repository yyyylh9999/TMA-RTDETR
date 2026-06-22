import math

import torch
import torch.nn as nn

try:
    import torchvision
    from torchvision.ops import deform_conv2d
except Exception as exc:  # pragma: no cover - environment dependent
    torchvision = None
    deform_conv2d = None
    _DEFORM_IMPORT_ERROR = exc
else:
    _DEFORM_IMPORT_ERROR = None

from .layers import Conv, autopad


def check_deform_conv2d_available(device=None):
    """
    Validate that torchvision's DCNv2 operator is available.

    Call this once at training startup and save the printed versions/logs with
    the experiment. The MGDE implementation intentionally has no standard
    convolution fallback; if this check fails, the experiment is not using MGDE.
    """

    info = {
        "torch": torch.__version__,
        "torchvision": getattr(torchvision, "__version__", None),
        "cuda": torch.version.cuda,
        "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        "deform_conv2d": False,
    }
    if deform_conv2d is None:
        raise RuntimeError(
            "torchvision.ops.deform_conv2d is unavailable. Install a torchvision "
            "build that matches the active PyTorch/CUDA runtime before training MGDE."
        ) from _DEFORM_IMPORT_ERROR

    dev = torch.device(info["device"])
    x = torch.zeros(1, 1, 5, 5, device=dev)
    weight = torch.zeros(1, 1, 3, 3, device=dev)
    offset = torch.zeros(1, 18, 5, 5, device=dev)
    mask = torch.ones(1, 9, 5, 5, device=dev)
    _ = deform_conv2d(x, offset, weight, None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), mask=mask)
    info["deform_conv2d"] = True
    return info


class MPCA(nn.Module):
    """Multi-path coordinate attention for morphology-guided feature refinement."""

    def __init__(self, channels):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels),
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_g = self.gap(x)

        x_hw = torch.cat([x_h, x_w], dim=2)
        x_hw = self.conv_hw(x_hw)
        x_h, x_w = torch.split(x_hw, [h, w], dim=2)

        hw_weight = self.conv_pool_hw(x_hw).sigmoid()
        h_weight, w_weight = torch.split(hw_weight, [h, w], dim=2)
        x_h = x_h * h_weight
        x_w = x_w * w_weight
        x_g = x_g * torch.mean(hw_weight, dim=2, keepdim=True)

        return x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid() * x_g.sigmoid()


class MorphologyOffsetAttention(nn.Module):
    """Offset/mask predictor: X -> Conv_off(X) -> MPCA -> Split."""

    def __init__(self, in_channels, kernel_size=3, stride=1, deformable_groups=1):
        super().__init__()
        padding = autopad(kernel_size)
        out_channels = deformable_groups * 3 * kernel_size * kernel_size
        self.offset_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.attention = MPCA(out_channels)

    def forward(self, x):
        offset_mask = self.offset_mask(x)
        return self.attention(offset_mask)


class MorphologyGuidedDeformableConv(nn.Module):
    """
    Morphology-Guided Deformable Extraction.

    Execution order:
        X' -> Conv_off -> MPCA -> Split(offset_x, offset_y, mask)
        -> modulated DCNv2 -> BN -> activation.

    This module requires torchvision.ops.deform_conv2d. It deliberately does
    not fall back to a standard convolution, because such a fallback would make
    the MGDE ablation inconsistent with the paper.
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
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups for deform_conv2d.")

        if deform_conv2d is None:
            raise RuntimeError(
                "MGDE requires torchvision.ops.deform_conv2d, but it could not be imported. "
                "Fix the PyTorch/torchvision/CUDA installation before running MGDE."
            ) from _DEFORM_IMPORT_ERROR

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.offset_attention = MorphologyOffsetAttention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels // self.groups
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

        x = deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return self.act(self.bn(x))


class MGDEBlock(nn.Module):
    """Residual MGDE block: ConvNorm -> MPCA-guided DCNv2 -> Add -> ReLU."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.mgde = MorphologyGuidedDeformableConv(channels, channels, kernel_size, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.mgde(self.pre(x)))
