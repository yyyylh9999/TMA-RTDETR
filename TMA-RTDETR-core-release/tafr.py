import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv, DWConv


class LiteSepConv(nn.Module):
    """Lightweight point-wise projection followed by depth-wise refinement."""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.proj = Conv(c1, c2, 1)
        self.dw = DWConv(c2, c2, k)

    def forward(self, x):
        return self.dw(self.proj(x))


class TaskAwareRouter(nn.Module):
    """
    Task-Aware Feature Router.

    The router adaptively combines three branches:
    high-resolution detail, adjacent-scale semantics, and global context.
    Inputs are expected as [P2, P3, P4, P5].
    """

    def __init__(self, in_channels, out_channels, num_experts=3):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)):
            in_channels = [in_channels, in_channels, in_channels, in_channels]
        if len(in_channels) != 4:
            raise ValueError("TaskAwareRouter expects [P2, P3, P4, P5] channel sizes.")

        gate_in_ch = in_channels[1]
        hidden = max(gate_in_ch // 4, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
            nn.Softmax(dim=1),
        )

        self.expert_high_res = nn.Sequential(
            LiteSepConv(in_channels[0], out_channels, 3),
            DWConv(out_channels, out_channels, 3),
        )
        self.expert_multi_scale = LiteSepConv(in_channels[1] + in_channels[2], out_channels, 3)
        self.expert_context = nn.Sequential(
            Conv(in_channels[3], out_channels, 1),
            DWConv(out_channels, out_channels, 3),
        )
        self.output_conv = nn.Sequential(
            DWConv(out_channels, out_channels, 3),
            Conv(out_channels, out_channels, 1),
        )

    def forward(self, features):
        p2, p3, p4, p5 = features
        target_size = p3.shape[2:]
        weights = self.gate(p3)

        out_high = self.expert_high_res(F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False))
        out_multi = self.expert_multi_scale(
            torch.cat([p3, F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)], dim=1)
        )
        out_context = F.interpolate(self.expert_context(p5), size=target_size, mode="bilinear", align_corners=False)

        fused = (
            weights[:, 0:1] * out_high
            + weights[:, 1:2] * out_multi
            + weights[:, 2:3] * out_context
        )
        return self.output_conv(fused), weights


class TAFRModule(nn.Module):
    """
    TAFR wrapper with an optional lightweight P2 fusion branch.

    The router first produces the routed P3-level feature Y3. When return_y2 is
    enabled, Y3 is upsampled and fused with a projected P2 feature to produce
    the high-resolution Y2 branch described in the paper.
    """

    def __init__(self, in_channels, out_channels, return_y2=False, return_all=False):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)):
            in_channels = [in_channels, in_channels, in_channels, in_channels]
        if len(in_channels) != 4:
            raise ValueError("TAFRModule expects [P2, P3, P4, P5] channel sizes.")

        self.return_y2 = return_y2
        self.return_all = return_all
        self.router = TaskAwareRouter(in_channels, out_channels)
        self.p2_proj = Conv(in_channels[0], out_channels, 1)
        self.y2_fuse = LiteSepConv(out_channels * 2, out_channels, 3)

    def forward(self, features):
        p2 = features[0]
        y3, weights = self.router(features)
        y3_up = F.interpolate(y3, size=p2.shape[2:], mode="nearest")
        y2 = self.y2_fuse(torch.cat([y3_up, self.p2_proj(p2)], dim=1))

        if self.return_all:
            return {"y2": y2, "y3": y3, "weights": weights}
        if self.return_y2:
            return y2
        return y3
