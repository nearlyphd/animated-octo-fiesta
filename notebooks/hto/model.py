"""U-Net landmark-detection architecture — the heatmap-regression baseline.

Shared by the training notebook and the OAI external-validation notebook so both build
the identical model and the same checkpoint (best_model_unet_global.pt) loads in either.
Input  : (B, 3, H, W) letterboxed radiograph.
Output : (B, num_keypoints, H/2, W/2) raw half-resolution heatmaps (linear head).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BN -> ReLU) x2 -- the standard U-Net convolutional block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 (MaxPool) then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscale by 2, concatenate the encoder skip, then DoubleConv.

    bilinear=False uses a learned ConvTranspose2d (halves channels); bilinear=True
    uses parameter-free bilinear upsampling and lets the following DoubleConv absorb
    the channels. Either way the output spatial size is padded to match the skip.
    """
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=False):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad to the skip's spatial size (safety net for non-power-of-two inputs)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNetKeypointHalfHeatmap(nn.Module):
    """Plain U-Net landmark detector -- the heatmap-regression baseline.

    Input  : (B, 3, TARGET_SIZE, TARGET_SIZE)                 letterboxed radiograph
    Output : (B, num_keypoints, TARGET_SIZE/2, TARGET_SIZE/2) raw heatmap predictions

    The decoder stops one stage short of full resolution, so the output stride is 2
    and the head emits a half-resolution heatmap (HEATMAP_SCALE = 0.5). No final
    activation is applied (linear head): the MSE loss is computed directly on the raw
    heatmap output, and coordinates are decoded with the inline extract_coordinates (argmax) helper.
    """
    def __init__(self, num_keypoints=12, base_ch=64, bilinear=False):
        super().__init__()
        c = base_ch
        # encoder
        self.inc   = DoubleConv(3, c)             # @ 1/1     C=c
        self.down1 = Down(c,      2 * c)          # @ 1/2     C=2c
        self.down2 = Down(2 * c,  4 * c)          # @ 1/4     C=4c
        self.down3 = Down(4 * c,  8 * c)          # @ 1/8     C=8c
        self.down4 = Down(8 * c, 16 * c)          # @ 1/16    C=16c  (bottleneck)
        # decoder (stops at 1/2 -> half-resolution heatmap)
        self.up1   = Up(16 * c, 8 * c, 8 * c, bilinear)   # 1/16 -> 1/8
        self.up2   = Up(8 * c,  4 * c, 4 * c, bilinear)   # 1/8  -> 1/4
        self.up3   = Up(4 * c,  2 * c, 2 * c, bilinear)   # 1/4  -> 1/2
        self.outc  = nn.Conv2d(2 * c, num_keypoints, kernel_size=1)  # half-res heatmaps

    def forward(self, x):
        x1 = self.inc(x)       # @ 1/1
        x2 = self.down1(x1)    # @ 1/2
        x3 = self.down2(x2)    # @ 1/4
        x4 = self.down3(x3)    # @ 1/8
        x5 = self.down4(x4)    # @ 1/16
        x  = self.up1(x5, x4)  # @ 1/8
        x  = self.up2(x,  x3)  # @ 1/4
        x  = self.up3(x,  x2)  # @ 1/2   (the 1/1 skip x1 is intentionally unused)
        return self.outc(x)    # (B, num_keypoints, H/2, W/2) raw heatmaps
