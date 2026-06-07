# ============================================================
# 多尺度特征增强网络 (Multi-Scale Feature Enhancement Network)
# 架构: 编码器-解码器 + 残差块 + 通道注意力 + 多尺度聚合
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------- 基础模块 ---------------------
class ResidualBlock(nn.Module):
    """残差块 (带可选通道注意力)"""
    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.use_attention = use_attention
        if use_attention:
            self.attn = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        out = self.norm2(self.conv2(out))
        if self.use_attention:
            out = self.attn(out) * out
        return F.leaky_relu(out + residual, 0.2)


class ChannelAttention(nn.Module):
    """通道注意力 (Squeeze-and-Excitation)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return w


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x) * x
        x = self.spatial_attn(x) * x
        return x


# --------------------- 多尺度模块 ---------------------
class MultiScaleConv(nn.Module):
    """多尺度卷积: 不同感受野并行提取特征"""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid = out_c // 4
        self.conv1 = nn.Conv2d(in_c, mid, 1)
        self.conv3 = nn.Conv2d(in_c, mid, 3, padding=1)
        self.conv5 = nn.Conv2d(in_c, mid, 5, padding=2)
        self.conv7 = nn.Conv2d(in_c, out_c - 3 * mid, 7, padding=3)
        self.norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        return F.leaky_relu(self.norm(torch.cat([f1, f3, f5, f7], dim=1)), 0.2)


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_c: int, out_c: int, use_attention: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_c)
        self.res = ResidualBlock(out_c, use_attention)

    def forward(self, x):
        x = F.leaky_relu(self.norm(self.conv(x)), 0.2)
        return self.res(x)


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_c: int, skip_c: int, out_c: int,
                 use_attention: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_c + skip_c, out_c, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_c)
        self.res = ResidualBlock(out_c, use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = F.leaky_relu(self.norm(self.conv(x)), 0.2)
        return self.res(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — 多尺度上下文聚合"""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid = out_c // 5
        self.conv1 = nn.Conv2d(in_c, mid, 1)
        self.conv3_r6 = nn.Conv2d(in_c, mid, 3, padding=6, dilation=6)
        self.conv3_r12 = nn.Conv2d(in_c, mid, 3, padding=12, dilation=12)
        self.conv3_r18 = nn.Conv2d(in_c, mid, 3, padding=18, dilation=18)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c - 4 * mid, 1)
        )
        self.norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv3_r6(x)
        f3 = self.conv3_r12(x)
        f4 = self.conv3_r18(x)
        f5 = F.interpolate(self.pool(x), size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        return F.leaky_relu(self.norm(torch.cat([f1, f2, f3, f4, f5], dim=1)), 0.2)


# --------------------- 主干网络 ---------------------
class MSFENet(nn.Module):
    """Multi-Scale Feature Enhancement Network
    用于联合去噪 / 去模糊 / 超分辨率
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, num_blocks: int = 4,
                 num_scales: int = 3, use_attention: bool = True):
        super().__init__()
        self.num_scales = num_scales

        # 输入层
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 多尺度特征提取
        self.ms_conv = MultiScaleConv(base_channels, base_channels)

        # 编码器
        self.encoders = nn.ModuleList()
        ch = base_channels
        for i in range(num_scales):
            self.encoders.append(
                DownBlock(ch, ch * 2, use_attention)
            )
            ch *= 2

        # 瓶颈层 (ASPP多尺度上下文)
        self.bottleneck = nn.Sequential(
            ResidualBlock(ch, use_attention),
            ASPP(ch, ch),
            ResidualBlock(ch, use_attention),
        )

        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(num_scales):
            skip_c = ch // 2
            self.decoders.append(
                UpBlock(ch, skip_c, skip_c, use_attention)
            )
            ch //= 2

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存跳跃连接
        skips = []
        feat = F.leaky_relu(self.input_conv(x), 0.2)
        feat = self.ms_conv(feat)

        for encoder in self.encoders:
            skips.append(feat)
            feat = encoder(feat)

        feat = self.bottleneck(feat)

        for decoder in self.decoders:
            skip = skips.pop()
            feat = decoder(feat, skip)

        # 残差输出: 学习残差而非直接预测
        residual = self.output_conv(feat)
        return torch.clamp(x + residual, -1, 1)


# --------------------- 轻量版 (快速推理) ---------------------
class MSFENetLite(nn.Module):
    """轻量版多尺度增强网络"""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 32):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.enc1 = DownBlock(base_channels, base_channels * 2, True)
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4, True)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4, True),
            ResidualBlock(base_channels * 4, True),
        )

        self.dec2 = UpBlock(base_channels * 4, base_channels * 2,
                            base_channels * 2, True)
        self.dec1 = UpBlock(base_channels * 2, base_channels,
                            base_channels, True)

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        f0 = F.leaky_relu(self.input_conv(x), 0.2)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f2 = self.bottleneck(f2)
        f1 = self.dec2(f2, f1)
        f0 = self.dec1(f1, f0)
        residual = self.output_conv(f0)
        return torch.clamp(x + residual, -1, 1)
