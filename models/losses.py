# ============================================================
# 损失函数: 多任务组合损失
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss — L1的平滑近似, 对抗异常值更鲁棒"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class PerceptualLoss(nn.Module):
    """感知损失 (VGG19-based)"""
    def __init__(self, layers: tuple = (4, 9, 18), device='cpu'):
        super().__init__()
        from torchvision import models
        vgg = models.vgg19(pretrained=True).features[:max(layers) + 1].to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.layers = layers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred * self.std) + self.mean
        target = (target * self.std) + self.mean
        loss = 0.0
        x = pred
        y = target
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += F.l1_loss(x, y)
        return loss


class GradientLoss(nn.Module):
    """梯度损失 — 保持边缘锐度"""
    def forward(self, pred, target):
        def gradient(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy
        dp_x, dp_y = gradient(pred)
        dt_x, dt_y = gradient(target)
        return F.l1_loss(dp_x, dt_x) + F.l1_loss(dp_y, dt_y)


class SSIMLoss(nn.Module):
    """SSIM Loss (1 - SSIM)"""
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1

    def _gaussian(self, window_size, sigma):
        import math
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1d = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size)

    def _ssim(self, img1, img2, window, C1=0.01**2, C2=0.03**2):
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=img2.shape[1])
        s1 = mu1 ** 2
        s2 = mu2 ** 2
        s12 = mu1 * mu2
        v1 = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=img1.shape[1]) - s1
        v2 = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=img2.shape[1]) - s2
        v12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=img2.shape[1]) - s12
        ssim_map = ((2 * s12 + C1) * (2 * v12 + C2)) / ((s1 + s2 + C1) * (v1 + v2 + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        _, c, _, _ = pred.shape
        window = self._create_window(self.window_size, c).to(pred.device)
        return 1 - self._ssim(pred, target, window)


class CompositeLoss(nn.Module):
    """组合损失: L1 + SSIM + 感知 + 梯度"""
    def __init__(self, weights: dict = None, use_perceptual: bool = True,
                 device='cpu'):
        super().__init__()
        default_weights = {"l1": 1.0, "ssim": 0.5, "perceptual": 0.1, "gradient": 0.05}
        self.weights = weights or default_weights
        self.l1 = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss(device=device)

    def forward(self, pred, target):
        total = 0.0
        losses = {}

        l1 = self.weights.get("l1", 1.0) * self.l1(pred, target)
        losses["l1"] = l1.item()
        total += l1

        ssim_l = self.weights.get("ssim", 0.5) * self.ssim_loss(pred, target)
        losses["ssim"] = ssim_l.item()
        total += ssim_l

        grad_l = self.weights.get("gradient", 0.05) * self.gradient_loss(pred, target)
        losses["gradient"] = grad_l.item()
        total += grad_l

        if self.use_perceptual:
            perc_l = self.weights.get("perceptual", 0.1) * self.perceptual(pred, target)
            losses["perceptual"] = perc_l.item()
            total += perc_l

        return total, losses
