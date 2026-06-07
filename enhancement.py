# ============================================================
# 图像增强模块: 去噪 / 锐化 / 超分辨率
# 支持深度学习与传统方法
# ============================================================

import cv2
import numpy as np
import torch
from typing import Optional


class ImageEnhancement:
    """图像增强 — 深度学习 + 传统CV方法"""

    def __init__(self, model: Optional[torch.nn.Module] = None,
                 device: str = 'cpu'):
        self.model = model
        self.device = device
        if model is not None:
            self.model.to(device)
            self.model.eval()

    # ==================== 深度学习增强 ====================

    def enhance_dl(self, img: np.ndarray) -> np.ndarray:
        """使用深度学习模型增强图像"""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [0,255] -> [-1,1]
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x)
            out = torch.clamp(out, -1, 1)

        out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255
        return np.clip(out, 0, 255).astype(np.uint8)

    def load_model(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def load_checkpoint(self, path: str, device: str = 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.device = device
        self.model.to(device)
        self.model.eval()

    # ==================== 传统算法增强 ====================

    @staticmethod
    def denoise_nlm(img: np.ndarray, h: float = 10,
                    template_size: int = 7, search_size: int = 21) -> np.ndarray:
        """非局部均值去噪 (Non-Local Means)"""
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_size, search_size)

    @staticmethod
    def denoise_bilateral(img: np.ndarray, d: int = 9,
                          sigma_color: float = 75,
                          sigma_space: float = 75) -> np.ndarray:
        """双边滤波去噪 (保留边缘)"""
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    @staticmethod
    def denoise_median(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """中值滤波 (椒盐噪声)"""
        return cv2.medianBlur(img, kernel_size)

    @staticmethod
    def denoise_wavelet(img: np.ndarray, threshold: float = 30) -> np.ndarray:
        """小波阈值去噪"""
        import pywt
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            coeffs = pywt.wavedec2(img[:, :, c].astype(np.float32), 'db4', level=3)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = threshold * sigma / 255.0
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [
                tuple(pywt.threshold(d, thresh, mode='soft') for d in level)
                for level in coeffs[1:]
            ]
            result[:, :, c] = pywt.waverec2(coeffs_thresh, 'db4')[:img.shape[0], :img.shape[1]]
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def sharpen_unsharp_mask(img: np.ndarray, sigma: float = 1.0,
                             amount: float = 1.5) -> np.ndarray:
        """USM锐化 (Unsharp Masking)"""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def sharpen_laplacian(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """拉普拉斯锐化"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        laplacian = np.clip(laplacian, -255, 255).astype(np.int16)
        if img.ndim == 3:
            result = img.astype(np.int16) - laplacian[..., np.newaxis]
        else:
            result = img.astype(np.int16) - laplacian
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def super_resolution_bicubic(img: np.ndarray,
                                  scale_factor: float = 2.0) -> np.ndarray:
        """双三次插值超分辨率"""
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)),
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def super_resolution_lanczos(img: np.ndarray,
                                  scale_factor: float = 2.0) -> np.ndarray:
        """Lanczos插值超分辨率"""
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)),
                          interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def deblur_wiener(img: np.ndarray, kernel_size: int = 15,
                      angle: float = 30, snr: float = 50) -> np.ndarray:
        """维纳滤波去模糊"""
        from scipy.signal import convolve2d
        k = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        rad = np.deg2rad(angle)
        for i in range(kernel_size):
            x = int(center + (i - center) * np.cos(rad))
            y = int(center + (i - center) * np.sin(rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                k[y, x] = 1
        k /= k.sum()

        if img.ndim == 3:
            result = np.zeros_like(img)
            for c in range(3):
                result[:, :, c] = cv2.filter2D(img[:, :, c], -1, k)
            return result

        return cv2.filter2D(img, -1, k)

    @staticmethod
    def enhance_clahe(img: np.ndarray, clip_limit: float = 2.0,
                      tile_size: int = 8) -> np.ndarray:
        """CLAHE 对比度增强"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def enhance_histogram_equalization(img: np.ndarray) -> np.ndarray:
        """直方图均衡化"""
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # ==================== 组合增强流程 ====================

    @classmethod
    def enhance_pipeline(cls, img: np.ndarray, tasks: list) -> np.ndarray:
        """按顺序执行多个增强操作

        tasks: [{"op": "denoise_nlm", "kwargs": {...}}, ...]
        """
        op_map = {
            "denoise_nlm": cls.denoise_nlm,
            "denoise_bilateral": cls.denoise_bilateral,
            "denoise_median": cls.denoise_median,
            "denoise_wavelet": cls.denoise_wavelet,
            "sharpen_unsharp": cls.sharpen_unsharp_mask,
            "sharpen_laplacian": cls.sharpen_laplacian,
            "sr_bicubic": cls.super_resolution_bicubic,
            "sr_lanczos": cls.super_resolution_lanczos,
            "deblur_wiener": cls.deblur_wiener,
            "enhance_clahe": cls.enhance_clahe,
            "hist_eq": cls.enhance_histogram_equalization,
        }

        result = img.copy()
        for task in tasks:
            op = task["op"]
            kwargs = task.get("kwargs", {})
            if op in op_map:
                result = op_map[op](result, **kwargs)
        return result
