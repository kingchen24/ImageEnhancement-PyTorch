# ============================================================
# 图像退化模型: 噪声 / 模糊 / 压缩 / 下采样
# ============================================================

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional


class ImageDegradation:
    """图像退化模型，模拟真实场景中的图像质量下降"""

    # ----------------------------------------------------------------
    # 噪声退化
    # ----------------------------------------------------------------
    @staticmethod
    def add_gaussian_noise(img: np.ndarray, mean: float = 0,
                           std: float = 25) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(mean, std, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
        """添加椒盐噪声"""
        out = img.copy()
        h, w = img.shape[:2]
        num_salt = int(h * w * amount / 2)
        num_pepper = int(h * w * amount / 2)

        for c in range(3) if img.ndim == 3 else [None]:
            coords_salt = (np.random.randint(0, h, num_salt),
                           np.random.randint(0, w, num_salt))
            coords_pepper = (np.random.randint(0, h, num_pepper),
                             np.random.randint(0, w, num_pepper))
            if img.ndim == 3:
                out[coords_salt[0], coords_salt[1], c] = 255
                out[coords_pepper[0], coords_pepper[1], c] = 0
            else:
                out[coords_salt] = 255
                out[coords_pepper] = 0
        return out

    @staticmethod
    def add_poisson_noise(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """添加泊松噪声（模拟光子计数噪声）"""
        img_f = img.astype(np.float32) / 255.0 * scale * 255
        noisy = np.random.poisson(np.maximum(img_f, 0)).astype(np.float32)
        noisy = noisy / scale / 255 * 255
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------
    # 模糊退化
    # ----------------------------------------------------------------
    @staticmethod
    def add_gaussian_blur(img: np.ndarray, kernel_size: int = 7,
                          sigma: float = 1.5) -> np.ndarray:
        """高斯模糊"""
        return gaussian_filter(img, sigma=(sigma, sigma, 0)
                               if img.ndim == 3 else sigma).astype(np.uint8)

    @staticmethod
    def add_motion_blur(img: np.ndarray, kernel_size: int = 15,
                        angle: float = 30) -> np.ndarray:
        """运动模糊"""
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
    def add_defocus_blur(img: np.ndarray, radius: int = 5) -> np.ndarray:
        """散焦模糊（圆盘模糊）"""
        kernel_size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
        disk /= disk.sum()

        if img.ndim == 3:
            result = np.zeros_like(img)
            for c in range(3):
                result[:, :, c] = cv2.filter2D(img[:, :, c], -1, disk)
            return result
        return cv2.filter2D(img, -1, disk)

    # ----------------------------------------------------------------
    # 压缩退化
    # ----------------------------------------------------------------
    @staticmethod
    def add_jpeg_compression(img: np.ndarray, quality: int = 20) -> np.ndarray:
        """JPEG 压缩失真"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    @staticmethod
    def add_webp_compression(img: np.ndarray, quality: int = 20) -> np.ndarray:
        """WebP 压缩失真"""
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        _, enc = cv2.imencode('.webp', img, encode_param)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # ----------------------------------------------------------------
    # 分辨率退化
    # ----------------------------------------------------------------
    @staticmethod
    def downsample(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """下采样"""
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    # ----------------------------------------------------------------
    # 组合退化
    # ----------------------------------------------------------------
    @classmethod
    def apply_degradation_pipeline(cls, img: np.ndarray,
                                   config: dict) -> Tuple[np.ndarray, dict]:
        """按配置组合应用多种退化"""
        meta = {}
        degraded = img.copy()

        if config.get("gaussian_noise", {}).get("enabled", False):
            cfg = config["gaussian_noise"]
            degraded = cls.add_gaussian_noise(degraded, cfg.get("mean", 0),
                                              cfg.get("std", 25))
            meta["noise"] = "gaussian"

        if config.get("salt_pepper_noise", {}).get("enabled", False):
            degraded = cls.add_salt_pepper_noise(
                degraded, config["salt_pepper_noise"].get("amount", 0.02))
            meta["noise"] = meta.get("noise", "") + "+salt_pepper"

        if config.get("poisson_noise", {}).get("enabled", False):
            degraded = cls.add_poisson_noise(
                degraded, config["poisson_noise"].get("scale", 1.0))
            meta["noise"] = meta.get("noise", "") + "+poisson"

        if config.get("gaussian_blur", {}).get("enabled", False):
            cfg = config["gaussian_blur"]
            degraded = cls.add_gaussian_blur(degraded, cfg.get("kernel_size", 7),
                                             cfg.get("sigma", 1.5))
            meta["blur"] = "gaussian"

        if config.get("motion_blur", {}).get("enabled", False):
            cfg = config["motion_blur"]
            degraded = cls.add_motion_blur(degraded, cfg.get("kernel_size", 15),
                                           cfg.get("angle", 30))
            meta["blur"] = meta.get("blur", "") + "+motion"

        if config.get("jpeg_compression", {}).get("enabled", False):
            degraded = cls.add_jpeg_compression(
                degraded, config["jpeg_compression"].get("quality", 20))
            meta["compression"] = "jpeg"

        if config.get("downsample", {}).get("enabled", False):
            degraded = cls.downsample(
                degraded, config["downsample"].get("scale", 0.5))
            meta["resolution"] = "downsampled"

        return degraded, meta
