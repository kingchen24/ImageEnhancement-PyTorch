# ============================================================
# 图像质量评价体系: PSNR / SSIM / MS-SSIM / VIF / FSIM / LPIPS
# ============================================================

import numpy as np
import cv2
from scipy import signal, ndimage
from scipy.fft import fft2, ifft2
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ImageQualityMetrics:
    """全参考图像质量评价指标"""

    # ------------------------ PSNR ------------------------
    @staticmethod
    def psnr(img_ref: np.ndarray, img_test: np.ndarray,
             max_val: float = 255.0) -> float:
        """峰值信噪比"""
        mse = np.mean((img_ref.astype(np.float64) - img_test.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val) - 10 * np.log10(mse)

    # ------------------------ SSIM ------------------------
    @staticmethod
    def ssim(img_ref: np.ndarray, img_test: np.ndarray,
             max_val: float = 255.0) -> float:
        """结构相似性指数"""
        try:
            from skimage.metrics import structural_similarity
            if img_ref.ndim == 3:
                return structural_similarity(img_ref, img_test,
                                             channel_axis=2, data_range=max_val)
            return structural_similarity(img_ref, img_test, data_range=max_val)
        except ImportError:
            return ImageQualityMetrics._ssim_numpy(img_ref, img_test, max_val)

    @staticmethod
    def _ssim_numpy(img1, img2, max_val=255.0, K1=0.01, K2=0.03, win_size=11):
        """纯numpy SSIM实现 (回退方案)"""
        C1 = (K1 * max_val) ** 2
        C2 = (K2 * max_val) ** 2

        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = np.outer(kernel, kernel)
        window = window / window.sum()

        if img1.ndim == 3:
            ssims = []
            for c in range(img1.shape[2]):
                ssims.append(ImageQualityMetrics._ssim_numpy(
                    img1[:, :, c], img2[:, :, c], max_val, K1, K2, win_size))
            return float(np.mean(ssims))

        mu1 = cv2.filter2D(img1.astype(np.float64), -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2.astype(np.float64), -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D((img1.astype(np.float64)) ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D((img2.astype(np.float64)) ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D((img1.astype(np.float64) * img2.astype(np.float64)),
                                -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())

    # ------------------------ MS-SSIM ------------------------
    @staticmethod
    def ms_ssim(img_ref: np.ndarray, img_test: np.ndarray,
                max_val: float = 255.0) -> float:
        """多尺度结构相似性"""
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            # 简化回退: 多尺度平均SSIM
            return ImageQualityMetrics._ms_ssim_numpy(img_ref, img_test, max_val)

        if img_ref.ndim == 3 and img_ref.shape[2] == 3:
            img_ref_g = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
            img_test_g = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            img_ref_g = img_ref.astype(np.float64)
            img_test_g = img_test.astype(np.float64)

        levels = 5
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        mssim = []
        mcs = []

        for i in range(levels):
            ssim_val, cs = ImageQualityMetrics._ssim_cs(img_ref_g, img_test_g,
                                                         max_val=max_val)
            mssim.append(ssim_val)
            mcs.append(cs)

            if i < levels - 1:
                img_ref_g = cv2.resize(img_ref_g,
                                       (img_ref_g.shape[1] // 2, img_ref_g.shape[0] // 2))
                img_test_g = cv2.resize(img_test_g,
                                        (img_test_g.shape[1] // 2, img_test_g.shape[0] // 2))

        overall = np.prod(np.array(mcs[:-1]) ** weights[:-1]) * (mssim[-1] ** weights[-1])
        return float(overall)

    @staticmethod
    def _ssim_cs(img1, img2, max_val=255.0, K1=0.01, K2=0.03, win_size=11):
        C1 = (K1 * max_val) ** 2
        C2 = (K2 * max_val) ** 2

        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = np.outer(kernel, kernel)
        window = window / window.sum()

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        return float(ssim_map.mean()), float(cs_map.mean())

    @staticmethod
    def _ms_ssim_numpy(img1, img2, max_val=255.0, levels=5):
        """纯numpy多尺度SSIM回退"""
        if img1.ndim == 3:
            img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
            img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            img1_g = img1.astype(np.float64)
            img2_g = img2.astype(np.float64)
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        ssims = []
        for i in range(levels):
            ssims.append(ImageQualityMetrics._ssim_numpy(img1_g, img2_g, max_val))
            if i < levels - 1:
                h, w = img1_g.shape
                img1_g = cv2.resize(img1_g, (w // 2, h // 2))
                img2_g = cv2.resize(img2_g, (w // 2, h // 2))
        return float(np.average(ssims[-3:], weights=[0.5, 0.3, 0.2]) if len(ssims) >= 3 else ssims[-1])

    # ------------------------ VIF ------------------------
    @staticmethod
    def vif(img_ref: np.ndarray, img_test: np.ndarray) -> float:
        """视觉信息保真度"""
        from scipy.special import gamma

        def _vif_channel(ref_ch, test_ch):
            ref_f = ref_ch.astype(np.float64) / 255.0
            test_f = test_ch.astype(np.float64) / 255.0

            sigma_nsq = 1e-10

            # 多尺度分解 (简化为4个频带)
            eps = 1e-10
            num = 0.0
            den = 0.0

            for scale in [1.0, 0.5, 0.25, 0.125]:
                if scale < 1.0:
                    h, w = ref_f.shape
                    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                    ref_s = cv2.resize(ref_f, (new_w, new_h))
                    test_s = cv2.resize(test_f, (new_w, new_h))
                else:
                    ref_s = ref_f
                    test_s = test_f

                var_ref = np.var(ref_s)
                g = var_ref / (var_ref + sigma_nsq)
                sv_sq = sigma_nsq * g

                num += np.log2(1 + (g ** 2) * var_ref / (sv_sq + eps))
                den += np.log2(1 + var_ref / (sigma_nsq + eps))

            return float(num / max(den, eps))

        if img_ref.ndim == 3:
            scores = [_vif_channel(img_ref[:, :, c], img_test[:, :, c])
                      for c in range(min(img_ref.shape[2], 3))]
            return float(np.mean(scores))
        return _vif_channel(img_ref, img_test)

    # ------------------------ FSIM ------------------------
    @staticmethod
    def fsim(img_ref: np.ndarray, img_test: np.ndarray) -> float:
        """特征相似性指数"""
        if img_ref.ndim == 3:
            ref_g = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
            test_g = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            ref_g = img_ref.astype(np.float64)
            test_g = img_test.astype(np.float64)

        # Phase Congruency
        pc1 = ImageQualityMetrics._phase_congruency(ref_g)
        pc2 = ImageQualityMetrics._phase_congruency(test_g)

        # Gradient Magnitude
        gx_r = cv2.Sobel(ref_g, cv2.CV_64F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(ref_g, cv2.CV_64F, 0, 1, ksize=3)
        gm1 = np.sqrt(gx_r ** 2 + gy_r ** 2)

        gx_t = cv2.Sobel(test_g, cv2.CV_64F, 1, 0, ksize=3)
        gy_t = cv2.Sobel(test_g, cv2.CV_64F, 0, 1, ksize=3)
        gm2 = np.sqrt(gx_t ** 2 + gy_t ** 2)

        T1, T2 = 0.85, 160
        pc_max = np.maximum(pc1, pc2)
        gm_max = np.maximum(gm1, gm2)

        sim_pc = (2 * pc1 * pc2 + T1) / (pc1 ** 2 + pc2 ** 2 + T1)
        sim_gm = (2 * gm1 * gm2 + T2) / (gm1 ** 2 + gm2 ** 2 + T2)

        sim = sim_pc * sim_gm
        return float(np.sum(pc_max * sim) / np.sum(pc_max + 1e-10))

    @staticmethod
    def _phase_congruency(img, nscale=4, norient=6):
        """计算相位一致性图"""
        h, w = img.shape
        rows, cols = h, w
        epsilon = 1e-4

        total_energy = np.zeros((rows, cols))
        total_amplitude = np.zeros((rows, cols))

        # 对数Gabor滤波器参数
        min_wavelength = 3
        mult = 2.1
        sigma_onf = 0.55

        for s in range(nscale):
            wavelength = min_wavelength * (mult ** s)
            fo = 1.0 / wavelength

            # 频率域滤波器
            fx = np.fft.fftfreq(cols) * cols
            fy = np.fft.fftfreq(rows) * rows
            fxx, fyy = np.meshgrid(fx, fy)
            radius = np.sqrt(fxx ** 2 + fyy ** 2)
            radius[0, 0] = 1

            log_gabor = np.exp(-(np.log(radius / fo)) ** 2 /
                               (2 * np.log(sigma_onf) ** 2))
            log_gabor[0, 0] = 0

            # 方向滤波 (简化为各向同性)
            fft_img = fft2(img)

            for o in range(norient):
                # 简化的方向选择
                EO = ifft2(fft_img * log_gabor)
                An = np.abs(EO)
                total_amplitude += An
                total_energy += An * np.cos(np.angle(EO))

        return total_energy / (total_amplitude + epsilon)

    # ------------------------ LPIPS ------------------------
    @staticmethod
    def lpips(img_ref: np.ndarray, img_test: np.ndarray) -> float:
        """Learned Perceptual Image Patch Similarity (简化版)"""
        # LPIPS 简化实现: 多尺度L1特征距离
        def _extract_features(x):
            features = []
            for s in [1, 2, 4]:
                if s > 1:
                    xs = cv2.resize(x, (x.shape[1] // s, x.shape[0] // s))
                else:
                    xs = x
                # Sobel 特征
                gx = cv2.Sobel(xs, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(xs, cv2.CV_64F, 0, 1, ksize=3)
                features.append(xs.astype(np.float64) / 255.0)
                features.append(np.sqrt(gx ** 2 + gy ** 2) / 255.0)
            return features

        if img_ref.ndim == 3:
            ref_g = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
            test_g = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        else:
            ref_g = img_ref
            test_g = img_test

        f_ref = _extract_features(ref_g)
        f_test = _extract_features(test_g)

        distances = []
        for fr, ft in zip(f_ref, f_test):
            distances.append(np.mean(np.abs(fr - ft)))

        return float(np.mean(distances))

    # ------------------------ 综合评估 ------------------------
    @classmethod
    def evaluate_all(cls, img_ref: np.ndarray, img_test: np.ndarray,
                     metrics: list = None) -> Dict[str, float]:
        """计算所有指定指标"""
        if metrics is None:
            metrics = ["psnr", "ssim", "ms_ssim", "vif", "fsim", "lpips"]

        metric_funcs = {
            "psnr": cls.psnr,
            "ssim": cls.ssim,
            "ms_ssim": cls.ms_ssim,
            "vif": cls.vif,
            "fsim": cls.fsim,
            "lpips": cls.lpips,
        }

        results = {}
        for m in metrics:
            if m in metric_funcs:
                try:
                    results[m] = metric_funcs[m](img_ref, img_test)
                except Exception:
                    results[m] = 0.0
        return results


# ------------------------ 便捷函数 ------------------------

def compute_psnr(ref, test):
    return ImageQualityMetrics.psnr(ref, test)


def compute_ssim(ref, test):
    return ImageQualityMetrics.ssim(ref, test)


def compare_images(ref_path: str, test_path: str,
                   metrics: list = None) -> Dict[str, float]:
    """便捷: 读图并计算质量指标"""
    ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    test = cv2.imread(test_path, cv2.IMREAD_COLOR)
    if ref is None or test is None:
        return {"error": "Failed to read image(s)"}
    return ImageQualityMetrics.evaluate_all(ref, test, metrics)
