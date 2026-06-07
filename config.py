# ============================================================
# Configuration / 配置文件 — 低质量图像重建与评价系统
# Low-Quality Image Enhancement & Evaluation System
# ============================================================

import os

# ---- Language / 语言设置 ----
# 'zh' = Chinese (中文), 'en' = English
LANG = "zh"

# ---- Paths / 路径 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")          # Original HD images / 原始高清图像
DEGRADED_DIR = os.path.join(DATA_DIR, "degraded")    # Degraded images / 退化图像
ENHANCED_DIR = os.path.join(DATA_DIR, "enhanced")    # Enhanced results / 增强结果
RESULTS_DIR = os.path.join(BASE_DIR, "results")      # Evaluation reports / 评测报告
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [DATA_DIR, INPUT_DIR, DEGRADED_DIR, ENHANCED_DIR,
          RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Degradation Parameters / 退化参数 ----
DEGRADATION = {
    "gaussian_noise": {"mean": 0, "std": 25},        # Gaussian noise std / 高斯噪声标准差
    "salt_pepper_noise": {"amount": 0.02},            # Salt & pepper ratio / 椒盐噪声比例
    "poisson_noise": {"scale": 1.0},                  # Poisson noise / 泊松噪声
    "gaussian_blur": {"kernel_size": 7, "sigma": 1.5},# Gaussian blur / 高斯模糊
    "motion_blur": {"kernel_size": 15, "angle": 30},  # Motion blur / 运动模糊
    "jpeg_compression": {"quality": 20},              # JPEG quality [1-100] / JPEG压缩质量
    "downsample": {"scale": 0.5},                     # Downsample scale / 下采样尺度
}

# ---- Model Parameters / 模型参数 ----
MODEL = {
    "in_channels": 3,
    "out_channels": 3,
    "base_channels": 64,
    "num_blocks": 4,
    "num_scales": 3,
    "use_attention": True,
}

# ---- Training Parameters / 训练参数 ----
TRAIN = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_scheduler_step": 30,
    "lr_scheduler_gamma": 0.5,
    "patch_size": 128,
    "num_workers": 4,
    "save_interval": 10,
    "val_interval": 5,
}

# ---- Evaluation Parameters / 评测参数 ----
EVAL = {
    "metrics": ["psnr", "ssim", "lpips", "ms_ssim", "vif", "fsim"],
    "save_comparison": True,
    "report_format": "html",   # html / markdown / json
}
