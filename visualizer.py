# ============================================================
# Visualization Module / 可视化模块
# High-quality charts & comparisons for image enhancement
# 高质量的图像增强对比图表
# ============================================================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import List, Dict, Optional, Tuple

from config import RESULTS_DIR, LANG  # LANG: 'zh' or 'en'


# ==================== Bilingual Labels / 双语标签 ====================

class L:
    """Lightweight i18n helper / 轻量级双语助手"""

    _DICT = {
        # Chart titles / 图表标题
        "metrics_comparison": {
            "zh": "图像质量指标: 退化 vs 增强",
            "en": "Image Quality Metrics: Degraded vs Enhanced"
        },
        "training_loss": {
            "zh": "训练损失曲线",
            "en": "Training Loss Curve"
        },
        "training_psnr": {
            "zh": "训练 PSNR 曲线",
            "en": "Training PSNR Curve"
        },
        "image_grid": {
            "zh": "增强效果对比",
            "en": "Enhancement Comparison"
        },
        "radar_chart": {
            "zh": "多维指标雷达图",
            "en": "Multi-Metric Radar Chart"
        },
        "gain_bar": {
            "zh": "各项指标提升幅度",
            "en": "Metric Improvement Gains"
        },
        "per_image": {
            "zh": "逐图 PSNR/SSIM 对比",
            "en": "Per-Image PSNR/SSIM Comparison"
        },
        "time_distribution": {
            "zh": "处理时间分布",
            "en": "Processing Time Distribution"
        },
        # Axis labels / 坐标轴
        "score": {"zh": "得分", "en": "Score"},
        "epoch": {"zh": "训练轮次", "en": "Epoch"},
        "loss": {"zh": "损失值", "en": "Loss"},
        "psnr_db": {"zh": "PSNR (dB)", "en": "PSNR (dB)"},
        "ssim": {"zh": "SSIM", "en": "SSIM"},
        "image_index": {"zh": "图像序号", "en": "Image Index"},
        "processing_time_s": {"zh": "处理时间 (秒)", "en": "Processing Time (s)"},
        "gain_percent": {"zh": "提升幅度 (%)", "en": "Improvement (%)"},
        # Legend / 图例
        "degraded": {"zh": "退化图像", "en": "Degraded"},
        "enhanced": {"zh": "增强图像", "en": "Enhanced"},
        "train_loss": {"zh": "训练损失", "en": "Train Loss"},
        "val_loss": {"zh": "验证损失", "en": "Val Loss"},
        "train_psnr": {"zh": "训练 PSNR", "en": "Train PSNR"},
        "val_psnr": {"zh": "验证 PSNR", "en": "Val PSNR"},
        "psnr_gain": {"zh": "PSNR 提升", "en": "PSNR Gain"},
        "ssim_gain": {"zh": "SSIM 提升", "en": "SSIM Gain"},
        "original": {"zh": "原始图像", "en": "Original"},
        "input": {"zh": "输入图像", "en": "Input"},
        # Misc
        "total_images": {"zh": "图像总数", "en": "Total Images"},
        "avg_time": {"zh": "平均耗时", "en": "Avg Time"},
        "improvement": {"zh": "改善", "en": "Improvement"},
        "before": {"zh": "增强前", "en": "Before"},
        "after": {"zh": "增强后", "en": "After"},
    }

    @classmethod
    def t(cls, key: str) -> str:
        return cls._DICT.get(key, {}).get(LANG, key)


# ==================== Modern Color Palette / 现代配色 ====================

class Palette:
    """Curated color palette / 精选配色方案"""

    # Primary / 主色
    DEEP_BLUE    = "#1a73e8"
    VIVID_GREEN  = "#0d9488"
    CORAL_RED    = "#e74c3c"
    AMBER        = "#f59e0b"
    PURPLE       = "#8b5cf6"
    ROSE         = "#f43f5e"
    SKY          = "#0ea5e9"
    LIME         = "#84cc16"

    # Gradients / 渐变色组
    BLUE_GRADIENT = ["#667eea", "#764ba2"]
    GREEN_GRADIENT = ["#11998e", "#38ef7d"]
    WARM_GRADIENT = ["#f093fb", "#f5576c"]
    SKY_GRADIENT = ["#4facfe", "#00f2fe"]

    # Category palettes / 分类色板
    CATEGORY_10 = ["#1a73e8", "#e74c3c", "#0d9488", "#f59e0b",
                   "#8b5cf6", "#0ea5e9", "#f43f5e", "#84cc16",
                   "#ec4899", "#6366f1"]

    # Light backgrounds
    BG_LIGHT = "#f8fafc"
    GRID_COLOR = "#e2e8f0"

    @classmethod
    def gradient_cmap(cls, colors: List[str]) -> plt.matplotlib.colors.LinearSegmentedColormap:
        """Create a gradient colormap from hex colors"""
        return plt.matplotlib.colors.LinearSegmentedColormap.from_list("grad", colors)


# ==================== Global matplotlib style / 全局样式 ====================

def _set_style():
    """Apply modern matplotlib style / 应用现代 matplotlib 样式"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.facecolor": Palette.BG_LIGHT,
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.color": Palette.GRID_COLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#e2e8f0",
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
    })


_set_style()


# ==================== Helper / 辅助函数 ====================

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_fig(fig, save_path: str, dpi: int = 200) -> str:
    """Save figure with tight layout and high DPI"""
    _ensure_dir(save_path)
    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return save_path


# ==================== Main Visualizer / 主可视化器 ====================

class Visualizer:
    """Advanced visualization for image enhancement results
    图像增强结果的高级可视化"""

    # ----------------------------------------------------------------
    # Chart 1: Metrics Bar Comparison / 指标柱状图对比
    # ----------------------------------------------------------------
    @staticmethod
    def plot_metrics_comparison(metrics_before: Dict[str, float],
                                metrics_after: Dict[str, float],
                                save_path: str = None,
                                title: str = None) -> str:
        """Dual bar chart: degraded vs enhanced per metric"""
        keys = [k for k in metrics_before if k in metrics_after]
        if not keys:
            return ""

        x = np.arange(len(keys))
        w = 0.35

        fig, ax = plt.subplots(figsize=(11, 6.5))

        colors_degraded = [Palette.CORAL_RED, Palette.ROSE, Palette.AMBER]
        colors_enhanced = [Palette.VIVID_GREEN, Palette.DEEP_BLUE, Palette.PURPLE]

        bars1 = ax.bar(x - w / 2, [metrics_before[k] for k in keys],
                       w, label=L.t("degraded"),
                       color=colors_degraded[:len(keys)], alpha=0.88,
                       edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + w / 2, [metrics_after[k] for k in keys],
                       w, label=L.t("enhanced"),
                       color=colors_enhanced[:len(keys)], alpha=0.88,
                       edgecolor='white', linewidth=0.8)

        ax.set_ylabel(L.t("score"), fontweight='bold')
        ax.set_title(title or L.t("metrics_comparison"), fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([k.upper() for k in keys], fontweight='bold')
        ax.legend(loc='upper left', frameon=True)

        # Value labels on bars / 柱上数值标注
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 4), textcoords="offset points",
                                ha='center', fontsize=8.5, fontweight='bold',
                                color='#334155')

        ax.set_ylim(0, max(max(metrics_before.values()), max(metrics_after.values())) * 1.18)

        save_path = save_path or os.path.join(RESULTS_DIR, "metrics_comparison.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 2: Radar Chart / 雷达图
    # ----------------------------------------------------------------
    @staticmethod
    def plot_radar_chart(metrics_before: Dict[str, float],
                         metrics_after: Dict[str, float],
                         save_path: str = None) -> str:
        """Multi-dimensional radar chart for holistic comparison"""
        keys = [k for k in metrics_before if k in metrics_after and k != "lpips"]
        if len(keys) < 3:
            return ""

        # Normalize each metric to [0, 1] / 归一化到 [0,1]
        norm_before = {}
        norm_after = {}
        for k in keys:
            vmax = max(metrics_before[k], metrics_after[k], 1e-6)
            norm_before[k] = metrics_before[k] / vmax if vmax else 0
            norm_after[k] = metrics_after[k] / vmax if vmax else 0

        labels = [k.upper() for k in keys]
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        val_before = [norm_before[k] for k in keys] + [norm_before[keys[0]]]
        val_after  = [norm_after[k] for k in keys] + [norm_after[keys[0]]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.fill(angles, val_before, color=Palette.CORAL_RED, alpha=0.25)
        ax.plot(angles, val_before, color=Palette.CORAL_RED, linewidth=2.5,
                label=L.t("degraded"), marker='o', markersize=7)

        ax.fill(angles, val_after, color=Palette.VIVID_GREEN, alpha=0.25)
        ax.plot(angles, val_after, color=Palette.VIVID_GREEN, linewidth=2.5,
                label=L.t("enhanced"), marker='s', markersize=7)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([f'{v:.0%}' for v in [0.2, 0.4, 0.6, 0.8, 1.0]], fontsize=8, color='#94a3b8')
        ax.set_title(L.t("radar_chart"), fontweight='bold', pad=25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12))

        save_path = save_path or os.path.join(RESULTS_DIR, "radar_chart.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 3: Improvement Gain Bar / 提升幅度图
    # ----------------------------------------------------------------
    @staticmethod
    def plot_gain_bars(improvements: Dict[str, float],
                       save_path: str = None) -> str:
        """Horizontal bar chart showing per-metric gains"""
        if not improvements:
            return ""

        # Filter and sort / 过滤排序
        items = [(k.replace("_gain", "").upper(), v)
                 for k, v in improvements.items()
                 if isinstance(v, (int, float))]
        items.sort(key=lambda x: x[1], reverse=True)

        labels = [it[0] for it in items]
        values = [it[1] for it in items]
        colors = [Palette.VIVID_GREEN if v > 0 else Palette.CORAL_RED for v in values]

        fig, ax = plt.subplots(figsize=(10, max(3.5, len(items) * 0.7)))

        bars = ax.barh(labels, values, color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.8, height=0.55)

        ax.axvline(x=0, color='#94a3b8', linewidth=1.2, linestyle='-')
        ax.set_xlabel(L.t("gain_percent"), fontweight='bold')
        ax.set_title(L.t("gain_bar"), fontweight='bold', pad=15)
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            offset = 0.5 if val >= 0 else -0.5
            ax.annotate(f'{val:+.1f}%',
                        xy=(val, bar.get_y() + bar.get_height() / 2),
                        xytext=(8 * np.sign(val) if abs(val) > 1 else 15 * np.sign(val), 0),
                        textcoords="offset points",
                        ha='left' if val >= 0 else 'right',
                        va='center', fontsize=10, fontweight='bold',
                        color='#334155')

        save_path = save_path or os.path.join(RESULTS_DIR, "gain_bars.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 4: Training History / 训练曲线
    # ----------------------------------------------------------------
    @staticmethod
    def plot_training_history(history: Dict,
                              save_path: str = None) -> str:
        """Dual-panel training curves: loss + PSNR"""
        epochs = history.get("epoch", [])
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_psnr = history.get("train_psnr", [])
        val_psnr = history.get("val_psnr", [])

        if not epochs:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # ---- Loss subplot / 损失子图 ----
        ax1.plot(epochs, train_loss, color=Palette.DEEP_BLUE,
                 linewidth=2.2, label=L.t("train_loss"), marker='.', markersize=4)
        if val_loss:
            val_epochs = epochs[::max(1, len(epochs) // len(val_loss))][:len(val_loss)]
            ax1.plot(val_epochs, val_loss, color=Palette.CORAL_RED,
                     linewidth=2.2, linestyle='--', label=L.t("val_loss"),
                     marker='.', markersize=4)
        ax1.set_xlabel(L.t("epoch"), fontweight='bold')
        ax1.set_ylabel(L.t("loss"), fontweight='bold')
        ax1.set_title(L.t("training_loss"), fontweight='bold')
        ax1.legend(frameon=True)
        ax1.fill_between(epochs, train_loss, alpha=0.08, color=Palette.DEEP_BLUE)

        # ---- PSNR subplot / PSNR子图 ----
        if train_psnr:
            ax2.plot(epochs, train_psnr, color=Palette.VIVID_GREEN,
                     linewidth=2.2, label=L.t("train_psnr"), marker='.', markersize=4)
            if val_psnr:
                val_ep_psnr = epochs[::max(1, len(epochs) // len(val_psnr))][:len(val_psnr)]
                ax2.plot(val_ep_psnr, val_psnr, color=Palette.AMBER,
                         linewidth=2.2, linestyle='--', label=L.t("val_psnr"),
                         marker='.', markersize=4)
            ax2.set_xlabel(L.t("epoch"), fontweight='bold')
            ax2.set_ylabel(L.t("psnr_db"), fontweight='bold')
            ax2.set_title(L.t("training_psnr"), fontweight='bold')
            ax2.legend(frameon=True)
        else:
            ax2.text(0.5, 0.5, "PSNR data not available\n无 PSNR 数据",
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=12, color='#94a3b8')
            ax2.set_title(L.t("training_psnr"), fontweight='bold')

        fig.suptitle("Training Dashboard / 训练面板", fontweight='bold', fontsize=16, y=1.01)

        save_path = save_path or os.path.join(RESULTS_DIR, "training_history.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 5: Per-Image PSNR/SSIM / 逐图对比
    # ----------------------------------------------------------------
    @staticmethod
    def plot_per_image_metrics(results: List[Dict],
                                save_path: str = None) -> str:
        """Per-image PSNR/SSIM before and after enhancement"""
        if not results:
            return ""

        valid = [r for r in results if "error" not in r]
        if not valid:
            return ""

        names = [r.get("name", f"img{i}") for i, r in enumerate(valid)]
        psnr_before = [r["metrics_degraded"].get("psnr", 0) for r in valid]
        psnr_after  = [r["metrics_enhanced"].get("psnr", 0) for r in valid]
        ssim_before = [r["metrics_degraded"].get("ssim", 0) for r in valid]
        ssim_after  = [r["metrics_enhanced"].get("ssim", 0) for r in valid]

        x = np.arange(len(names))
        w = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # PSNR
        ax1.bar(x - w/2, psnr_before, w, color=Palette.CORAL_RED, alpha=0.85,
                label=L.t("degraded"), edgecolor='white', linewidth=0.5)
        ax1.bar(x + w/2, psnr_after, w, color=Palette.VIVID_GREEN, alpha=0.85,
                label=L.t("enhanced"), edgecolor='white', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n[:8] for n in names], rotation=30, ha='right', fontsize=8)
        ax1.set_ylabel(L.t("psnr_db"), fontweight='bold')
        ax1.set_title(L.t("psnr_db"), fontweight='bold')
        ax1.legend(frameon=True, fontsize=9)

        # SSIM
        ax2.bar(x - w/2, ssim_before, w, color=Palette.CORAL_RED, alpha=0.85,
                label=L.t("degraded"), edgecolor='white', linewidth=0.5)
        ax2.bar(x + w/2, ssim_after, w, color=Palette.VIVID_GREEN, alpha=0.85,
                label=L.t("enhanced"), edgecolor='white', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([n[:8] for n in names], rotation=30, ha='right', fontsize=8)
        ax2.set_ylabel(L.t("ssim"), fontweight='bold')
        ax2.set_title(L.t("ssim"), fontweight='bold')
        ax2.legend(frameon=True, fontsize=9)

        fig.suptitle(L.t("per_image"), fontweight='bold', fontsize=15, y=1.01)

        save_path = save_path or os.path.join(RESULTS_DIR, "per_image_metrics.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 6: Processing Time Distribution / 处理时间分布
    # ----------------------------------------------------------------
    @staticmethod
    def plot_time_distribution(results: List[Dict],
                                save_path: str = None) -> str:
        """Histogram + stats of processing time per image"""
        if not results:
            return ""

        times = [r.get("processing_time", 0) for r in results
                 if "error" not in r and "processing_time" in r]
        if not times:
            return ""

        fig, ax = plt.subplots(figsize=(9, 5.5))

        n, bins, patches = ax.hist(times, bins=min(15, len(times)),
                                    color=Palette.SKY, alpha=0.8,
                                    edgecolor='white', linewidth=1.2)

        # Color gradient on bars / 柱上渐变色
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.Blues(0.3 + 0.7 * i / len(patches)))

        mean_t = np.mean(times)
        ax.axvline(mean_t, color=Palette.CORAL_RED, linewidth=2.5, linestyle='--',
                   label=f'{L.t("avg_time")}: {mean_t:.3f}s')

        ax.set_xlabel(L.t("processing_time_s"), fontweight='bold')
        ax.set_ylabel(f'{L.t("total_images")}', fontweight='bold')
        ax.set_title(L.t("time_distribution"), fontweight='bold')
        ax.legend(frameon=True)

        # Stats box / 统计信息框
        stats_text = (
            f'Min: {np.min(times):.3f}s\n'
            f'Mean: {mean_t:.3f}s\n'
            f'Max: {np.max(times):.3f}s\n'
            f'Std: {np.std(times):.3f}s'
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#cbd5e1', alpha=0.9),
                family='monospace')

        save_path = save_path or os.path.join(RESULTS_DIR, "time_distribution.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 7: Image Grid Comparison / 图像网格对比
    # ----------------------------------------------------------------
    @staticmethod
    def plot_image_grid(images: List[np.ndarray], titles: List[str],
                        save_path: str = None, cols: int = 3,
                        main_title: str = None, figsize_per_cell: float = 3.5) -> str:
        """High-quality multi-image grid layout"""
        n = len(images)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                  figsize=(cols * figsize_per_cell,
                                           rows * figsize_per_cell + 0.6))
        # Normalize axes to 2D
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for i in range(rows * cols):
            r, c = i // cols, i % cols
            ax = axes[r][c]
            if i < n:
                img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(titles[i] if i < len(titles) else '',
                             fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.85,
                                       edgecolor='#e2e8f0'))
            ax.axis('off')

        if main_title:
            fig.suptitle(main_title, fontweight='bold', fontsize=15, y=1.01)

        save_path = save_path or os.path.join(RESULTS_DIR, "image_grid.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Chart 8: Comprehensive Dashboard / 综合仪表盘
    # ----------------------------------------------------------------
    @staticmethod
    def plot_dashboard(metrics_before: Dict[str, float],
                       metrics_after: Dict[str, float],
                       improvements: Dict[str, float],
                       results: List[Dict] = None,
                       save_path: str = None) -> str:
        """4-in-1 comprehensive dashboard / 四合一综合仪表盘"""
        fig = plt.figure(figsize=(18, 13))

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                              height_ratios=[1, 1])

        # (0,0): Bar comparison / 柱状图
        ax1 = fig.add_subplot(gs[0, 0])
        keys = [k for k in metrics_before if k in metrics_after]
        if keys:
            x = np.arange(len(keys))
            w = 0.35
            ax1.bar(x - w/2, [metrics_before[k] for k in keys], w,
                    color=Palette.CORAL_RED, alpha=0.85, label=L.t("degraded"))
            ax1.bar(x + w/2, [metrics_after[k] for k in keys], w,
                    color=Palette.VIVID_GREEN, alpha=0.85, label=L.t("enhanced"))
            ax1.set_xticks(x)
            ax1.set_xticklabels([k.upper() for k in keys], fontsize=9)
            ax1.set_title(L.t("metrics_comparison"), fontweight='bold', fontsize=12)
            ax1.legend(fontsize=8)

        # (0,1): Radar / 雷达图
        ax2 = fig.add_subplot(gs[0, 1], polar=True)
        radar_keys = [k for k in keys if k != "lpips"][:6]
        if len(radar_keys) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
            angles += angles[:1]
            for data, color, label in [
                ([metrics_before[k] for k in radar_keys] + [metrics_before[radar_keys[0]]],
                 Palette.CORAL_RED, L.t("degraded")),
                ([metrics_after[k] for k in radar_keys] + [metrics_after[radar_keys[0]]],
                 Palette.VIVID_GREEN, L.t("enhanced"))
            ]:
                vmax = max(data[:-1]) if max(data[:-1]) > 0 else 1
                normed = [v / vmax for v in data]
                ax2.fill(angles, normed, color=color, alpha=0.2)
                ax2.plot(angles, normed, color=color, linewidth=2, label=label)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels([k.upper() for k in radar_keys], fontsize=8)
            ax2.set_ylim(0, 1.15)
            ax2.set_title(L.t("radar_chart"), fontweight='bold', fontsize=12, pad=20)
            ax2.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # (0,2): Gain bars / 提升幅度
        ax3 = fig.add_subplot(gs[0, 2])
        if improvements:
            items = [(k.replace("_gain", "").upper(), v)
                     for k, v in improvements.items()
                     if isinstance(v, (int, float))]
            items.sort(key=lambda x: x[1], reverse=True)
            labels_i = [it[0] for it in items]
            values_i = [it[1] for it in items]
            colors_i = [Palette.VIVID_GREEN if v > 0 else Palette.CORAL_RED for v in values_i]
            ax3.barh(labels_i, values_i, color=colors_i, alpha=0.85, height=0.55)
            ax3.axvline(x=0, color='#94a3b8', linewidth=1)
            ax3.set_title(L.t("gain_bar"), fontweight='bold', fontsize=12)
            for bar, val in zip(ax3.patches, values_i):
                ax3.annotate(f'{val:+.1f}%', xy=(val, bar.get_y() + bar.get_height()/2),
                            xytext=(5 * np.sign(val), 0), textcoords="offset points",
                            ha='left' if val >= 0 else 'right', va='center', fontsize=9)

        # (1,0): Per-image PSNR / 逐图PSNR
        ax4 = fig.add_subplot(gs[1, 0])
        if results:
            valid_r = [r for r in results if "error" not in r]
            if valid_r:
                names = [r.get("name", f"#{i}")[:8] for i, r in enumerate(valid_r)]
                xi = np.arange(len(names))
                psnr_b = [r["metrics_degraded"].get("psnr", 0) for r in valid_r]
                psnr_a = [r["metrics_enhanced"].get("psnr", 0) for r in valid_r]
                ax4.plot(xi, psnr_b, 'o-', color=Palette.CORAL_RED, linewidth=1.8,
                        markersize=6, label=L.t("degraded"))
                ax4.plot(xi, psnr_a, 's-', color=Palette.VIVID_GREEN, linewidth=1.8,
                        markersize=6, label=L.t("enhanced"))
                ax4.set_xticks(xi)
                ax4.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
                ax4.set_ylabel(L.t("psnr_db"), fontweight='bold')
                ax4.set_title(L.t("psnr_db"), fontweight='bold', fontsize=12)
                ax4.legend(fontsize=8)

        # (1,1): Per-image SSIM / 逐图SSIM
        ax5 = fig.add_subplot(gs[1, 1])
        if results:
            valid_r = [r for r in results if "error" not in r]
            if valid_r:
                names = [r.get("name", f"#{i}")[:8] for i, r in enumerate(valid_r)]
                xi = np.arange(len(names))
                ssim_b = [r["metrics_degraded"].get("ssim", 0) for r in valid_r]
                ssim_a = [r["metrics_enhanced"].get("ssim", 0) for r in valid_r]
                ax5.plot(xi, ssim_b, 'o-', color=Palette.CORAL_RED, linewidth=1.8,
                        markersize=6, label=L.t("degraded"))
                ax5.plot(xi, ssim_a, 's-', color=Palette.VIVID_GREEN, linewidth=1.8,
                        markersize=6, label=L.t("enhanced"))
                ax5.set_xticks(xi)
                ax5.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
                ax5.set_ylabel(L.t("ssim"), fontweight='bold')
                ax5.set_title(L.t("ssim"), fontweight='bold', fontsize=12)
                ax5.legend(fontsize=8)

        # (1,2): Time distribution / 时间分布
        ax6 = fig.add_subplot(gs[1, 2])
        if results:
            times = [r.get("processing_time", 0) for r in valid_r
                     if "processing_time" in r]
            if times:
                ax6.hist(times, bins=min(10, len(times)), color=Palette.SKY,
                        alpha=0.8, edgecolor='white', linewidth=1.2)
                ax6.axvline(np.mean(times), color=Palette.CORAL_RED,
                           linewidth=2, linestyle='--',
                           label=f'Avg: {np.mean(times):.3f}s')
                ax6.set_xlabel(L.t("processing_time_s"), fontweight='bold', fontsize=9)
                ax6.set_title(L.t("time_distribution"), fontweight='bold', fontsize=12)
                ax6.legend(fontsize=8)

        fig.suptitle("Image Enhancement Dashboard / 图像增强仪表盘",
                     fontweight='bold', fontsize=17, y=1.01)

        save_path = save_path or os.path.join(RESULTS_DIR, "dashboard.png")
        return _save_fig(fig, save_path, dpi=180)

    # ----------------------------------------------------------------
    # Chart 9: Side-by-side image comparison with labels / 带标注的并排对比
    # ----------------------------------------------------------------
    @staticmethod
    def plot_side_by_side(original: np.ndarray,
                          degraded: np.ndarray,
                          enhanced: np.ndarray,
                          metrics_degraded: Dict[str, float] = None,
                          metrics_enhanced: Dict[str, float] = None,
                          save_path: str = None) -> str:
        """Professional side-by-side comparison with metric overlays"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

        images = [original, degraded, enhanced]
        labels = [L.t("original"), L.t("degraded"), L.t("enhanced")]
        colors = [Palette.DEEP_BLUE, Palette.CORAL_RED, Palette.VIVID_GREEN]
        metrics_list = [None, metrics_degraded, metrics_enhanced]

        for i, ax in enumerate(axes):
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(labels[i], fontweight='bold', fontsize=13,
                         color=colors[i], pad=10)
            ax.axis('off')

            # Metric annotation / 指标标注
            if metrics_list[i]:
                text = ""
                if "psnr" in metrics_list[i]:
                    text += f"PSNR: {metrics_list[i]['psnr']:.2f} dB\n"
                if "ssim" in metrics_list[i]:
                    text += f"SSIM: {metrics_list[i]['ssim']:.4f}"
                if text:
                    ax.text(0.5, -0.08, text.strip(), transform=ax.transAxes,
                            ha='center', fontsize=10, fontfamily='monospace',
                            color='#475569',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f1f5f9',
                                      edgecolor='#cbd5e1', alpha=0.9))

        fig.suptitle(L.t("image_grid"), fontweight='bold', fontsize=15, y=1.01)

        save_path = save_path or os.path.join(RESULTS_DIR, "side_by_side.png")
        return _save_fig(fig, save_path)

    # ----------------------------------------------------------------
    # Bulk generation from evaluator results / 批量生成
    # ----------------------------------------------------------------
    @classmethod
    def generate_all(cls, summary: Dict, results: List[Dict],
                     output_dir: str = None) -> List[str]:
        """Generate all visualization charts from evaluator output"""
        output_dir = output_dir or RESULTS_DIR
        saved = []

        degraded_avg = summary.get("degraded_avg", {})
        enhanced_avg = summary.get("enhanced_avg", {})
        improvements = summary.get("improvements_avg", {})

        # 1. Metrics comparison / 指标对比
        p = cls.plot_metrics_comparison(
            degraded_avg, enhanced_avg,
            os.path.join(output_dir, "metrics_comparison.png"))
        if p: saved.append(p)

        # 2. Radar chart / 雷达图
        p = cls.plot_radar_chart(
            degraded_avg, enhanced_avg,
            os.path.join(output_dir, "radar_chart.png"))
        if p: saved.append(p)

        # 3. Gain bars / 提升幅度
        p = cls.plot_gain_bars(
            improvements,
            os.path.join(output_dir, "gain_bars.png"))
        if p: saved.append(p)

        # 4. Per-image metrics / 逐图指标
        p = cls.plot_per_image_metrics(
            results,
            os.path.join(output_dir, "per_image_metrics.png"))
        if p: saved.append(p)

        # 5. Time distribution / 时间分布
        p = cls.plot_time_distribution(
            results,
            os.path.join(output_dir, "time_distribution.png"))
        if p: saved.append(p)

        # 6. Dashboard / 综合仪表盘
        p = cls.plot_dashboard(
            degraded_avg, enhanced_avg, improvements, results,
            os.path.join(output_dir, "dashboard.png"))
        if p: saved.append(p)

        return saved


# ==================== Convenience / 便捷函数 ====================

def quick_compare(original: np.ndarray, degraded: np.ndarray,
                  enhanced: np.ndarray, save_path: str = None) -> str:
    """Quick side-by-side comparison / 快速并排对比"""
    return Visualizer.plot_side_by_side(original, degraded, enhanced,
                                        save_path=save_path)
