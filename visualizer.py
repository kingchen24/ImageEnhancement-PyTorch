# ============================================================
# 可视化模块
# ============================================================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

from config import RESULTS_DIR


class Visualizer:
    """结果可视化"""

    @staticmethod
    def plot_metrics_comparison(metrics_before: Dict[str, float],
                                metrics_after: Dict[str, float],
                                save_path: str = None) -> str:
        """绘制增强前后指标对比柱状图"""
        keys = [k for k in metrics_before if k in metrics_after]
        if not keys:
            return ""

        x = np.arange(len(keys))
        w = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - w / 2, [metrics_before[k] for k in keys],
                       w, label='Degraded', color='#e74c3c', alpha=0.85)
        bars2 = ax.bar(x + w / 2, [metrics_after[k] for k in keys],
                       w, label='Enhanced', color='#2ecc71', alpha=0.85)

        ax.set_ylabel('Score')
        ax.set_title('Image Quality Metrics: Before vs After Enhancement', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([k.upper() for k in keys])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

        plt.tight_layout()
        save_path = save_path or os.path.join(RESULTS_DIR, "metrics_comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    @staticmethod
    def plot_training_history(history: Dict,
                              save_path: str = None) -> str:
        """绘制训练损失曲线"""
        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = history.get("epoch", [])
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        if val_loss:
            val_epochs = epochs[::max(1, len(epochs) // len(val_loss))][:len(val_loss)]
            ax.plot(val_epochs, val_loss, 'r--', label='Val Loss', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = save_path or os.path.join(RESULTS_DIR, "training_loss.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    @staticmethod
    def plot_image_grid(images: List[np.ndarray], titles: List[str],
                        save_path: str = None, cols: int = 3) -> str:
        """绘制多图网格"""
        n = len(images)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
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
                             fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        save_path = save_path or os.path.join(RESULTS_DIR, "image_grid.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
