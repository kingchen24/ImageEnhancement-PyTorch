# ============================================================
# 主入口 - 低质量图像重建与评价系统
# 用法:
#   python main.py demo           # 运行完整演示流程
#   python main.py eval           # 批量评测模式
#   python main.py train          # 训练模型
#   python main.py enhance <img>  # 单图增强
# ============================================================

import os
import sys
import io

# 强制UTF-8输出 (修复Windows控制台编码问题)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import cv2
import torch
import argparse
import json
import numpy as np
from utils import imread, imwrite

from config import *
from degradation import ImageDegradation
from enhancement import ImageEnhancement
from evaluation import ImageQualityMetrics
from automated_eval import AutomatedEvaluator
from visualizer import Visualizer
from models import MSFENet, MSFENetLite


# ==================== 辅助函数 ====================

def create_demo_images():
    """创建演示用测试图像"""
    os.makedirs(INPUT_DIR, exist_ok=True)

    demo_paths = []
    for i in range(3):
        # 生成彩色渐变图
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                img[y, x, 0] = int(128 + 127 * np.sin(x / 30.0) * np.cos(y / 30.0))  # B
                img[y, x, 1] = int(128 + 127 * np.sin((x + y) / 40.0))                 # G
                img[y, x, 2] = int(128 + 127 * np.cos(x / 25.0) * np.sin(y / 25.0))    # R

        # 添加圆和矩形
        cv2.circle(img, (180, 180), 60, (200, 100, 50), -1)
        cv2.rectangle(img, (30, 30), (100, 100), (50, 150, 200), -1)
        cv2.putText(img, f"Test {i+1}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        path = os.path.join(INPUT_DIR, f"demo_{i:02d}.png")
        imwrite(path, img)
        demo_paths.append(path)
        print(f"  Created: {path}")

    return demo_paths


def load_or_create_model(device: str = 'cpu',
                         checkpoint_path: str = None) -> torch.nn.Module:
    """加载模型或创建新模型"""
    model = MSFENet(
        in_channels=MODEL["in_channels"],
        out_channels=MODEL["out_channels"],
        base_channels=MODEL["base_channels"],
        num_blocks=MODEL["num_blocks"],
        num_scales=MODEL["num_scales"],
        use_attention=MODEL["use_attention"],
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print("Using untrained model (random weights) -- results will be suboptimal.")

    model.to(device)
    model.eval()
    return model


# ==================== 子命令 ====================

def cmd_demo():
    """完整演示流程"""
    print("\n" + "=" * 60)
    print("  低质量图像重建与评价系统 - 演示流程")
    print("=" * 60)

    # 1. 准备数据
    print("\n[Step 1] 创建演示图像...")
    demo_paths = create_demo_images()

    # 2. 退化模拟
    print("\n[Step 2] 图像退化模拟...")
    deg_config = {
        "gaussian_noise": {"enabled": True, "std": 25},
        "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
        "jpeg_compression": {"enabled": True, "quality": 20},
    }

    for path in demo_paths:
        img = imread(path)
        degraded, meta = ImageDegradation.apply_degradation_pipeline(img, deg_config)
        name = os.path.basename(path)
        out = os.path.join(DEGRADED_DIR, name)
        imwrite(out, degraded)
        print(f"  {name}: degraded -> {meta}")

    # 3. 传统方法增强
    print("\n[Step 3] 传统方法增强...")
    enhance_tasks = [
        {"op": "denoise_bilateral", "kwargs": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
        {"op": "sharpen_unsharp", "kwargs": {"sigma": 1.0, "amount": 1.5}},
        {"op": "enhance_clahe", "kwargs": {"clip_limit": 2.0, "tile_size": 8}},
    ]

    for path in demo_paths:
        img = imread(path)
        degraded, _ = ImageDegradation.apply_degradation_pipeline(img, deg_config)
        enhanced = ImageEnhancement.enhance_pipeline(degraded, enhance_tasks)
        name = os.path.basename(path)
        imwrite(os.path.join(ENHANCED_DIR, f"trad_{name}"), enhanced)

        # 评估
        metrics = ImageQualityMetrics.evaluate_all(img, enhanced)
        print(f"  {name}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")

    # 4. 深度学习增强 (如果有模型)
    print("\n[Step 4] 深度学习增强...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    best_ckpt = os.path.join(CHECKPOINT_DIR, "msfe_best.pth")
    model = load_or_create_model(device, best_ckpt if os.path.exists(best_ckpt) else None)
    enhancer = ImageEnhancement(model, device)

    for path in demo_paths:
        img = imread(path)
        degraded, _ = ImageDegradation.apply_degradation_pipeline(img, deg_config)
        enhanced = enhancer.enhance_dl(degraded)
        name = os.path.basename(path)
        imwrite(os.path.join(ENHANCED_DIR, f"dl_{name}"), enhanced)

        metrics = ImageQualityMetrics.evaluate_all(img, enhanced)
        print(f"  {name}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")

    # 5. 自动化评测
    print("\n[Step 5] 自动化评测平台...")
    evaluator = AutomatedEvaluator(model, device)
    evaluator.process_batch(INPUT_DIR, [deg_config], enhance_method="dl")

    summary = evaluator.summary()
    if "error" not in summary:
        print(f"\n  📊 Summary:")
        print(f"     Images processed: {summary['total_images']}")
        print(f"     Degraded PSNR avg: {summary['degraded_avg'].get('psnr','N/A')}")
        print(f"     Enhanced PSNR avg: {summary['enhanced_avg'].get('psnr','N/A')}")
        print(f"     PSNR average gain: {summary['improvements_avg'].get('psnr_gain','N/A')}%")
        print(f"     SSIM average gain: {summary['improvements_avg'].get('ssim_gain','N/A')}%")

    # 6. 生成报告
    print("\n[Step 6] 生成报告...")
    evaluator.generate_report(format="html")
    evaluator.generate_report(format="markdown")
    evaluator.generate_report(format="json")
    evaluator.save_comparisons()

    # 7. 可视化
    print("\n[Step 7] 可视化...")
    # 指标对比图
    metrics_before = summary.get("degraded_avg", {})
    metrics_after = summary.get("enhanced_avg", {})
    Visualizer.plot_metrics_comparison(metrics_before, metrics_after)

    print("\n" + "=" * 60)
    print("  演示完成！")
    print(f"  结果文件请查看: {RESULTS_DIR}")
    print(f"  增强图像请查看: {ENHANCED_DIR}")
    print("=" * 60 + "\n")


def cmd_eval():
    """批量评测模式"""
    print("\n" + "=" * 60)
    print("  批量评测模式")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_ckpt = os.path.join(CHECKPOINT_DIR, "msfe_best.pth")

    model = None
    if os.path.exists(best_ckpt):
        model = load_or_create_model(device, best_ckpt)

    # 多组退化配置
    deg_configs = [
        {
            "gaussian_noise": {"enabled": True, "std": 15},
            "jpeg_compression": {"enabled": True, "quality": 30},
        },
        {
            "gaussian_noise": {"enabled": True, "std": 25},
            "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
            "jpeg_compression": {"enabled": True, "quality": 20},
        },
        {
            "salt_pepper_noise": {"enabled": True, "amount": 0.03},
            "motion_blur": {"enabled": True, "kernel_size": 15, "angle": 30},
            "jpeg_compression": {"enabled": True, "quality": 15},
        },
    ]

    evaluator = AutomatedEvaluator(model, device)
    evaluator.process_batch(INPUT_DIR, deg_configs, enhance_method="dl" if model else "traditional")

    summary = evaluator.summary()
    print(f"\n📊 Evaluation Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    evaluator.generate_report(format="html")
    evaluator.generate_report(format="markdown")
    evaluator.save_comparisons()

    print(f"\nReports saved to: {RESULTS_DIR}")


def cmd_enhance(img_path: str):
    """单图增强"""
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_ckpt = os.path.join(CHECKPOINT_DIR, "msfe_best.pth")

    model = load_or_create_model(device, best_ckpt if os.path.exists(best_ckpt) else None)
    enhancer = ImageEnhancement(model, device)

    img = imread(img_path)
    if img is None:
        print(f"Error: Cannot read image: {img_path}")
        sys.exit(1)

    # 退化 + 增强 + 对比
    deg_config = {
        "gaussian_noise": {"enabled": True, "std": 25},
        "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
        "jpeg_compression": {"enabled": True, "quality": 20},
    }

    degraded, _ = ImageDegradation.apply_degradation_pipeline(img, deg_config)
    enhanced = enhancer.enhance_dl(degraded)

    # 保存
    name = os.path.splitext(os.path.basename(img_path))[0]
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_original.png"), img)
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_degraded.png"), degraded)
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_enhanced.png"), enhanced)

    # 评估
    metrics_d = ImageQualityMetrics.evaluate_all(img, degraded)
    metrics_e = ImageQualityMetrics.evaluate_all(img, enhanced)

    print(f"\n{'='*50}")
    print(f"  Image: {img_path}")
    print(f"  Size: {img.shape}")
    print(f"{'='*50}")
    print(f"  Degraded -> Enhanced:")
    for k in metrics_d:
        if k in metrics_e:
            print(f"    {k.upper():8s}: {metrics_d[k]:.4f} -> {metrics_e[k]:.4f}")
    print(f"{'='*50}")

    # 对比图
    comparison = np.hstack([img, degraded, enhanced])
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_comparison.png"), comparison)
    print(f"  Results saved to: {ENHANCED_DIR}")


def cmd_train():
    """训练模型"""
    from train import train, DegradationDataset

    print("\n" + "=" * 60)
    print("  模型训练模式")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # 确保有训练数据
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print("\n  No training images found. Creating demo images...")
        create_demo_images()

    model = MSFENet(**MODEL)
    print(f"  Model: MSFENet ({sum(p.numel() for p in model.parameters()):,} params)")

    # 数据集
    deg_config = {
        "gaussian_noise": {"enabled": True, "std": 25},
        "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
        "jpeg_compression": {"enabled": True, "quality": 20},
    }

    train_ds = DegradationDataset(INPUT_DIR, TRAIN["patch_size"],
                                  num_patches_per_img=16,
                                  degradation_config=deg_config)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=TRAIN["batch_size"],
        shuffle=True, num_workers=TRAIN["num_workers"],
        drop_last=True
    )

    print(f"  Training samples: {len(train_ds)}")
    print(f"  Batch size: {TRAIN['batch_size']}")
    print(f"  Epochs: {TRAIN['num_epochs']}")
    print(f"  LR: {TRAIN['learning_rate']}")
    print()

    model, history = train(model, train_loader, None, TRAIN, device)

    # 保存训练曲线
    Visualizer.plot_training_history(history)


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="低质量图像重建与评价系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo             完整演示流程
  python main.py eval             批量评测
  python main.py train            训练模型
  python main.py enhance img.jpg  单图增强
        """)
    parser.add_argument("command", nargs="?",
                        choices=["demo", "eval", "train", "enhance"],
                        default="demo",
                        help="运行模式 (default: demo)")
    parser.add_argument("path", nargs="?",
                        help="图像路径 (仅 enhance 模式需要)")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo()
    elif args.command == "eval":
        cmd_eval()
    elif args.command == "train":
        cmd_train()
    elif args.command == "enhance":
        if not args.path:
            print("Error: enhance mode requires an image path")
            print("Usage: python main.py enhance <image_path>")
            sys.exit(1)
        cmd_enhance(args.path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
