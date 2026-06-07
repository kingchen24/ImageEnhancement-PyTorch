# ============================================================
# Main Entry / 主入口 — 低质量图像重建与评价系统
# Low-Quality Image Enhancement & Evaluation System
#
# Usage / 用法:
#   python main.py demo              Run full demo / 运行完整演示流程
#   python main.py eval              Batch evaluation / 批量评测模式
#   python main.py train             Train model / 训练模型
#   python main.py enhance <img>     Single image enhancement / 单图增强
#   python main.py lang en           Switch to English / 切换英文
#   python main.py lang zh           Switch to Chinese / 切换中文
# ============================================================

import os
import sys
import io

# Force UTF-8 output (fix Windows console encoding) / 强制UTF-8输出
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


# ==================== Bilingual Output / 双语输出 ====================

class T:
    """Terminal i18n messages / 终端双语消息"""

    _DICT = {
        "demo_header": {
            "zh": "低质量图像重建与评价系统 - 演示流程",
            "en": "Low-Quality Image Enhancement & Evaluation System - Demo"
        },
        "step_create": {
            "zh": "[Step 1] 创建演示图像...",
            "en": "[Step 1] Creating demo images..."
        },
        "step_degrade": {
            "zh": "[Step 2] 图像退化模拟...",
            "en": "[Step 2] Image degradation simulation..."
        },
        "step_traditional": {
            "zh": "[Step 3] 传统方法增强...",
            "en": "[Step 3] Traditional enhancement..."
        },
        "step_dl": {
            "zh": "[Step 4] 深度学习增强...",
            "en": "[Step 4] Deep learning enhancement..."
        },
        "step_eval": {
            "zh": "[Step 5] 自动化评测平台...",
            "en": "[Step 5] Automated evaluation..."
        },
        "step_report": {
            "zh": "[Step 6] 生成报告...",
            "en": "[Step 6] Generating reports..."
        },
        "step_visual": {
            "zh": "[Step 7] 生成可视化图表...",
            "en": "[Step 7] Generating visualization charts..."
        },
        "device_info": {
            "zh": "设备",
            "en": "Device"
        },
        "created": {
            "zh": "已创建",
            "en": "Created"
        },
        "degraded_to": {
            "zh": "退化 ->",
            "en": "degraded ->"
        },
        "loaded_ckpt": {
            "zh": "已加载检查点",
            "en": "Loaded checkpoint from"
        },
        "untrained_warning": {
            "zh": "使用未训练模型 (随机权重) -- 结果仅供参考",
            "en": "Using untrained model (random weights) -- results will be suboptimal"
        },
        "demo_done": {
            "zh": "演示完成！",
            "en": "Demo complete!"
        },
        "results_in": {
            "zh": "结果文件请查看",
            "en": "Results saved to"
        },
        "enhanced_in": {
            "zh": "增强图像请查看",
            "en": "Enhanced images in"
        },
        "eval_mode": {
            "zh": "批量评测模式",
            "en": "Batch Evaluation Mode"
        },
        "eval_summary": {
            "zh": "评测汇总",
            "en": "Evaluation Summary"
        },
        "summary_label": {
            "zh": "📊 汇总:",
            "en": "📊 Summary:"
        },
        "images_processed": {
            "zh": "处理图像数",
            "en": "Images processed"
        },
        "degraded_psnr_avg": {
            "zh": "退化 PSNR 均值",
            "en": "Degraded PSNR avg"
        },
        "enhanced_psnr_avg": {
            "zh": "增强 PSNR 均值",
            "en": "Enhanced PSNR avg"
        },
        "psnr_avg_gain": {
            "zh": "PSNR 平均提升",
            "en": "PSNR average gain"
        },
        "ssim_avg_gain": {
            "zh": "SSIM 平均提升",
            "en": "SSIM average gain"
        },
        "reports_to": {
            "zh": "报告已保存至",
            "en": "Reports saved to"
        },
        "train_mode": {
            "zh": "模型训练模式",
            "en": "Model Training Mode"
        },
        "no_train_images": {
            "zh": "未找到训练图像，正在创建演示图像...",
            "en": "No training images found. Creating demo images..."
        },
        "model_params": {
            "zh": "模型参数量",
            "en": "params"
        },
        "train_samples": {
            "zh": "训练样本数",
            "en": "Training samples"
        },
        "batch_size_label": {
            "zh": "批次大小",
            "en": "Batch size"
        },
        "epochs_label": {
            "zh": "训练轮次",
            "en": "Epochs"
        },
        "lr_label": {
            "zh": "学习率",
            "en": "LR"
        },
        "enhance_single": {
            "zh": "单图增强模式",
            "en": "Single Image Enhancement"
        },
        "img_not_found": {
            "zh": "错误: 图像未找到",
            "en": "Error: Image not found"
        },
        "img_read_error": {
            "zh": "错误: 无法读取图像",
            "en": "Error: Cannot read image"
        },
        "img_info": {
            "zh": "图像信息",
            "en": "Image"
        },
        "size_info": {
            "zh": "尺寸",
            "en": "Size"
        },
        "degraded_enhanced_arrow": {
            "zh": "退化 -> 增强",
            "en": "Degraded -> Enhanced"
        },
        "results_saved": {
            "zh": "结果已保存至",
            "en": "Results saved to"
        },
        "charts_generated": {
            "zh": "已生成",
            "en": "Generated"
        },
        "charts_count": {
            "zh": "张可视化图表",
            "en": "visualization charts"
        },
        "lang_switch": {
            "zh": "语言已切换为: 中文",
            "en": "Language switched to: English"
        },
        "enhance_need_path": {
            "zh": "错误: enhance 模式需要提供图像路径",
            "en": "Error: enhance mode requires an image path"
        },
        "enhance_usage": {
            "zh": "用法: python main.py enhance <图像路径>",
            "en": "Usage: python main.py enhance <image_path>"
        },
    }

    @classmethod
    def t(cls, key: str) -> str:
        return cls._DICT.get(key, {}).get(LANG, key)


# ==================== Helper Functions / 辅助函数 ====================

def create_demo_images():
    """Create demo test images / 创建演示用测试图像"""
    os.makedirs(INPUT_DIR, exist_ok=True)

    demo_paths = []
    for i in range(3):
        # Generate color gradient patterns / 生成彩色渐变图
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                img[y, x, 0] = int(128 + 127 * np.sin(x / 30.0) * np.cos(y / 30.0))  # B
                img[y, x, 1] = int(128 + 127 * np.sin((x + y) / 40.0))                 # G
                img[y, x, 2] = int(128 + 127 * np.cos(x / 25.0) * np.sin(y / 25.0))    # R

        # Add shapes / 添加几何图形
        cv2.circle(img, (180, 180), 60, (200, 100, 50), -1)
        cv2.rectangle(img, (30, 30), (100, 100), (50, 150, 200), -1)
        cv2.putText(img, f"Test {i+1}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        path = os.path.join(INPUT_DIR, f"demo_{i:02d}.png")
        imwrite(path, img)
        demo_paths.append(path)
        print(f"  {T.t('created')}: {path}")

    return demo_paths


def load_or_create_model(device: str = 'cpu',
                         checkpoint_path: str = None) -> torch.nn.Module:
    """Load model or create new one / 加载模型或创建新模型"""
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
        print(f"  {T.t('loaded_ckpt')}: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"  ⚠ {T.t('untrained_warning')}")

    model.to(device)
    model.eval()
    return model


# ==================== Sub-commands / 子命令 ====================

def cmd_demo():
    """Full demo pipeline / 完整演示流程"""
    print("\n" + "=" * 60)
    print(f"  {T.t('demo_header')}")
    print("=" * 60)

    # 1. Prepare data / 准备数据
    print(f"\n{T.t('step_create')}")
    demo_paths = create_demo_images()

    # 2. Degradation simulation / 退化模拟
    print(f"\n{T.t('step_degrade')}")
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
        print(f"  {name}: {T.t('degraded_to')} {meta}")

    # 3. Traditional enhancement / 传统方法增强
    print(f"\n{T.t('step_traditional')}")
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

        # Evaluate / 评估
        metrics = ImageQualityMetrics.evaluate_all(img, enhanced)
        print(f"  {name}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")

    # 4. Deep learning enhancement / 深度学习增强
    print(f"\n{T.t('step_dl')}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  {T.t('device_info')}: {device}")

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

    # 5. Automated evaluation / 自动化评测
    print(f"\n{T.t('step_eval')}")
    evaluator = AutomatedEvaluator(model, device)
    evaluator.process_batch(INPUT_DIR, [deg_config], enhance_method="dl")

    summary = evaluator.summary()
    if "error" not in summary:
        print(f"\n  {T.t('summary_label')}")
        print(f"     {T.t('images_processed')}: {summary['total_images']}")
        print(f"     {T.t('degraded_psnr_avg')}: {summary['degraded_avg'].get('psnr','N/A')}")
        print(f"     {T.t('enhanced_psnr_avg')}: {summary['enhanced_avg'].get('psnr','N/A')}")
        print(f"     {T.t('psnr_avg_gain')}: {summary['improvements_avg'].get('psnr_gain','N/A')}%")
        print(f"     {T.t('ssim_avg_gain')}: {summary['improvements_avg'].get('ssim_gain','N/A')}%")

    # 6. Generate reports / 生成报告
    print(f"\n{T.t('step_report')}")
    evaluator.generate_report(format="html")
    evaluator.generate_report(format="markdown")
    evaluator.generate_report(format="json")
    evaluator.save_comparisons()

    # 7. Visualization / 可视化
    print(f"\n{T.t('step_visual')}")
    saved_charts = Visualizer.generate_all(summary, evaluator.results)
    print(f"  {T.t('charts_generated')} {len(saved_charts)} {T.t('charts_count')}")

    print("\n" + "=" * 60)
    print(f"  {T.t('demo_done')}")
    print(f"  {T.t('results_in')}: {RESULTS_DIR}")
    print(f"  {T.t('enhanced_in')}: {ENHANCED_DIR}")
    print("=" * 60 + "\n")


def cmd_eval():
    """Batch evaluation mode / 批量评测模式"""
    print("\n" + "=" * 60)
    print(f"  {T.t('eval_mode')}")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_ckpt = os.path.join(CHECKPOINT_DIR, "msfe_best.pth")

    model = None
    if os.path.exists(best_ckpt):
        model = load_or_create_model(device, best_ckpt)

    # Multiple degradation configs / 多组退化配置
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
    evaluator.process_batch(INPUT_DIR, deg_configs,
                            enhance_method="dl" if model else "traditional")

    summary = evaluator.summary()
    print(f"\n{T.t('eval_summary')}:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    evaluator.generate_report(format="html")
    evaluator.generate_report(format="markdown")
    evaluator.generate_report(format="json")
    evaluator.save_comparisons()

    # Generate all visualizations / 生成所有可视化
    saved_charts = Visualizer.generate_all(summary, evaluator.results)
    print(f"\n{T.t('charts_generated')} {len(saved_charts)} {T.t('charts_count')}")

    print(f"\n{T.t('reports_to')}: {RESULTS_DIR}")


def cmd_enhance(img_path: str):
    """Single image enhancement / 单图增强"""
    if not os.path.exists(img_path):
        print(f"{T.t('img_not_found')}: {img_path}")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_ckpt = os.path.join(CHECKPOINT_DIR, "msfe_best.pth")

    model = load_or_create_model(device, best_ckpt if os.path.exists(best_ckpt) else None)
    enhancer = ImageEnhancement(model, device)

    img = imread(img_path)
    if img is None:
        print(f"{T.t('img_read_error')}: {img_path}")
        sys.exit(1)

    # Degradation + Enhancement + Comparison / 退化 + 增强 + 对比
    deg_config = {
        "gaussian_noise": {"enabled": True, "std": 25},
        "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
        "jpeg_compression": {"enabled": True, "quality": 20},
    }

    degraded, _ = ImageDegradation.apply_degradation_pipeline(img, deg_config)
    enhanced = enhancer.enhance_dl(degraded)

    # Save results / 保存结果
    name = os.path.splitext(os.path.basename(img_path))[0]
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_original.png"), img)
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_degraded.png"), degraded)
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_enhanced.png"), enhanced)

    # Evaluate / 评估
    metrics_d = ImageQualityMetrics.evaluate_all(img, degraded)
    metrics_e = ImageQualityMetrics.evaluate_all(img, enhanced)

    print(f"\n{'='*50}")
    print(f"  {T.t('img_info')}: {img_path}")
    print(f"  {T.t('size_info')}: {img.shape}")
    print(f"{'='*50}")
    print(f"  {T.t('degraded_enhanced_arrow')}:")
    for k in metrics_d:
        if k in metrics_e:
            print(f"    {k.upper():8s}: {metrics_d[k]:.4f} -> {metrics_e[k]:.4f}")
    print(f"{'='*50}")

    # Side-by-side comparison / 并排对比图
    Visualizer.plot_side_by_side(
        img, degraded, enhanced,
        metrics_degraded=metrics_d,
        metrics_enhanced=metrics_e,
        save_path=os.path.join(ENHANCED_DIR, f"{name}_side_by_side.png")
    )

    # Simple horizontal concatenation / 水平拼接对比
    comparison = np.hstack([img, degraded, enhanced])
    imwrite(os.path.join(ENHANCED_DIR, f"{name}_comparison.png"), comparison)
    print(f"  {T.t('results_saved')}: {ENHANCED_DIR}")


def cmd_train():
    """Train the model / 训练模型"""
    from train import train, DegradationDataset

    print("\n" + "=" * 60)
    print(f"  {T.t('train_mode')}")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  {T.t('device_info')}: {device}")

    # Ensure training data exists / 确保有训练数据
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"\n  {T.t('no_train_images')}")
        create_demo_images()

    model = MSFENet(**MODEL)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: MSFENet ({param_count:,} {T.t('model_params')})")

    # Dataset / 数据集
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

    print(f"  {T.t('train_samples')}: {len(train_ds)}")
    print(f"  {T.t('batch_size_label')}: {TRAIN['batch_size']}")
    print(f"  {T.t('epochs_label')}: {TRAIN['num_epochs']}")
    print(f"  {T.t('lr_label')}: {TRAIN['learning_rate']}")
    print()

    model, history = train(model, train_loader, None, TRAIN, device)

    # Save training curves / 保存训练曲线
    Visualizer.plot_training_history(history)
    print(f"\n  Training curves saved to {RESULTS_DIR}/training_history.png")


# ==================== Main Entry / 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="Low-Quality Image Enhancement & Evaluation System / 低质量图像重建与评价系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  python main.py demo              Full demo / 完整演示流程
  python main.py eval              Batch evaluation / 批量评测
  python main.py train             Train model / 训练模型
  python main.py enhance img.jpg   Single image enhancement / 单图增强
  python main.py lang zh           Switch to Chinese / 切换到中文
  python main.py lang en           Switch to English / 切换到英文
        """)
    parser.add_argument("command", nargs="?",
                        choices=["demo", "eval", "train", "enhance", "lang"],
                        default="demo",
                        help="Run mode / 运行模式 (default: demo)")
    parser.add_argument("path", nargs="?",
                        help="Image path (for enhance) or language code (for lang) / 图像路径或语言代码")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo()
    elif args.command == "eval":
        cmd_eval()
    elif args.command == "train":
        cmd_train()
    elif args.command == "enhance":
        if not args.path:
            print(f"{T.t('enhance_need_path')}")
            print(f"{T.t('enhance_usage')}")
            sys.exit(1)
        cmd_enhance(args.path)
    elif args.command == "lang":
        # Switch language / 切换语言
        if args.path in ("zh", "en"):
            # Update config.py's LANG variable / 更新 config.py 的 LANG 变量
            config_path = os.path.join(BASE_DIR, "config.py")
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = content.replace(
                f'LANG = "zh"', f'LANG = "{args.path}"'
            ).replace(
                f'LANG = "en"', f'LANG = "{args.path}"'
            )
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            lang_name = "中文" if args.path == "zh" else "English"
            print(f"{T.t('lang_switch').replace('中文', lang_name).replace('English', lang_name)}")
            print(f"  (LANG={args.path} written to config.py)")
        else:
            print("Usage: python main.py lang [zh|en]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
