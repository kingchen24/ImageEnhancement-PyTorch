# ============================================================
# 自动化评测平台
# 功能: 批量图像处理 + 质量评估 + 对比可视化 + HTML报告
# ============================================================

import os
import cv2
import json
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional
from utils import imread, imwrite

from config import *
from degradation import ImageDegradation
from enhancement import ImageEnhancement
from evaluation import ImageQualityMetrics


class AutomatedEvaluator:
    """自动化评测平台"""

    def __init__(self, model: Optional = None, device: str = 'cpu'):
        self.enhancer = ImageEnhancement(model, device)
        self.evaluator = ImageQualityMetrics()
        self.results = []

    # -------------------- 单图处理 --------------------
    def process_single(self, img_path: str, degradation_config: dict,
                       enhance_method: str = "dl",
                       traditional_tasks: list = None) -> Dict:
        """处理单张图像并记录所有中间结果"""
        img = imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": f"Failed to read {img_path}"}

        name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. 退化
        degraded, deg_meta = ImageDegradation.apply_degradation_pipeline(
            img, degradation_config
        )

        # 2. 增强
        t_start = time.time()
        if enhance_method == "dl" and self.enhancer.model is not None:
            enhanced = self.enhancer.enhance_dl(degraded)
        elif enhance_method == "traditional" and traditional_tasks:
            enhanced = ImageEnhancement.enhance_pipeline(degraded, traditional_tasks)
        else:
            enhanced = degraded  # 无增强
        t_elapsed = time.time() - t_start

        # 3. 评估
        metrics_degraded = self.evaluator.evaluate_all(img, degraded)
        metrics_enhanced = self.evaluator.evaluate_all(img, enhanced)

        # 4. 计算提升百分比
        improvements = {}
        for m in metrics_degraded:
            if m in metrics_enhanced and m not in ("lpips",):
                v_before = metrics_degraded.get(m, 0)
                v_after = metrics_enhanced.get(m, 0)
                if v_before != 0:
                    improvements[f"{m}_gain"] = round((v_after - v_before) / abs(v_before) * 100, 2)
                else:
                    improvements[f"{m}_gain"] = 0

        result = {
            "name": name,
            "path": img_path,
            "shape": img.shape,
            "degradation": deg_meta,
            "metrics_degraded": metrics_degraded,
            "metrics_enhanced": metrics_enhanced,
            "improvements": improvements,
            "processing_time": round(t_elapsed, 3),
        }
        self.results.append(result)
        return result

    # -------------------- 批量处理 --------------------
    def process_batch(self, input_dir: str,
                      degradation_configs: List[Dict] = None,
                      enhance_method: str = "dl",
                      traditional_tasks: list = None) -> List[Dict]:
        """批量处理并评估多张图像"""
        if degradation_configs is None:
            degradation_configs = [{
                "gaussian_noise": {"enabled": True, "std": 25},
                "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
                "jpeg_compression": {"enabled": True, "quality": 20},
            }]

        self.results = []
        files = [f for f in sorted(os.listdir(input_dir))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        print(f"\n{'='*60}")
        print(f"  Batch Processing: {len(files)} images")
        print(f"  Degradation configs: {len(degradation_configs)} types")
        print(f"  Enhancement method: {enhance_method}")
        print(f"{'='*60}\n")

        total = len(files) * len(degradation_configs)
        count = 0

        for fname in files:
            path = os.path.join(input_dir, fname)
            for dcfg in degradation_configs:
                count += 1
                print(f"[{count}/{total}] Processing {fname} ...", end=" ")
                result = self.process_single(path, dcfg, enhance_method, traditional_tasks)
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                else:
                    psnr = result["metrics_enhanced"].get("psnr", "N/A")
                    print(f"PSNR={psnr:.2f}dB")

        return self.results

    # -------------------- 汇总统计 --------------------
    def summary(self) -> Dict:
        """生成汇总统计"""
        if not self.results:
            return {"error": "No results"}

        valid = [r for r in self.results if "error" not in r]
        if not valid:
            return {"error": "All results contain errors"}

        metrics_keys = ["psnr", "ssim", "ms_ssim", "vif", "fsim", "lpips"]

        summary = {
            "total_images": len(valid),
            "degraded_avg": {},
            "enhanced_avg": {},
            "improvements_avg": {},
        }

        for key in metrics_keys:
            d_vals = [r["metrics_degraded"].get(key, 0) for r in valid
                      if isinstance(r["metrics_degraded"].get(key), (int, float))]
            e_vals = [r["metrics_enhanced"].get(key, 0) for r in valid
                      if isinstance(r["metrics_enhanced"].get(key), (int, float))]
            if d_vals:
                summary["degraded_avg"][key] = round(np.mean(d_vals), 4)
            if e_vals:
                summary["enhanced_avg"][key] = round(np.mean(e_vals), 4)

        # 改善百分比
        for r in valid:
            for k, v in r.get("improvements", {}).items():
                if k not in summary["improvements_avg"]:
                    summary["improvements_avg"][k] = []
                summary["improvements_avg"][k].append(v)

        for k in summary["improvements_avg"]:
            summary["improvements_avg"][k] = round(
                np.mean(summary["improvements_avg"][k]), 2
            )

        # 平均处理时间
        summary["avg_processing_time"] = round(
            np.mean([r["processing_time"] for r in valid]), 3
        )

        return summary

    # -------------------- HTML 报告 --------------------
    def generate_report(self, output_dir: str = None,
                        format: str = "html") -> str:
        """生成评测报告"""
        output_dir = output_dir or RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        if format == "json":
            return self._generate_json(output_dir)
        elif format == "markdown":
            return self._generate_markdown(output_dir)
        else:
            return self._generate_html(output_dir)

    def _generate_html(self, output_dir: str) -> str:
        """生成HTML报告"""
        summary = self.summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{timestamp}.html")

        # 构建数据行
        rows = ""
        for r in self.results:
            if "error" in r:
                rows += f"<tr><td colspan='7'>Error: {r['error']}</td></tr>"
                continue
            md = r["metrics_degraded"]
            me = r["metrics_enhanced"]
            rows += f"""
            <tr>
              <td>{r['name']}</td>
              <td>{md.get('psnr','?'):.2f}</td>
              <td>{me.get('psnr','?'):.2f}</td>
              <td>{md.get('ssim','?'):.4f}</td>
              <td>{me.get('ssim','?'):.4f}</td>
              <td>{r.get('improvements',{}).get('psnr_gain','?')}%</td>
              <td>{r.get('improvements',{}).get('ssim_gain','?')}%</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>图像增强评测报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui, sans-serif; background:#f4f6f9; color:#333; padding:40px 20px; }}
.container {{ max-width:1200px; margin:0 auto; }}
h1 {{ color:#1a73e8; border-bottom:3px solid #1a73e8; padding-bottom:10px; margin-bottom:30px; }}
h2 {{ color:#444; margin:30px 0 15px; }}
.card {{ background:#fff; border-radius:12px; padding:25px; margin-bottom:20px; box-shadow:0 2px 12px rgba(0,0,0,.08); }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:15px; }}
.stat {{ background:linear-gradient(135deg,#667eea,#764ba2); border-radius:10px; padding:20px; color:#fff; text-align:center; }}
.stat.green {{ background:linear-gradient(135deg,#11998e,#38ef7d); }}
.stat.blue {{ background:linear-gradient(135deg,#4facfe,#00f2fe); }}
.stat .value {{ font-size:2em; font-weight:700; }}
.stat .label {{ font-size:.85em; opacity:.9; margin-top:5px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:12px 16px; text-align:center; border-bottom:1px solid #eee; }}
th {{ background:#f8f9fa; font-weight:600; color:#555; }}
tr:hover {{ background:#f0f4ff; }}
.footer {{ margin-top:40px; text-align:center; color:#999; font-size:.85em; }}
</style>
</head>
<body>
<div class="container">
  <h1>🔬 图像质量增强评测报告</h1>

  <div class="card">
    <h2>📊 总体指标</h2>
    <div class="stats">
      <div class="stat green">
        <div class="value">{summary.get('improvements_avg',{}).get('psnr_gain','N/A')}%</div>
        <div class="label">PSNR 平均提升</div>
      </div>
      <div class="stat blue">
        <div class="value">{summary.get('improvements_avg',{}).get('ssim_gain','N/A')}%</div>
        <div class="label">SSIM 平均提升</div>
      </div>
      <div class="stat">
        <div class="value">{summary['total_images']}</div>
        <div class="label">处理图像数</div>
      </div>
      <div class="stat">
        <div class="value">{summary.get('avg_processing_time','N/A')}s</div>
        <div class="label">平均处理时间</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📋 详细结果</h2>
    <table>
      <thead>
        <tr>
          <th>图像</th>
          <th>退化 PSNR</th>
          <th>增强 PSNR</th>
          <th>退化 SSIM</th>
          <th>增强 SSIM</th>
          <th>PSNR 提升</th>
          <th>SSIM 提升</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  </div>
</div>
</body>
</html>"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML report saved to: {path}")
        return path

    def _generate_json(self, output_dir: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{timestamp}.json")
        report = {
            "timestamp": timestamp,
            "summary": self.summary(),
            "details": self.results,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON report saved to: {path}")
        return path

    def _generate_markdown(self, output_dir: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{timestamp}.md")
        summary = self.summary()

        md = f"""# 图像质量增强评测报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 总体指标

| 指标 | 退化图像 | 增强图像 | 提升 |
|------|----------|----------|------|
"""
        improvements = summary.get("improvements_avg", {})
        for k in summary.get("degraded_avg", {}):
            d_val = summary["degraded_avg"][k]
            e_val = summary["enhanced_avg"].get(k, "-")
            gain = improvements.get(f"{k}_gain", "-")
            md += f"| {k.upper()} | {d_val:.4f} | {e_val:.4f} | {gain}% |\n"

        md += f"\n- 处理图像总数: {summary['total_images']}\n"
        md += f"- 平均处理时间: {summary.get('avg_processing_time', 'N/A')}s\n"

        md += "\n## 详细结果\n\n"
        md += "| 图像 | 退化PSNR | 增强PSNR | 退化SSIM | 增强SSIM | PSNR提升 | SSIM提升 |\n"
        md += "|------|----------|----------|----------|----------|----------|----------|\n"
        for r in self.results:
            if "error" in r:
                md += f"| {r.get('name','?')} | Error | | | | | |\n"
                continue
            md_ = r["metrics_degraded"]
            me_ = r["metrics_enhanced"]
            imp_ = r.get("improvements", {})
            md += f"| {r['name']} | {md_.get('psnr',0):.2f} | {me_.get('psnr',0):.2f} | {md_.get('ssim',0):.4f} | {me_.get('ssim',0):.4f} | {imp_.get('psnr_gain',0)}% | {imp_.get('ssim_gain',0)}% |\n"

        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"Markdown report saved to: {path}")
        return path

    # -------------------- 对比图保存 --------------------
    def save_comparisons(self, output_dir: str = None) -> List[str]:
        """保存原始/退化/增强三者对比图"""
        output_dir = output_dir or os.path.join(RESULTS_DIR, "comparisons")
        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for r in self.results:
            if "error" in r:
                continue
            img = imread(r["path"])
            if img is None:
                continue

            degraded, _ = ImageDegradation.apply_degradation_pipeline(
                img,
                {"gaussian_noise": {"enabled": True, "std": 25},
                 "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5}}
            )

            if self.enhancer.model is not None:
                enhanced = self.enhancer.enhance_dl(degraded)
            else:
                enhanced = degraded

            h, w = img.shape[:2]
            # 拼接: 原图 | 退化 | 增强
            comparison = np.hstack([img, degraded, enhanced])

            # 添加文字标注
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, label in enumerate(["Original", "Degraded", "Enhanced"]):
                x = int(w * i + w * 0.05)
                cv2.putText(comparison, label, (x, 30), font, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            out_path = os.path.join(output_dir, f"cmp_{r['name']}.jpg")
            imwrite(out_path, comparison)
            saved.append(out_path)

        if saved:
            print(f"Saved {len(saved)} comparison images to {output_dir}")
        return saved
