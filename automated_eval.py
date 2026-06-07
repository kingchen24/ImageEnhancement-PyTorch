# ============================================================
# Automated Evaluation Platform / 自动化评测平台
# Features / 功能: Batch processing + Quality assessment +
#          Comparison visualization + Bilingual HTML/MD/JSON reports
#          批量图像处理 + 质量评估 + 对比可视化 + 双语HTML/MD/JSON报告
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
from visualizer import Visualizer, L as VL


# ==================== Bilingual Labels / 双语标签 ====================

class L:
    """i18n labels for reports / 报告双语标签"""

    _DICT = {
        # HTML/MD report / 报告标题
        "report_title": {
            "zh": "图像质量增强评测报告",
            "en": "Image Quality Enhancement Evaluation Report"
        },
        "summary_title": {
            "zh": "总体指标",
            "en": "Summary"
        },
        "detail_title": {
            "zh": "详细结果",
            "en": "Detailed Results"
        },
        "generated_at": {
            "zh": "生成时间",
            "en": "Generated at"
        },
        "avg_psnr_gain": {
            "zh": "PSNR 平均提升",
            "en": "Avg PSNR Gain"
        },
        "avg_ssim_gain": {
            "zh": "SSIM 平均提升",
            "en": "Avg SSIM Gain"
        },
        "images_processed": {
            "zh": "处理图像数",
            "en": "Images Processed"
        },
        "avg_time": {
            "zh": "平均处理时间",
            "en": "Avg Processing Time"
        },
        "image_name": {
            "zh": "图像",
            "en": "Image"
        },
        "degraded_psnr": {
            "zh": "退化 PSNR",
            "en": "Degraded PSNR"
        },
        "enhanced_psnr": {
            "zh": "增强 PSNR",
            "en": "Enhanced PSNR"
        },
        "degraded_ssim": {
            "zh": "退化 SSIM",
            "en": "Degraded SSIM"
        },
        "enhanced_ssim": {
            "zh": "增强 SSIM",
            "en": "Enhanced SSIM"
        },
        "psnr_gain_col": {
            "zh": "PSNR 提升",
            "en": "PSNR Gain"
        },
        "ssim_gain_col": {
            "zh": "SSIM 提升",
            "en": "SSIM Gain"
        },
        "metric": {
            "zh": "指标",
            "en": "Metric"
        },
        "degraded_val": {
            "zh": "退化图像",
            "en": "Degraded"
        },
        "enhanced_val": {
            "zh": "增强图像",
            "en": "Enhanced"
        },
        "gain": {
            "zh": "提升",
            "en": "Gain"
        },
        "error_col": {
            "zh": "错误",
            "en": "Error"
        },
        "total": {
            "zh": "总计",
            "en": "Total"
        },
        "batch_header": {
            "zh": "Batch Processing",
            "en": "Batch Processing"
        },
        "images": {
            "zh": "images",
            "en": "images"
        },
        "degradation_configs": {
            "zh": "Degradation configs",
            "en": "Degradation configs"
        },
        "enhancement_method": {
            "zh": "Enhancement method",
            "en": "Enhancement method"
        },
        "type": {
            "zh": "types",
            "en": "types"
        },
        "processing": {
            "zh": "Processing",
            "en": "Processing"
        },
    }

    @classmethod
    def t(cls, key: str) -> str:
        return cls._DICT.get(key, {}).get(LANG, key)


class AutomatedEvaluator:
    """Automated Evaluation Platform / 自动化评测平台"""

    def __init__(self, model: Optional = None, device: str = 'cpu'):
        self.enhancer = ImageEnhancement(model, device)
        self.evaluator = ImageQualityMetrics()
        self.results = []

    # -------------------- Single Image / 单图处理 --------------------
    def process_single(self, img_path: str, degradation_config: dict,
                       enhance_method: str = "dl",
                       traditional_tasks: list = None) -> Dict:
        """Process single image & record all intermediate results
        处理单张图像并记录所有中间结果"""
        img = imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": f"Failed to read {img_path}"}

        name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Degradation / 退化
        degraded, deg_meta = ImageDegradation.apply_degradation_pipeline(
            img, degradation_config
        )

        # 2. Enhancement / 增强
        t_start = time.time()
        if enhance_method == "dl" and self.enhancer.model is not None:
            enhanced = self.enhancer.enhance_dl(degraded)
        elif enhance_method == "traditional" and traditional_tasks:
            enhanced = ImageEnhancement.enhance_pipeline(degraded, traditional_tasks)
        else:
            enhanced = degraded  # No enhancement / 无增强
        t_elapsed = time.time() - t_start

        # 3. Evaluation / 评估
        metrics_degraded = self.evaluator.evaluate_all(img, degraded)
        metrics_enhanced = self.evaluator.evaluate_all(img, enhanced)

        # 4. Compute improvement percentages / 计算提升百分比
        improvements = {}
        for m in metrics_degraded:
            if m in metrics_enhanced and m not in ("lpips",):
                v_before = metrics_degraded.get(m, 0)
                v_after = metrics_enhanced.get(m, 0)
                if v_before != 0:
                    improvements[f"{m}_gain"] = round(
                        (v_after - v_before) / abs(v_before) * 100, 2
                    )
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

    # -------------------- Batch Processing / 批量处理 --------------------
    def process_batch(self, input_dir: str,
                      degradation_configs: List[Dict] = None,
                      enhance_method: str = "dl",
                      traditional_tasks: list = None) -> List[Dict]:
        """Batch process & evaluate multiple images / 批量处理并评估多张图像"""
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
        print(f"  {L.t('batch_header')}: {len(files)} {L.t('images')}")
        print(f"  {L.t('degradation_configs')}: {len(degradation_configs)} {L.t('type')}")
        print(f"  {L.t('enhancement_method')}: {enhance_method}")
        print(f"{'='*60}\n")

        total = len(files) * len(degradation_configs)
        count = 0

        for fname in files:
            path = os.path.join(input_dir, fname)
            for dcfg in degradation_configs:
                count += 1
                print(f"[{count}/{total}] {L.t('processing')} {fname} ...", end=" ")
                result = self.process_single(path, dcfg, enhance_method, traditional_tasks)
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                else:
                    psnr = result["metrics_enhanced"].get("psnr", "N/A")
                    print(f"PSNR={psnr:.2f}dB")

        return self.results

    # -------------------- Summary / 汇总统计 --------------------
    def summary(self) -> Dict:
        """Generate summary statistics / 生成汇总统计"""
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

        # Improvement percentages / 改善百分比
        for r in valid:
            for k, v in r.get("improvements", {}).items():
                if k not in summary["improvements_avg"]:
                    summary["improvements_avg"][k] = []
                summary["improvements_avg"][k].append(v)

        for k in summary["improvements_avg"]:
            summary["improvements_avg"][k] = round(
                np.mean(summary["improvements_avg"][k]), 2
            )

        # Avg processing time / 平均处理时间
        summary["avg_processing_time"] = round(
            np.mean([r["processing_time"] for r in valid]), 3
        )

        return summary

    # -------------------- Report Generation / 报告生成 --------------------
    def generate_report(self, output_dir: str = None,
                        format: str = "html") -> str:
        """Generate evaluation report / 生成评测报告"""
        output_dir = output_dir or RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        if format == "json":
            return self._generate_json(output_dir)
        elif format == "markdown":
            return self._generate_markdown(output_dir)
        else:
            return self._generate_html(output_dir)

    # ---- HTML Report / HTML 报告 ----
    def _generate_html(self, output_dir: str) -> str:
        """Generate bilingual HTML report / 生成双语HTML报告"""
        summary = self.summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{timestamp}.html")

        # Build data rows / 构建数据行
        rows = ""
        for r in self.results:
            if "error" in r:
                rows += f"<tr><td colspan='7' class='error'>{r['error']}</td></tr>"
                continue
            md = r["metrics_degraded"]
            me = r["metrics_enhanced"]
            imp = r.get("improvements", {})
            psnr_g = imp.get("psnr_gain", "?")
            ssim_g = imp.get("ssim_gain", "?")

            # Color-coded gains / 颜色编码提升幅度
            psnr_color = "gain-positive" if isinstance(psnr_g, (int, float)) and psnr_g > 0 else "gain-negative"
            ssim_color = "gain-positive" if isinstance(ssim_g, (int, float)) and ssim_g > 0 else "gain-negative"

            rows += f"""
            <tr>
              <td class='img-name'>{r['name']}</td>
              <td>{md.get('psnr','?'):.2f}</td>
              <td>{me.get('psnr','?'):.2f}</td>
              <td>{md.get('ssim','?'):.4f}</td>
              <td>{me.get('ssim','?'):.4f}</td>
              <td class='{psnr_color}'>{psnr_g}%</td>
              <td class='{ssim_color}'>{ssim_g}%</td>
            </tr>"""

        # Summary cards / 汇总卡片
        imp_avg = summary.get("improvements_avg", {})
        psnr_gain = imp_avg.get("psnr_gain", "N/A")
        ssim_gain = imp_avg.get("ssim_gain", "N/A")

        html = f"""<!DOCTYPE html>
<html lang="{LANG}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{L.t('report_title')}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif;
    background: linear-gradient(135deg, #f0f4ff 0%, #f8fafc 50%, #f0fdf4 100%);
    color: #1e293b;
    padding: 40px 20px;
    min-height: 100vh;
  }}
  .container {{ max-width: 1300px; margin:0 auto; }}

  /* Header / 标题区 */
  .header {{
    text-align: center;
    padding: 35px 30px;
    background: linear-gradient(135deg, #1a73e8, #0d9488);
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(26,115,232,0.2);
  }}
  .header h1 {{ font-size: 2em; margin-bottom: 8px; letter-spacing: -0.5px; }}
  .header .subtitle {{ font-size: 0.95em; opacity: 0.9; }}

  /* Summary Cards / 汇总卡片 */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 18px;
    margin-bottom: 30px;
  }}
  .card {{
    background: white;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid #f1f5f9;
  }}
  .card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.1); }}
  .card .card-value {{ font-size: 2.4em; font-weight: 800; margin-bottom: 6px; }}
  .card .card-label {{ font-size: 0.9em; color: #64748b; font-weight: 500; }}
  .card.green .card-value {{ color: #0d9488; }}
  .card.blue .card-value {{ color: #1a73e8; }}
  .card.purple .card-value {{ color: #8b5cf6; }}
  .card.amber .card-value {{ color: #f59e0b; }}

  /* Section / 分区 */
  .section {{
    background: white;
    border-radius: 16px;
    padding: 28px 30px;
    margin-bottom: 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.05);
    border: 1px solid #f1f5f9;
  }}
  .section h2 {{
    color: #1e293b;
    font-size: 1.35em;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section h2 .icon {{ font-size: 1.2em; }}

  /* Table / 表格 */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.93em; }}
  thead th {{
    background: #f8fafc;
    padding: 14px 16px;
    text-align: center;
    font-weight: 700;
    color: #475569;
    border-bottom: 2px solid #e2e8f0;
    white-space: nowrap;
    font-size: 0.88em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  tbody td {{
    padding: 12px 16px;
    text-align: center;
    border-bottom: 1px solid #f1f5f9;
  }}
  tbody tr:hover {{ background: #f8fafc; }}
  .img-name {{ font-weight: 600; color: #1a73e8; font-family: 'SF Mono', 'Consolas', monospace; font-size: 0.9em; }}
  .gain-positive {{ color: #0d9488; font-weight: 700; }}
  .gain-negative {{ color: #e74c3c; font-weight: 700; }}
  .error {{ color: #e74c3c; font-style: italic; }}

  /* Chart images / 图表嵌入 */
  .chart-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
    margin-top: 20px;
  }}
  .chart-row img {{
    width: 100%;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #f1f5f9;
  }}

  /* Footer / 页脚 */
  .footer {{
    text-align: center;
    padding: 24px;
    color: #94a3b8;
    font-size: 0.88em;
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>{L.t('report_title')}</h1>
    <div class="subtitle">{L.t('generated_at')}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  </div>

  <!-- Summary Cards / 汇总卡片 -->
  <div class="cards">
    <div class="card green">
      <div class="card-value">{psnr_gain}%</div>
      <div class="card-label">{L.t('avg_psnr_gain')}</div>
    </div>
    <div class="card blue">
      <div class="card-value">{ssim_gain}%</div>
      <div class="card-label">{L.t('avg_ssim_gain')}</div>
    </div>
    <div class="card purple">
      <div class="card-value">{summary.get('total_images', 0)}</div>
      <div class="card-label">{L.t('images_processed')}</div>
    </div>
    <div class="card amber">
      <div class="card-value">{summary.get('avg_processing_time', 'N/A')}s</div>
      <div class="card-label">{L.t('avg_time')}</div>
    </div>
  </div>

  <!-- Summary Table / 汇总表 -->
  <div class="section">
    <h2><span class="icon">📊</span> {L.t('summary_title')}</h2>
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>{L.t('metric')}</th>
          <th>{L.t('degraded_val')}</th>
          <th>{L.t('enhanced_val')}</th>
          <th>{L.t('gain')}</th>
        </tr>
      </thead>
      <tbody>
"""
        improvements = summary.get("improvements_avg", {})
        for k in summary.get("degraded_avg", {}):
            d_val = summary["degraded_avg"][k]
            e_val = summary["enhanced_avg"].get(k, "-")
            gain = improvements.get(f"{k}_gain", "-")
            gain_str = f"{gain}%" if isinstance(gain, (int, float)) else gain
            html += f"""
        <tr>
          <td style="font-weight:600;">{k.upper()}</td>
          <td>{d_val:.4f}</td>
          <td>{e_val:.4f}</td>
          <td class="gain-positive">{gain_str}</td>
        </tr>"""

        html += f"""
      </tbody>
    </table>
    </div>
  </div>

  <!-- Detailed Results / 详细结果 -->
  <div class="section">
    <h2><span class="icon">📋</span> {L.t('detail_title')}</h2>
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>{L.t('image_name')}</th>
          <th>{L.t('degraded_psnr')}</th>
          <th>{L.t('enhanced_psnr')}</th>
          <th>{L.t('degraded_ssim')}</th>
          <th>{L.t('enhanced_ssim')}</th>
          <th>{L.t('psnr_gain_col')}</th>
          <th>{L.t('ssim_gain_col')}</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    </div>
  </div>

  <!-- Charts Section / 可视化图表 -->
  <div class="section">
    <h2><span class="icon">📈</span> Visualization / 可视化</h2>
    <div class="chart-row">
"""

        # Embed chart images if they exist / 嵌入存在的图表
        chart_files = [
            ("dashboard.png", "Dashboard / 仪表盘"),
            ("metrics_comparison.png", "Metrics Comparison / 指标对比"),
            ("radar_chart.png", "Radar Chart / 雷达图"),
            ("gain_bars.png", "Gain Bars / 提升幅度"),
            ("per_image_metrics.png", "Per-Image Metrics / 逐图指标"),
            ("time_distribution.png", "Time Distribution / 时间分布"),
        ]
        for fname, caption in chart_files:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                rel = os.path.relpath(fpath, output_dir)
                html += f"""      <div>
        <img src="{rel}" alt="{caption}">
        <p style="text-align:center;color:#64748b;margin-top:8px;font-size:0.9em;">{caption}</p>
      </div>
"""

        html += """    </div>
  </div>

  <div class="footer">
    Generated by ImageEnhancement-PyTorch | MSFE-Net Evaluation Platform<br>
    Powered by PyTorch + OpenCV + NumPy
  </div>

</div>
</body>
</html>"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML report saved to: {path}")
        return path

    # ---- JSON Report / JSON 报告 ----
    def _generate_json(self, output_dir: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{timestamp}.json")
        report = {
            "timestamp": timestamp,
            "language": LANG,
            "summary": self.summary(),
            "details": self.results,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON report saved to: {path}")
        return path

    # ---- Markdown Report / Markdown 报告 ----
    def _generate_markdown(self, output_dir: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{timestamp}.md")
        summary = self.summary()

        md = f"# {L.t('report_title')}\n\n"
        md += f"**{L.t('generated_at')}**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary / 汇总
        md += f"## {L.t('summary_title')}\n\n"
        md += f"| {L.t('metric')} | {L.t('degraded_val')} | {L.t('enhanced_val')} | {L.t('gain')} |\n"
        md += "|------|----------|----------|------|\n"

        improvements = summary.get("improvements_avg", {})
        for k in summary.get("degraded_avg", {}):
            d_val = summary["degraded_avg"][k]
            e_val = summary["enhanced_avg"].get(k, "-")
            gain = improvements.get(f"{k}_gain", "-")
            gain_str = f"{gain}%" if isinstance(gain, (int, float)) else gain
            md += f"| {k.upper()} | {d_val:.4f} | {e_val:.4f} | {gain_str} |\n"

        md += f"\n- **{L.t('images_processed')}**: {summary.get('total_images', 0)}\n"
        md += f"- **{L.t('avg_time')}**: {summary.get('avg_processing_time', 'N/A')}s\n"

        # Details / 详细结果
        md += f"\n## {L.t('detail_title')}\n\n"
        md += f"| {L.t('image_name')} | {L.t('degraded_psnr')} | {L.t('enhanced_psnr')} | "
        md += f"{L.t('degraded_ssim')} | {L.t('enhanced_ssim')} | "
        md += f"{L.t('psnr_gain_col')} | {L.t('ssim_gain_col')} |\n"
        md += "|------|----------|----------|----------|----------|----------|----------|\n"

        for r in self.results:
            if "error" in r:
                md += f"| {r.get('name','?')} | Error | | | | | |\n"
                continue
            md_ = r["metrics_degraded"]
            me_ = r["metrics_enhanced"]
            imp_ = r.get("improvements", {})
            md += (f"| {r['name']} | {md_.get('psnr',0):.2f} | {me_.get('psnr',0):.2f} | "
                   f"{md_.get('ssim',0):.4f} | {me_.get('ssim',0):.4f} | "
                   f"{imp_.get('psnr_gain',0)}% | {imp_.get('ssim_gain',0)}% |\n")

        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"Markdown report saved to: {path}")
        return path

    # -------------------- Comparison Images / 对比图保存 --------------------
    def save_comparisons(self, output_dir: str = None) -> List[str]:
        """Save original/degraded/enhanced comparison images
        保存原始/退化/增强三者对比图"""
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
            # Concatenate: Original | Degraded | Enhanced / 拼接: 原图 | 退化 | 增强
            comparison = np.hstack([img, degraded, enhanced])

            # Add labels / 添加文字标注
            font = cv2.FONT_HERSHEY_SIMPLEX
            labels_en = ["Original", "Degraded", "Enhanced"]
            labels_zh = ["原始", "退化", "增强"]
            labels = labels_zh if LANG == "zh" else labels_en

            for i, label in enumerate(labels):
                x = int(w * i + w * 0.05)
                # Background rect for readability / 背景矩形提高可读性
                (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
                cv2.rectangle(comparison, (x - 5, 10 - th - 5),
                             (x + tw + 5, 30), (0, 0, 0), -1)
                cv2.putText(comparison, label, (x, 28), font, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            out_path = os.path.join(output_dir, f"cmp_{r['name']}.jpg")
            imwrite(out_path, comparison)
            saved.append(out_path)

        if saved:
            print(f"Saved {len(saved)} comparison images to {output_dir}")
        return saved
