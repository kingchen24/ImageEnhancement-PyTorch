# ============================================================
# 工具函数: 安全文件I/O (处理中文路径)
# ============================================================

import cv2
import numpy as np
import os


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """安全读取图像, 支持中文路径"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


def imwrite(path: str, img: np.ndarray, params=None) -> bool:
    """安全写入图像, 支持中文路径"""
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.png':
        succ, data = cv2.imencode('.png', img, params or [])
    elif ext in ('.jpg', '.jpeg'):
        succ, data = cv2.imencode('.jpg', img, params or [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == '.bmp':
        succ, data = cv2.imencode('.bmp', img, params or [])
    elif ext == '.webp':
        succ, data = cv2.imencode('.webp', img, params or [])
    elif ext == '.tiff':
        succ, data = cv2.imencode('.tiff', img, params or [])
    else:
        succ, data = cv2.imencode('.png', img, params or [])

    if succ:
        data.tofile(path)
    return succ
