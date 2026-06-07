# ============================================================
# 训练流程
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import imread
from config import *
from degradation import ImageDegradation
from models import MSFENet, MSFENetLite
from models.losses import CompositeLoss


class DegradationDataset(Dataset):
    """退化图像数据集 — 从高清图生成退化-干净图像对"""

    def __init__(self, img_dir: str, patch_size: int = 128,
                 num_patches_per_img: int = 16,
                 degradation_config: dict = None):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.num_patches = num_patches_per_img
        self.deg_config = degradation_config or {
            "gaussian_noise": {"enabled": True, "std": 25},
            "gaussian_blur": {"enabled": True, "kernel_size": 7, "sigma": 1.5},
            "jpeg_compression": {"enabled": True, "quality": 20},
        }
        self.files = [f for f in os.listdir(img_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not self.files:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.files) * self.num_patches

    def __getitem__(self, idx):
        file_idx = idx // self.num_patches
        img_path = os.path.join(self.img_dir, self.files[file_idx])
        img = imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 随机裁剪
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            img = cv2.resize(img, (max(w, self.patch_size), max(h, self.patch_size)),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        clean = img[top:top + self.patch_size, left:left + self.patch_size]

        # 生成退化图
        degraded, _ = ImageDegradation.apply_degradation_pipeline(
            clean, self.deg_config
        )

        # 归一化到 [-1, 1]
        clean_t = torch.from_numpy(clean.astype(np.float32) / 255.0)
        clean_t = clean_t.permute(2, 0, 1) * 2 - 1

        degraded_t = torch.from_numpy(degraded.astype(np.float32) / 255.0)
        degraded_t = degraded_t.permute(2, 0, 1) * 2 - 1

        return degraded_t, clean_t


def train(model, train_loader, val_loader, config, device):
    """训练循环"""
    model = model.to(device)

    criterion = CompositeLoss(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["learning_rate"],
                            weight_decay=config.get("weight_decay", 1e-5))
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.get("lr_scheduler_step", 30),
                                          gamma=config.get("lr_scheduler_gamma", 0.5))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_file = open(log_path, 'w', encoding='utf-8')

    best_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "epoch": []}

    for epoch in range(1, config["num_epochs"] + 1):
        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Train]")
        for degraded, clean in pbar:
            degraded, clean = degraded.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(degraded)
            loss, loss_dict = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / len(train_loader)
        history["train_loss"].append(avg_train)
        history["epoch"].append(epoch)

        log_line = f"[Epoch {epoch:3d}] Train Loss: {avg_train:.4f}"
        print(log_line)
        log_file.write(log_line + "\n")

        # ---- 验证 ----
        if epoch % config.get("val_interval", 5) == 0 and val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
                for degraded, clean in pbar_val:
                    degraded, clean = degraded.to(device), clean.to(device)
                    pred = model(degraded)
                    loss, _ = criterion(pred, clean)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            history["val_loss"].append(avg_val)
            log_line = f"[Epoch {epoch:3d}] Val Loss:   {avg_val:.4f}"
            print(log_line)
            log_file.write(log_line + "\n")

        # ---- 保存 ----
        if epoch % config.get("save_interval", 10) == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"msfe_epoch{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train,
            }, ckpt_path)
            print(f"  -> saved: {ckpt_path}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(CHECKPOINT_DIR, "msfe_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train,
            }, best_path)

        scheduler.step()

    final_path = os.path.join(CHECKPOINT_DIR, "msfe_final.pth")
    torch.save({
        'epoch': config["num_epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train,
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Best model saved to {best_path}")

    log_file.close()
    return model, history
