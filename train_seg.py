import os
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
# 规避 Windows 上 torch 初始化时对无效 DLL 目录的添加
os.environ.pop("CONDA_PREFIX", None)
try:
    _orig_add_dll = os.add_dll_directory  # type: ignore[attr-defined]
    def _noop_add_dll(_path: str):
        return None
    os.add_dll_directory = _noop_add_dll  # type: ignore[attr-defined]
except Exception:
    pass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_ROOT = Path(r"D:/桌面桌面/Restore/dataset")
MODEL_OUT = Path(r"D:/桌面桌面/Restore/model/seg_model.onnx")
INPUT_SIZE = 512
NUM_CLASSES = 4
EPOCHS = 5
BATCH_SIZE = 2
LR = 1e-3


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


class SegDataset(Dataset):
    def __init__(self, split: str):
        split_file = DATA_ROOT / "splits" / f"{split}.txt"
        self.ids = [x.strip() for x in open(split_file, "r", encoding="utf-8").read().splitlines() if x.strip()]
        self.img_dir = DATA_ROOT / "images"
        self.msk_dir = DATA_ROOT / "masks"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        base = self.ids[idx]
        img = imread_unicode(self.img_dir / f"{base}.png", cv2.IMREAD_COLOR)
        msk = imread_unicode(self.msk_dir / f"{base}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C,H,W
        img_t = torch.from_numpy(img)
        msk_t = torch.from_numpy(msk.astype(np.int64))
        return img_t, msk_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        c4 = torch.cat([u4, d4], dim=1)
        d5 = self.dec4(c4)
        u3 = self.up3(d5)
        c3 = torch.cat([u3, d3], dim=1)
        d6 = self.dec3(c3)
        u2 = self.up2(d6)
        c2 = torch.cat([u2, d2], dim=1)
        d7 = self.dec2(c2)
        u1 = self.up1(d7)
        c1 = torch.cat([u1, d1], dim=1)
        d8 = self.dec1(c1)
        out = self.head(d8)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for imgs, msks in loader:
        imgs = imgs.to(device)
        msks = msks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, msks)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, msks in loader:
            imgs = imgs.to(device)
            msks = msks.to(device)
            logits = model(imgs)
            loss = criterion(logits, msks)
            total += float(loss.item())
    return total / max(1, len(loader))


def export_onnx(model, device, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )


def main():
    device = torch.device("cpu")
    train_ds = SegDataset("train")
    val_ds = SegDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(in_ch=3, num_classes=NUM_CLASSES, base=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = 1e9
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(f"epoch {epoch}/{EPOCHS} - train_loss {tr:.4f} val_loss {va:.4f} time {dt:.1f}s")
        if va < best_val:
            best_val = va
            export_onnx(model, device, MODEL_OUT)
            print(f"exported best model to {MODEL_OUT}")

    if not MODEL_OUT.exists():
        export_onnx(model, device, MODEL_OUT)
        print(f"exported final model to {MODEL_OUT}")


if __name__ == "__main__":
    main()


