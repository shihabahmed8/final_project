# -*- coding: utf-8 -*-
"""
Train MobileNetV2 on banana dataset with 5 classes.
Expected folder layout (relative to this file):
  ../data/banana/
      train/    (optional if you only use CSV)
      valid/    (optional)
      _classes.csv   -> columns: filename, freshrip, freshunrip, overripe, ripe, rotten, unripe
Images are referenced by filename column (must exist under data/banana/*).
Model is saved to ../models/banana_mnv2.pth
"""

from pathlib import Path
import os, time, math
import pandas as pd
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIG ----------------
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "train"
CSV_PATH =DATA_DIR / "_classes.csv"
    # labels file
OUT_DIR  = PROJECT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
CLASSES = ["Unripe", "Perfectly Ripe", "Overripe", "Bruised/Damaged", "Healthy Leaf"]

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
NUM_WORKERS = 2

# --------- label mapping from CSV one-hot to our 5 classes ----------
# الأولوية: rotten -> overripe -> ripe/freshrip -> unripe/freshunrip -> healthy leaf
def row_to_target(row: pd.Series) -> int:
    g = lambda k: int(row.get(k, 0))  # gracefully handle missing
    if g("rotten") == 1:
        return CLASSES.index("Bruised/Damaged")
    if g("overripe") == 1:
        return CLASSES.index("Overripe")
    if g("ripe") == 1 or g("freshrip") == 1:
        return CLASSES.index("Perfectly Ripe")
    if g("unripe") == 1 or g("freshunrip") == 1:
        return CLASSES.index("Unripe")
    return CLASSES.index("Healthy Leaf")

# ---------------- Dataset ----------------
class BananaCsvDataset(Dataset):
    def __init__(self, csv_path: Path, img_root: Path, tfm: transforms.Compose):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.tfm = tfm

        # build (path, target) list
        self.samples: List[Tuple[Path, int]] = []
        miss = 0
        for _, r in self.df.iterrows():
            fname = str(r["filename"])
            # الصورة قد تكون في data/banana/train أو valid أو test؛ نحاول إيجادها
            p = None
            for sub in ["train", "valid", "test", ""]:
                cand = (img_root / sub / fname) if sub else (img_root / fname)
                if cand.exists():
                    p = cand
                    break
            if p is None:
                miss += 1
                continue
            y = row_to_target(r)
            self.samples.append((p, y))

        print(f"[Dataset] Found {len(self.samples)} images; missing={miss}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, y

# --------------- Utils -------------------
def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

def split_dataframe(df: pd.DataFrame, valid_ratio=0.15):
    # split by rows (stratify roughly using target)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    targets = df.apply(row_to_target, axis=1)
    df = df.assign(_y=targets)

    # simple stratified-ish split
    val_idx = []
    for c in range(NUM_CLASSES):
        idx = df[df._y==c].index.tolist()
        k = max(1, int(len(idx)*valid_ratio))
        val_idx += idx[:k]
    mask = df.index.isin(val_idx)
    return df[~mask].drop(columns=["_y"]), df[mask].drop(columns=["_y"])

def build_loaders() -> Tuple[DataLoader, DataLoader]:
    assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)

    # لو عندك مجلدات train/valid جاهزة وتُريد تجاهل التقسيم، ما مشكلة
    # هنا بنقسم من CSV إذا ما وفرت مجلدات
    train_df, valid_df = split_dataframe(df, valid_ratio=0.15)

    # نكتبهم مؤقتًا (للتشخيص فقط)
    print(f"[Split] train={len(train_df)}, valid={len(valid_df)}")

    train_ds = BananaCsvDataset(csv_path=None, img_root=DATA_DIR, tfm=get_transforms(True))
    train_ds.df = train_df  # حقن DataFrame بعد التقسيم
    train_ds.__init__(csv_path=None, img_root=DATA_DIR, tfm=get_transforms(True))
    train_ds.df = train_df  # إعادة الحقن لأن __init__ يقرأ CSV

    # نعيد بناء القائمة samples يدويًا لأننا غيّرنا df
    train_ds.samples.clear()
    for _, r in train_df.iterrows():
        fname = str(r["filename"])
        p = None
        for sub in ["train", "valid", "test", ""]:
            cand = (DATA_DIR/sub/fname) if sub else (DATA_DIR/fname)
            if cand.exists():
                p = cand; break
        if p is None: 
            continue
        y = row_to_target(r)
        train_ds.samples.append((p,y))

    valid_ds = BananaCsvDataset(csv_path=None, img_root=DATA_DIR, tfm=get_transforms(False))
    valid_ds.samples.clear()
    for _, r in valid_df.iterrows():
        fname = str(r["filename"])
        p = None
        for sub in ["train", "valid", "test", ""]:
            cand = (DATA_DIR/sub/fname) if sub else (DATA_DIR/fname)
            if cand.exists():
                p = cand; break
        if p is None:
            continue
        y = row_to_target(r)
        valid_ds.samples.append((p,y))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"[Loaders] train batches={len(train_loader)}, valid batches={len(valid_loader)}")
    return train_loader, valid_loader

def build_model(num_classes=NUM_CLASSES):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_ch = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_ch, num_classes)
    return m

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

# ----------------- Training -----------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for imgs, ys in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        loss_sum += loss.item()*bs
        acc_sum  += accuracy(logits, ys)*bs
        n += bs
    return loss_sum/n, acc_sum/n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for imgs, ys in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        logits = model(imgs)
        loss = criterion(logits, ys)
        bs = imgs.size(0)
        loss_sum += loss.item()*bs
        acc_sum  += accuracy(logits, ys)*bs
        n += bs
    return loss_sum/n, acc_sum/n

def main():
    print("[Info] Project:", PROJECT)
    print("[Info] Data   :", DATA_DIR)
    print("[Info] CSV    :", CSV_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] Device :", device)

    train_loader, valid_loader = build_loaders()
    if len(train_loader.dataset) == 0:
        print("[Error] No training samples found. Check paths/CSV filenames.")
        return

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = math.inf
    best_path = OUT_DIR / "banana_mnv2.pth"

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, valid_loader, criterion, device)
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}  ({dt:.1f}s)")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "classes": CLASSES}, best_path)
            print(f"  ↳ Saved: {best_path}")

    print("[Done] Best model:", best_path)

if __name__ == "__main__":
    main()
