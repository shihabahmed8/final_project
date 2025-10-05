from pathlib import Path
import json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_OUT = OUT_DIR / "produce_mnv2_torch.pt"
LABELS_OUT = OUT_DIR / "labels.json"

CLASS_NAMES = ["Unripe", "Ripe", "Overripe", "BruisedDamaged", "HealthyLeaf"]

train_tfms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])
val_tfms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

assert [c for c,_ in sorted(train_ds.class_to_idx.items(), key=lambda x:x[1])] == CLASS_NAMES, \
    f"Folder classes must be: {CLASS_NAMES}. Found: {train_ds.class_to_idx}"

def make_loader(ds, batch, balanced=False):
    if not balanced:
        return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    from collections import Counter
    counts = Counter([y for _,y in ds.samples])
    weights = [len(ds)/counts[i] for i in range(len(CLASS_NAMES))]
    sample_weights = [weights[y] for _,y in ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=2, pin_memory=True)

train_dl = make_loader(train_ds, batch=32, balanced=True)
val_dl   = make_loader(val_ds,   batch=64, balanced=False)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

def run_epoch(loader, train):
    if train: model.train()
    else: model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for xb,yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
        loss_sum += float(loss)*xb.size(0)
        pred = out.argmax(1)
        correct += (pred==yb).sum().item()
        total += xb.size(0)
    return loss_sum/total, correct/total

best_acc, best_state = 0.0, None
EPOCHS = 20
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_dl, True)
    va_loss, va_acc = run_epoch(val_dl,   False)
    scheduler.step()
    print(f"[{ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
    if va_acc > best_acc:
        best_acc = va_acc
        best_state = model.state_dict().copy()

state = {"model_state": best_state if best_state is not None else model.state_dict(),
         "class_names": CLASS_NAMES}
torch.save(state, WEIGHTS_OUT)
LABELS_OUT.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
print(f"Saved: {WEIGHTS_OUT}")
