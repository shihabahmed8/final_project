from pathlib import Path
from typing import Tuple
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np
import json

CANONICAL = ["Unripe", "Ripe", "Overripe", "BruisedDamaged", "HealthyLeaf"]
DISPLAY_MAP = {
    "Unripe": "Unripe",
    "Ripe": "Perfectly Ripe",
    "Overripe": "Overripe",
    "BruisedDamaged": "Bruised/Damaged",
    "HealthyLeaf": "Healthy Leaf",
}

class MobileNetV2Head(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        base = mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_ch = base.classifier[1].in_features
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        logits = self.classifier(x)
        return logits

def _tfms():
    w = MobileNet_V2_Weights.IMAGENET1K_V1
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        w.transforms().normalize,
    ])

def _load_labels(weights_dir: Path) -> list:
    p = weights_dir / "labels.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return CANONICAL

def load_model(weights_path: str|None, num_classes=5, device: str|None=None) -> tuple:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2Head(num_classes=num_classes).to(device)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            model.load_state_dict(state)
        else:
            sd = state.get("state_dict", state)
            model.load_state_dict(sd)
    model.eval()
    return model, device

def predict(pil_img: Image.Image, weights_dir: str) -> tuple[str, float, np.ndarray]:
    weights_dir = Path(weights_dir)
    weights_path = weights_dir / "produce_mnv2_torch.pt"
    labels = _load_labels(weights_dir)
    model, device = load_model(str(weights_path), num_classes=len(labels))
    x = _tfms()(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    canon = labels[idx] if idx < len(labels) else CANONICAL[idx]
    display = DISPLAY_MAP.get(canon, canon)
    return display, float(probs[idx]), probs
