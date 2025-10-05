"""Simple end-to-end pipeline with model + HSV/LAB fallback."""
from pathlib import Path
from typing import Dict
import sys

from PIL import Image  # type: ignore

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import db_utils
import preprocessing

QUALITY_LABELS = [
    "Unripe",
    "Perfectly Ripe",
    "Overripe",
    "Bruised/Damaged",
    "Healthy Leaf",
]

def _freshness_from_brightness(pil_img) -> float:
    img = pil_img.convert("L")
    vals = list(img.getdata())
    mean = sum(vals) / max(1, len(vals))
    return max(0.0, min(10.0, (mean / 255.0) * 10.0))

def _compute_color_stats(pil_img):
    import numpy as np, cv2
    rgb = np.array(pil_img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype("float32")
    S = hsv[:, :, 1].astype("float32")
    V = hsv[:, :, 2].astype("float32")
    green  = ((45 <= H) & (H <= 85) & (S > 60) & (V > 50)).astype("uint8")
    yellow = ((20 <= H) & (H <= 40) & (S > 60) & (V > 70)).astype("uint8")
    red    = ((((H <= 10) | (H >= 170)) & (S > 60) & (V > 50))).astype("uint8")
    def frac(mask): 
        total = float(mask.size) if mask.size else 1.0
        return float((mask > 0).sum()) / total
    return {"p_green": frac(green), "p_yellow": frac(yellow), "p_red": frac(red)}

def classify_ripeness(pil_img, item_name: str):
    import numpy as np, cv2
    rgb = np.array(pil_img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    H, S, V = hsv[:,:,0].astype("float32"), hsv[:,:,1].astype("float32"), hsv[:,:,2].astype("float32")
    L = lab[:,:,0].astype("float32")
    fg = (((S > 15) | (V < 240))).astype("uint8")
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    total = float(fg.sum()) if fg.sum() > 0 else 1.0
    dark_any  = (((V < 80) | (L < 70))).astype("uint8")
    very_dark = (((V < 55) | (L < 55))).astype("uint8")
    dark_fg   = cv2.bitwise_and(dark_any,  dark_any,  mask=fg)
    vdark_fg  = cv2.bitwise_and(very_dark, very_dark, mask=fg)
    p_dark  = float(dark_fg.sum())  / total
    p_vdark = float(vdark_fg.sum()) / total
    largest_comp = 0.0
    if vdark_fg.sum() > 0:
        num, labels = cv2.connectedComponents(vdark_fg, connectivity=8)
        if num > 1:
            areas = [int((labels == i).sum()) for i in range(1, num)]
            largest_comp = max(areas) / total if areas else 0.0
    stats = _compute_color_stats(pil_img)
    p_green, p_yellow, p_red = stats["p_green"], stats["p_yellow"], stats["p_red"]
    name = (item_name or "").strip().lower()
    if name in ("banana", "plantain"):
        damaged = (p_vdark >= 0.08) or (largest_comp >= 0.06) or (p_dark >= 0.18)
    else:
        damaged = (p_dark >= 0.12) or (p_vdark >= 0.06) or (largest_comp >= 0.06)
    if damaged:
        quality = "Bruised/Damaged"
    else:
        if p_green >= max(p_yellow + p_red, 0.35):
            quality = "Unripe"
        elif (p_yellow + p_red) >= 0.45:
            quality = "Ripe"
        else:
            quality = "Ripe" if (p_yellow + p_red) > p_green else "Unripe"
    fresh = 10.0
    fresh -= 6.5 * p_dark
    fresh -= 8.0  * p_vdark
    fresh -= 5.0  * largest_comp
    fresh += 1.5  * (p_yellow + 0.8 * p_red)
    fresh = max(0.0, min(10.0, fresh))
    conf = 0.70 if quality != "Bruised/Damaged" else min(0.95, 0.6 + 0.6 * max(p_vdark, largest_comp))
    return quality, round(fresh, 2), conf


import streamlit as st
# pipeline.py
from typing import Dict, Optional, List, Tuple
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


FI_MAP = {
    "freshripe":    9.4,
    "freshunripe":  9.7,
    "overripe":     8.5,
    "ripe":         9.0,
    "rotten":       6.0,
    "unripe":       9.9,
}

def _compute_fi(label: str) -> float:
    return float(FI_MAP.get(label, 9.0))

def process_and_insert(
    image_path: str,
    item_name: str,
    precomputed: Optional[Dict] = None,   
) -> Dict:
    """
    precomputed = {
        "label": str,                   # التصنيف النهائي (مثلاً "unripe")
        "top":   List[Tuple[str,float]] # [(class, prob), ...] أعلى احتمال في index 0
    }
    """
    from db_utils import insert_sample, insert_quality, insert_shelf

    
    sample_id, created = insert_sample(image_path=image_path, item_name=item_name)

    if not precomputed or "label" not in precomputed:
        raise RuntimeError(
            "process_and_insert expects precomputed={'label':..., 'top':...} from app.py"
        )

    label: str = str(precomputed["label"])
    top: List[Tuple[str, float]] = precomputed.get("top", [])
    confidence: float = float(top[0][1]) if top else 0.0

   
    fi: float = _compute_fi(label)

    
    predicted_days = max(1, int(round(0.9 * fi)))
    optimal_temp = 12.0 if (item_name or "").lower() in ("banana", "bananas") else 4.0
    decay_rate = round(1.0 / (predicted_days + 1e-6), 3)

   
    insert_quality(sample_id, label, confidence, fi)
    insert_shelf(sample_id, predicted_days, optimal_temp, decay_rate)

    
    return {
        "sample_id": sample_id,
        "quality_class": label,
        "freshness_index": float(fi),
        "confidence": float(confidence),
        "predicted_storage_days": predicted_days,
        "optimal_temp_C": optimal_temp,
        "decay_rate": decay_rate,
    }

