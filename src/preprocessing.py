from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def basic_mask_and_crop(pil_img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    rgb = np.array(pil_img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    fg = ((S > 15) | (V < 240)).astype("uint8")*255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    ys, xs = np.where(fg > 0)
    if len(xs)==0 or len(ys)==0:
        return pil_img, fg
    x1, x2 = max(0, xs.min()-5), min(rgb.shape[1], xs.max()+5)
    y1, y2 = max(0, ys.min()-5), min(rgb.shape[0], ys.max()+5)
    crop = rgb[y1:y2, x1:x2]
    return Image.fromarray(crop), fg
