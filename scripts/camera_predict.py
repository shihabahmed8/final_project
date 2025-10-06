"""Live camera prediction using OpenCV and your EfficientNet-B0 model.

Usage (from project root, venv active):
    python scripts/camera_predict.py                # default camera index 0
    python scripts/camera_predict.py --camera 1     # alternate camera
    python scripts/camera_predict.py --device cuda  # if you have GPU

Keys while window focused:
    q / ESC  Quit
    c        Capture current frame to data/ and show path
    s        Save frame AND insert prediction into database (uses pipeline.process_and_insert)

The database insert needs the existing SQLite DB at db/produce.db and will write image under data/.
"""
from __future__ import annotations
import argparse, time, json, sys, os, math
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------- Paths & Setup ----------------
ROOT = Path(__file__).resolve().parents[1]  # project root
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_efficientnet_b0.pth"
CLASS_PATH = MODELS_DIR / "class_names.json"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pipeline  # for process_and_insert
    from pipeline import process_and_insert  # type: ignore
    PIPELINE_AVAILABLE = True
except Exception:
    PIPELINE_AVAILABLE = False

# ---------------- Model Loading ----------------
_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model_cache = None
_class_names = None

def load_model(device: torch.device):
    global _model_cache, _class_names
    if _model_cache is not None:
        return _model_cache, _class_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_PATH}")

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        _class_names = json.load(f)

    mdl = models.efficientnet_b0(weights=None)
    in_features = mdl.classifier[1].in_features
    mdl.classifier[1] = nn.Linear(in_features, len(_class_names))
    state = torch.load(MODEL_PATH, map_location="cpu")

    # Accept either direct state_dict or torch.save-style object
    if isinstance(state, dict) and any(k.startswith("classifier") for k in state.keys()):
        mdl.load_state_dict(state)
    elif isinstance(state, dict) and "model" in state:
        mdl.load_state_dict(state["model"])
    else:
        raise RuntimeError("Unrecognized model checkpoint format")

    mdl.eval().to(device)
    _model_cache = mdl
    return mdl, _class_names

# ---------------- Prediction ----------------
@torch.inference_mode()
def predict_frame(frame_bgr: np.ndarray, device: torch.device, topk: int = 3):
    mdl, names = load_model(device)
    # Convert BGR (OpenCV) to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = _tfms(pil).unsqueeze(0).to(device)
    y = mdl(x)
    probs = torch.softmax(y, dim=1).squeeze(0).cpu().numpy()
    idx_sorted = np.argsort(-probs)[:topk]
    top = [(names[i], float(probs[i])) for i in idx_sorted]
    return top

# ---------------- Drawing Helpers ----------------
def draw_overlay(frame: np.ndarray, top_preds, fps: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 70 + 22*len(top_preds)), (0,0,0), -1)
    alpha = 0.45
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    for i, (cls, p) in enumerate(top_preds):
        bar_w = int((p) * 220)
        y0 = 40 + i*22
        cv2.putText(frame, f"{cls} {p*100:5.1f}%", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (160, y0-12), (160+bar_w, y0-4), (0,180,255), -1)
    return frame

# ---------------- Saving & DB Insert ----------------
def save_frame(frame_bgr: np.ndarray, prefix: str = "capture") -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    out_path = DATA_DIR / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(out_path), frame_bgr)
    return out_path


def insert_prediction(image_path: Path, top_preds):
    if not PIPELINE_AVAILABLE:
        print("[WARN] pipeline/process_and_insert not available; skipping DB insertion.")
        return None
    label, prob = top_preds[0]
    precomputed = {"label": label, "top": top_preds}
    try:
        res = process_and_insert(str(image_path), item_name=label, precomputed=precomputed)
        print(f"[DB] Inserted sample_id={res['sample_id']} label={res['quality_class']} FI={res['freshness_index']}")
        return res
    except Exception as e:
        print(f"[DB] Insert failed: {e}")
        return None

# ---------------- Main Loop ----------------
def main():
    ap = argparse.ArgumentParser(description="Live camera prediction with EfficientNet-B0")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="Device to run inference")
    ap.add_argument("--topk", type=int, default=3, help="Top-K classes to display")
    ap.add_argument("--width", type=int, default=640, help="Resize width for display (keeps aspect)")
    ap.add_argument("--no-db", action="store_true", help="Disable DB insertion even if pipeline present")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")

    try:
        load_model(device)  # warm load
        print(f"[Info] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[FATAL] Could not load model: {e}")
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open camera index {args.camera}")
        return
    print("[Info] Press 'q' or ESC to quit | 'c' capture | 's' save & insert (DB)")

    t_prev = time.time()
    fps = 0.0
    last_pred = []
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame; retrying...")
            time.sleep(0.05)
            continue
        frame_count += 1

        # Optionally resize for speed
        h, w = frame.shape[:2]
        if w > args.width:
            new_h = int(h * (args.width / w))
            frame = cv2.resize(frame, (args.width, new_h))

        # Predict every 2 frames or so to save CPU
        if frame_count % 2 == 0:
            try:
                last_pred = predict_frame(frame, device, topk=args.topk)
            except Exception as e:
                last_pred = []
                cv2.putText(frame, f"Predict err: {e}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        now = time.time()
        dt = now - t_prev
        if dt > 0:
            fps = 1.0 / dt
        t_prev = now

        if last_pred:
            frame = draw_overlay(frame, last_pred, fps)
        else:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Live Prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('c'):
            path = save_frame(frame)
            print(f"[Saved] {path}")
        elif key == ord('s'):
            path = save_frame(frame, prefix="pred")
            if last_pred and not args.no_db:
                insert_prediction(path, last_pred)
            else:
                print(f"[Saved] {path} (DB skipped)")

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Exited.")


if __name__ == "__main__":
    main()
