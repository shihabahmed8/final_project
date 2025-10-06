

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import datetime

def make_df_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    order_cols = ["sample_id"]

    if "result_id" in df.columns:                   
        order_cols.append("result_id")

    elif "scan_date" in df.columns:                 
        df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
        order_cols.append("scan_date")

    else:                                           
        df["_ord"] = np.arange(len(df))
        order_cols.append("_ord")

    out = (df.sort_values(order_cols)
             .drop_duplicates(subset="sample_id", keep="last")
             .reset_index(drop=True))

    return out.drop(columns=["_ord"], errors="ignore")



ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

assert (SRC / "db_utils.py").exists(), f"db_utils.py not found in {SRC}"

import importlib, os, io, hashlib
import streamlit as st
import plotly.express as px

import db_utils
importlib.reload(db_utils)

import pipeline
importlib.reload(pipeline)
from pipeline import process_and_insert



BASE = ROOT
import sqlite3

DB_FILE = BASE / "db" / "produce.db"

@st.cache_data(show_spinner=False)
def load_df(ver: int) -> pd.DataFrame:
    with sqlite3.connect(str(DB_FILE)) as conn:
        q = """
        SELECT
            s.sample_id,
            s.item_name,
            s.scan_date,
            s.image_path,
            r.result_id,
            r.quality_class,
            r.confidence,
            r.freshness_index
        FROM Produce_Samples s
        LEFT JOIN Quality_Results r
            ON r.sample_id = s.sample_id
        ORDER BY
            s.sample_id DESC,
            COALESCE(r.result_id, -1) DESC
        """
        df = pd.read_sql_query(q, conn)

    if "scan_date" in df.columns:
        df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    return df


# ----------------- [STEP 1] Model & Inference setup -----------------
MODELS_DIR = BASE / "models"
MODEL_PATH  = MODELS_DIR / "best_efficientnet_b0.pth"
CLASS_PATH  = MODELS_DIR / "class_names.json"

import json, torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

DEVICE = "cpu"

try:
    torch.set_num_threads(1)
except Exception:
    pass


with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

@st.cache_resource
def _load_model_and_tfms():
    mdl = models.efficientnet_b0(weights=None)
    in_features = mdl.classifier[1].in_features
    mdl.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    mdl.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return mdl, tfm

@st.cache_data(show_spinner=False)
def _predict_from_bytes(img_bytes: bytes, tta: bool=False, topk: int=3):
    
    model, infer_tf = _load_model_and_tfms()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
   
    img.thumbnail((1024, 1024))

    def _run(pil):
        x = infer_tf(pil).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            y = model(x)
        p = torch.softmax(y, dim=1).squeeze(0).cpu().numpy()
        return p

    p = _run(img)
    if tta:
        p = np.maximum(p, _run(ImageOps.mirror(img))) 

    idx = int(np.argmax(p))
    top = sorted([(CLASS_NAMES[i], float(p[i])) for i in range(len(CLASS_NAMES))],
                 key=lambda kv: kv[1], reverse=True)[:topk]
    return CLASS_NAMES[idx], top

@torch.no_grad()
def _predict_tensor(x, tta=False):
    
    model, _ = _load_model_and_tfms()
    y1 = model(x)
    if tta:
        y2 = model(torch.flip(x, dims=[-1]))
        y  = (y1 + y2) / 2
    else:
        y  = y1
    p = torch.softmax(y, dim=1).squeeze(0)
    idx = int(p.argmax().item())
    return idx, p.cpu().numpy()

def predict_pil(img: Image.Image, tta=False, topk=3):
    
    buf = io.BytesIO()
   
    img.convert("RGB").save(buf, format="PNG")
    label, top = _predict_from_bytes(buf.getvalue(), tta=tta, topk=topk)
    return label, top
# ----------------- [END STEP 1] -----------------


# ----------------- UI -----------------
st.set_page_config(page_title="Produce Quality Dashboard", layout="wide")
st.title(" Produce Quality Dashboard")

with st.sidebar.expander("Debug", expanded=False):
    st.caption(f"DB used by db_utils: {db_utils.DB_FILE}")
    st.caption(f"db_utils file: {db_utils.__file__}")
    st.caption(f"has insert_quality? {hasattr(db_utils, 'insert_quality')}")
    st.caption(f"has insert_shelf? {hasattr(db_utils, 'insert_shelf')}")

st.sidebar.subheader("Add a new scan")
uploaded = st.sidebar.file_uploader("Upload JPEG/PNG", type=["jpg","jpeg","png"], key="file_upl")
item = st.sidebar.text_input("Item name", value="Banana", key="item_name").strip()

# ====== Main area capture section (repositioned) ======
st.markdown("### Capture / Camera")
cap_col1, cap_col2, cap_col3 = st.columns([2,1,1])
with cap_col1:
    use_browser_cam = st.toggle("Use browser camera", value=False, help="Use Streamlit's camera widget (may need Chrome/Edge)")
with cap_col2:
    opencv_btn = st.button("Capture via OpenCV", help="Single frame using system camera (fallback if browser camera fails)")
with cap_col3:
    clear_cam = st.button("Clear capture")

camera_image = None
if use_browser_cam:
    camera_image = st.camera_input("Take a picture", key="camera_input_main")

# OpenCV fallback capture
if opencv_btn:
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        cap.release()
        if ok:
            import cv2 as _cv2
            _ok, buf = _cv2.imencode('.png', frame)
            if _ok:
                st.session_state['opencv_capture'] = buf.tobytes()
                st.success("OpenCV frame captured.")
            else:
                st.error("Failed to encode frame.")
        else:
            st.error("Could not read from camera (OpenCV). Close other apps using the camera.")
    except Exception as e:
        st.error(f"OpenCV capture error: {e}")

if clear_cam:
    st.session_state.pop('opencv_capture', None)
    st.experimental_rerun()

opencv_bytes = st.session_state.get('opencv_capture')

# Determine active input content in priority order: browser cam > opencv fallback > upload
input_content = None
input_name = None
use_camera = False  # keep variable used later for saving flow
if camera_image is not None:
    input_content = camera_image.getvalue()
    input_name = f"browsercam_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    use_camera = True
elif opencv_bytes:
    input_content = opencv_bytes
    input_name = f"opencv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    use_camera = True
elif uploaded is not None:
    input_content = uploaded.read()
    input_name = uploaded.name


# --- Step 3: sidebar controls (TTA + Top-K) ---
use_tta = st.sidebar.checkbox("Use TTA (flip)", value=False)
topk = st.sidebar.slider(
    "Top-K",
    min_value=1,
    max_value=len(CLASS_NAMES),
    value=min(3, len(CLASS_NAMES)),
    step=1,
)


if "data_ver" not in st.session_state:
    st.session_state["data_ver"] = 0



def save_uploaded(file, item_name):
    data_dir = BASE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    content = file.read()
    h = hashlib.sha1(content).hexdigest()[:12]
    ext = os.path.splitext(file.name)[1].lower() or ".jpg"
    safe_item = (item_name or "item").strip().replace(" ", "_")
    out_path = data_dir / f"{safe_item}_{h}{ext}"
    with open(out_path, "wb") as f:
        f.write(content)
    return out_path, content



if input_content and item:
    # If camera (browser or opencv), save deterministic name
    if use_camera:
        data_dir = BASE / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(input_content).hexdigest()[:12]
        ext = ".png"
        out_path = data_dir / f"{item}_{h}{ext}"
        with open(out_path, "wb") as f:
            f.write(input_content)
        content = input_content
    else:
        if uploaded is not None:
            uploaded.seek(0)
            out_path, content = save_uploaded(uploaded, item)
        else:
            content = None
            out_path = None

    
    with st.spinner("Running prediction…"):
        label, top = _predict_from_bytes(content, tta=use_tta, topk=topk)

    
    st.subheader("Prediction")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(Image.open(io.BytesIO(content)), caption=item, use_container_width=True)
    with c2:
        st.metric("Class", label)
        st.table(pd.DataFrame({
            "class": [k for k, _ in top],
            "prob":  [f"{v*100:.2f}%" for _, v in top]
        }))

    # --- persist to DB ---
    precomputed = {
        "label": label,
        "top": [(k, float(v)) for k, v in top],
    }
    res = process_and_insert(str(out_path), item_name=item, precomputed=precomputed)

    st.sidebar.success(
        f'Inserted sample_id={res["sample_id"]} | class={res["quality_class"]} | FI={res["freshness_index"]}'
    )

    
    st.session_state["data_ver"] += 1



st.subheader("Filters")
with st.container():
    conn = sqlite3.connect(str(DB_FILE))
    df_items = pd.read_sql_query("SELECT DISTINCT item_name FROM Produce_Samples ORDER BY item_name", conn)
    item_opts = ["(All)"] + df_items["item_name"].dropna().tolist()
    sel_item = st.selectbox("Item name", item_opts, index=0)
    if sel_item == "(All)":
        q = """
        SELECT s.sample_id, s.item_name, s.scan_date, s.image_path,
               r.quality_class, r.confidence, r.freshness_index
        FROM Produce_Samples s
        LEFT JOIN Quality_Results r ON r.sample_id = s.sample_id
        ORDER BY s.sample_id DESC
        """
        df = pd.read_sql_query(q, conn)
    else:
        q = """
        SELECT s.sample_id, s.item_name, s.scan_date, s.image_path,
               r.quality_class, r.confidence, r.freshness_index
        FROM Produce_Samples s
        LEFT JOIN Quality_Results r ON r.sample_id = s.sample_id
        WHERE s.item_name = ?
        ORDER BY s.sample_id DESC
        """
        df = pd.read_sql_query(q, conn, params=[sel_item])
    conn.close()

left, right = st.columns(2)
with left:
    st.subheader("Quality Class Distribution")
    if len(df):
        fig = px.pie(df, names="quality_class", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

with right:
    st.subheader("Average Freshness Index")
    if len(df) and "freshness_index" in df:
        st.metric("Average FI", f"{df['freshness_index'].mean():.2f}")
    else:
        st.metric("Average FI", "—")



st.subheader("Recent Scans")


df_all = load_df(st.session_state["data_ver"])

if len(df_all):
    df_latest = make_df_latest(df_all)
else:
    df_latest = pd.DataFrame()

if len(df_latest):
    cols_show = ["sample_id", "item_name", "scan_date", "quality_class", "confidence", "freshness_index"]
    cols_show = [c for c in cols_show if c in df_latest.columns]
    st.dataframe(df_latest[cols_show], use_container_width=True, height=300)
else:
    st.info("No scans yet.")



st.subheader("Gallery")

if len(df_latest):
    cols = st.columns(4)
    for i, (_, row) in enumerate(df_latest.head(24).iterrows()):
        with cols[i % 4]:
            img_path = None
            if "image_path" in row.index and pd.notna(row["image_path"]):
                raw = str(row["image_path"]).strip()
                if raw:
                    p = Path(raw).expanduser()
                    img_path = p if p.is_absolute() else (BASE / raw)

            if img_path and img_path.exists():
                st.image(str(img_path), use_container_width=True)
                st.caption(
                    f"#{row.get('sample_id','-')} • {row.get('item_name','-')} — "
                    f"{row.get('quality_class','-')} | FI={row.get('freshness_index','-')}"
                )
            elif img_path:
                st.caption(
                    f"(image missing on disk) — "
                    f"#{row.get('sample_id','-')} • {row.get('item_name','-')}"
                )
            else:
                st.caption("(no image_path)")
else:
    st.info("No images yet.")



st.sidebar.subheader("Delete saved fruits")
del_item = st.sidebar.selectbox("Select item to delete", ["(None)"] + (df_items["item_name"].dropna().tolist() if len(df_items) else []))
del_files = st.sidebar.checkbox("Also remove image files from disk", value=False)
confirm = st.sidebar.text_input("Type DELETE to confirm", value="")
if st.sidebar.button("Delete selected item") and del_item != "(None)" and confirm == "DELETE":
    with sqlite3.connect(str(DB_FILE)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT sample_id, image_path FROM Produce_Samples WHERE item_name=?", (del_item,))
        rows = cur.fetchall()
    db_utils.delete_item(del_item)
    if del_files:
        for _, ipath in rows:
            p = (BASE / ipath) if not str(ipath).startswith(str(BASE)) else Path(ipath)
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
    st.sidebar.success(f"Deleted all '{del_item}' samples.")
   
    st.session_state["data_ver"] += 1



import shutil

st.sidebar.subheader("Admin / Danger zone")

with st.sidebar.expander("Reset all data", expanded=False):
    try:
        conn = sqlite3.connect(str(DB_FILE))
        counts = pd.read_sql_query("""
            SELECT 'Produce_Samples' AS table_name, COUNT(*) AS cnt FROM Produce_Samples
            UNION ALL
            SELECT 'Quality_Results' AS table_name, COUNT(*) AS cnt FROM Quality_Results
        """, conn)
        conn.close()
        st.caption(counts.to_string(index=False))
    except Exception as e:
        st.caption(f"(info) {e}")

    also_remove_images = st.checkbox("Also remove image files from disk (data/)",
                                     value=False, key="wipe_images_all")

    if st.button("⚠️ Reset database", type="secondary"):
        try:
            conn = sqlite3.connect(str(DB_FILE))
            cur = conn.cursor()
            cur.execute("DELETE FROM Quality_Results;")
            cur.execute("DELETE FROM Produce_Samples;")
            cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('Produce_Samples','Quality_Results');")
            conn.commit()
            cur.execute("VACUUM;")
            conn.commit()
            conn.close()

            removed = 0
            if also_remove_images:
                data_dir = BASE / "data"
                if data_dir.exists():
                    for p in data_dir.iterdir():
                        try:
                            if p.is_file():
                                p.unlink()
                                removed += 1
                            elif p.is_dir():
                                shutil.rmtree(p, ignore_errors=True)
                        except Exception:
                            pass

            st.sidebar.success(f"Database cleared. Removed {removed} image file(s).")
            st.session_state["data_ver"] += 1
        except Exception as e:
            st.sidebar.error(f"Reset failed: {e}")
