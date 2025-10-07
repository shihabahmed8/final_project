

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

# ================= UI & State Setup =================
st.set_page_config(page_title="Produce Quality Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.markdown("## Produce Quality Dashboard")
st.caption("Auto prediction (upload / camera) with freshness heuristics.")

# Compact metric font sizing
st.markdown(
    """
    <style>
    /* Reduce metric label & value sizes */
    [data-testid='stMetric'] div:first-child { font-size:0.68rem; letter-spacing:.5px; text-transform:uppercase; opacity:.75; }
    [data-testid='stMetricValue'] { font-size:1.25rem; font-weight:600; }
    /* If value is long, constrain and add ellipsis */
    [data-testid='stMetricValue'] { max-width:110px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True
)

if "data_ver" not in st.session_state:
    st.session_state["data_ver"] = 0
for k in ["last_pred","last_image_bytes","last_image_path","last_opencv"]:
    st.session_state.setdefault(k, None)

# Create tabs
tab_predict, tab_analytics, tab_gallery, tab_admin, tab_debug = st.tabs([
    "🔮 Predict", "📈 Analytics", "🖼 Gallery", "🛠 Admin", "🐞 Debug"
])

with tab_predict:
    import plotly.graph_objects as go
    c1, c2, c3, c4 = st.columns([2,1.2,1,1])
    with c1:
        item = st.text_input("Item name", value="Banana").strip()
    with c2:
        topk = st.slider("Top-K", 1, len(CLASS_NAMES), min(3, len(CLASS_NAMES)))
    with c3:
        use_tta = st.toggle("TTA", value=False)
    with c4:
        capture_mode = st.selectbox("Source", ["Upload","Browser Cam","OpenCV"], index=0)

    left_col, right_col = st.columns([1,1.4])
    input_content = None
    if capture_mode == "Upload":
        up = left_col.file_uploader("Upload JPEG/PNG", type=["jpg","jpeg","png"], key="upl_main")
        if up: input_content = up.read()
    elif capture_mode == "Browser Cam":
        cam_img = left_col.camera_input("Browser Camera", key="browser_cam")
        if cam_img: input_content = cam_img.getvalue()
    else:
        if left_col.button("🎥 Capture via OpenCV", key="opencv_btn"):
            try:
                import cv2
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                ok, frame = cap.read(); cap.release()
                if ok:
                    import cv2 as _cv2
                    _ok, buf = _cv2.imencode('.png', frame)
                    if _ok:
                        st.session_state['last_opencv'] = buf.tobytes()
                        input_content = st.session_state['last_opencv']
                        left_col.success("Captured.")
                    else:
                        left_col.error("Encode failed.")
                else:
                    left_col.error("Camera busy / inaccessible.")
            except Exception as e:
                left_col.error(f"OpenCV error: {e}")
        if input_content is None:
            input_content = st.session_state.get('last_opencv')

    new_hash = None
    if input_content:
        new_hash = hashlib.sha1(input_content).hexdigest()
        left_col.image(input_content, caption=f"Input {new_hash[:8]}", use_container_width=True)

    trigger = False
    if input_content and item:
        last_path = st.session_state.get('last_image_path')
        last_hash = None
        if last_path:
            try: last_hash = Path(last_path).stem.split('_')[-1]
            except Exception: pass
        if (new_hash and new_hash != last_hash) or st.session_state.get('last_pred') is None:
            trigger = True

    if trigger:
        data_dir = BASE / 'data'; data_dir.mkdir(exist_ok=True, parents=True)
        out_path = data_dir / f"{item}_{new_hash[:12]}.png"
        with open(out_path,'wb') as f: f.write(input_content)
        with st.spinner("Predicting..."):
            label, top = _predict_from_bytes(input_content, tta=use_tta, topk=topk)
        precomputed = {"label": label, "top": [(k,float(v)) for k,v in top]}
        res = process_and_insert(str(out_path), item_name=item, precomputed=precomputed)
        st.session_state.update({
            'data_ver': st.session_state['data_ver'] + 1,
            'last_pred': (label, top, res),
            'last_image_bytes': input_content,
            'last_image_path': str(out_path)
        })

    if st.session_state.get('last_pred'):
        label, top, res = st.session_state['last_pred']
        m1, m2, m3, m4 = right_col.columns(4)
        m1.metric("Class", label)
        m2.metric("FI", f"{res['freshness_index']:.2f}")
        m3.metric("Conf", f"{top[0][1]*100:.1f}%")
        m4.metric("ID", res['sample_id'])
        prob_df = pd.DataFrame({"class":[k for k,_ in top], "prob":[float(v) for _,v in top]}).sort_values('prob')
        fig = go.Figure()
        fig.add_bar(x=prob_df.prob*100, y=prob_df['class'], orientation='h', marker=dict(color='rgba(16,185,129,0.75)'))
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10), xaxis=dict(range=[0,100], title='Probability (%)'))
        right_col.plotly_chart(fig, use_container_width=True)
        # Show probabilities with a % suffix for readability
        right_col.dataframe(
            prob_df.sort_values('prob', ascending=False)
                   .assign(prob_pct=lambda d: (d.prob*100).round(2).astype(str) + '%')
                   .drop(columns=['prob'])
                   .rename(columns={'prob_pct':'probability'}),
            use_container_width=True,
            height=170
        )
        with right_col.expander("Details", expanded=False):
            st.json(res)

with tab_analytics:
    st.markdown("### Dataset / Prediction Metrics")
    conn = sqlite3.connect(str(DB_FILE))
    df_items = pd.read_sql_query("SELECT DISTINCT item_name FROM Produce_Samples ORDER BY item_name", conn)
    item_opts = ["(All)"] + df_items["item_name"].dropna().tolist()
    filt_col1, filt_col2 = st.columns([1,2])
    with filt_col1:
        sel_item = st.selectbox("Filter item", item_opts, index=0)
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
    st.markdown("### Recent Scans")
    df_all = load_df(st.session_state["data_ver"])
    df_latest = make_df_latest(df_all) if len(df_all) else pd.DataFrame()
    if len(df_latest):
        cols_show = [c for c in ["sample_id","item_name","scan_date","quality_class","confidence","freshness_index"] if c in df_latest.columns]
        st.dataframe(df_latest[cols_show], use_container_width=True, height=320)
    else:
        st.info("No scans yet.")

with tab_gallery:
    st.markdown("### Gallery")
    df_all = load_df(st.session_state["data_ver"])
    df_latest = make_df_latest(df_all) if len(df_all) else pd.DataFrame()
    if len(df_latest):
        cols = st.columns(6)
        for i, (_, row) in enumerate(df_latest.head(36).iterrows()):
            with cols[i % 6]:
                img_path = None
                if "image_path" in row.index and pd.notna(row["image_path"]):
                    raw = str(row["image_path"]).strip(); p = Path(raw)
                    img_path = p if p.is_absolute() else (BASE / raw)
                if img_path and img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                    st.caption(f"#{row.get('sample_id','-')} • {row.get('item_name','-')} — {row.get('quality_class','-')}")
                else:
                    st.caption(f"#{row.get('sample_id','-')} (missing image)")
    else:
        st.info("No images yet.")

with tab_admin:
    st.markdown("### Admin & Maintenance")
    conn = sqlite3.connect(str(DB_FILE))
    df_items = pd.read_sql_query("SELECT DISTINCT item_name FROM Produce_Samples ORDER BY item_name", conn)
    conn.close()
    del_cols = st.columns([2,1,1,1])
    with del_cols[0]:
        del_item = st.selectbox("Delete item", ["(None)"] + df_items["item_name"].dropna().tolist())
    with del_cols[1]:
        del_files = st.checkbox("Remove images", value=False)
    with del_cols[2]:
        confirm = st.text_input("Type DELETE", value="")
    with del_cols[3]:
        if st.button("Delete", type="secondary") and del_item != "(None)" and confirm == "DELETE":
            with sqlite3.connect(str(DB_FILE)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT sample_id, image_path FROM Produce_Samples WHERE item_name=?", (del_item,))
                rows = cur.fetchall()
            db_utils.delete_item(del_item)
            removed = 0
            if del_files:
                for _, ipath in rows:
                    p = (BASE / ipath) if not str(ipath).startswith(str(BASE)) else Path(ipath)
                    try:
                        if p.exists():
                            p.unlink(); removed += 1
                    except Exception:
                        pass
            st.success(f"Deleted '{del_item}' (images removed={removed}).")
            st.session_state["data_ver"] += 1
    with st.expander("Reset ALL data", expanded=False):
        also_remove_images = st.checkbox("Also remove images (data/)", value=False)
        if st.button("⚠️ Reset database", type="secondary"):
            try:
                with sqlite3.connect(str(DB_FILE)) as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM Quality_Results;")
                    cur.execute("DELETE FROM Produce_Samples;")
                    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('Produce_Samples','Quality_Results');")
                    conn.commit(); cur.execute("VACUUM;"); conn.commit()
                removed = 0
                if also_remove_images:
                    data_dir = BASE / "data"
                    if data_dir.exists():
                        for p in data_dir.iterdir():
                            try:
                                if p.is_file(): p.unlink(); removed += 1
                            except Exception: pass
                st.warning(f"Database cleared. Removed {removed} image(s).")
                st.session_state["data_ver"] += 1
            except Exception as e:
                st.error(f"Reset failed: {e}")

with tab_debug:
    st.markdown("### Debug Info")
    st.caption(f"DB File: {db_utils.DB_FILE}")
    st.caption(f"db_utils path: {db_utils.__file__}")
    st.caption(f"Has insert_quality? {hasattr(db_utils, 'insert_quality')}")
    st.caption(f"Has insert_shelf? {hasattr(db_utils, 'insert_shelf')}")
    st.caption(f"Data version: {st.session_state['data_ver']}")
    st.caption(f"Last image saved: {st.session_state.get('last_image_path')}")



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



import shutil  # retained for admin operations (used in admin tab)
