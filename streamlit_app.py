import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
import time
import cv2
import numpy as np
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Deteksi Daun Herbal",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS STYLING
# =====================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .block-container { padding: 3rem 2rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #059669 0%, #047857 100%) !important; }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: white !important; font-weight: 700; }
    .card { background: white !important; padding: 2rem; border-radius: 24px; margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.15); color: #111827; }
    h1, h2, h3 { color: white !important; }
    .card h1, .card h2, .card h3 { color: #111827 !important; }
    .stButton > button { background: white !important; color: #047857 !important; border-radius: 12px; font-weight: 700; width: 100%; }
    .hero-title { font-size: 3.5rem; font-weight: bold; color: white; text-align: center; margin-bottom: 2rem; text-shadow: 3px 3px 0 #047857; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
@st.cache_data
def load_yaml():
    if not os.path.exists("data-baru.yaml"):
        st.error("File 'data-baru.yaml' tidak ditemukan!")
        return {"names": [], "info": {}}
    with open("data-baru.yaml", "r") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
class_names = yaml_data.get("names", [])
class_info = yaml_data.get("info", {})

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

MODEL_PATHS = {
    "Best Model": "best.pt",
    "YOLOv8n (Nano)": "models/bestnano.pt",
    "YOLOv8s (Small)": "models/bestsmall.pt"
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def detect_and_draw(model, image):
    start = time.time()
    results = model.predict(image, verbose=False)
    infer_time = time.time() - start
    detections = []
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes:
            for i, box in enumerate(r.boxes.xyxy):
                cls_id = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i]) * 100
                name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"
                info = class_info.get(name, {})

                detections.append({
                    "name": name,
                    "confidence": round(conf, 2),
                    "components": info.get("components", []),
                    "benefits": info.get("benefits", []),
                    "recipes": info.get("recipes", {}),
                    "gambar": info.get("gambar", "")
                })

                x1, y1, x2, y2 = [int(c) for c in box]
                draw.rectangle([x1, y1, x2, y2], outline="white", width=4)
                draw.text((x1, y1 - 25), f"{name} {conf:.1f}%", fill="white", font=font)

    return image, detections, infer_time

# =====================================================
# MAIN APP
# =====================================================
with st.sidebar:
    st.markdown("### 🌿 MENU NAVIGASI")
    menu = st.radio("", ["🏠 Beranda", "🔍 Deteksi Gambar", "📸 Deteksi Webcam", "💊 Rekomendasi Manfaat"])

if menu == "🏠 Beranda":
    st.markdown('<h1 class="hero-title">🌿 DETEKSI DAUN HERBAL</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h2 style="text-align: center;">Deep Learning Herbal Identifier</h2>
        <p>Aplikasi ini menggunakan YOLOv8 untuk mengenali jenis daun herbal dan memberikan informasi manfaat serta resep tradisional secara instan.</p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "🔍 Deteksi Gambar":
    st.title("🔍 Deteksi Gambar")
    choice = st.selectbox("Pilih Model", list(MODEL_PATHS.keys()))
    model = load_model(MODEL_PATHS[choice])
    
    if model:
        uploaded = st.file_uploader("Upload atau Ambil Foto", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            res_img, detections, t = detect_and_draw(model, img)
            st.image(res_img, use_container_width=True)
            st.success(f"Terdeteksi dalam {t:.3f} detik")
            
            for d in detections:
                with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if d['gambar']: st.image(d['gambar'])
                    with col_b:
                        st.write("**Manfaat:**", ", ".join(d['benefits']))
                        st.write("**Kandungan:**", ", ".join(d['components']))
    else:
        st.error("File model (.pt) tidak ditemukan di folder!")

elif menu == "📸 Deteksi Webcam":
    st.title("📸 Real-time Detection")
    st.warning("Gunakan tombol 'Stop' pada jendela browser untuk menghentikan akses kamera.")
    choice = st.selectbox("Pilih Model", list(MODEL_PATHS.keys()), key="wc")
    model = load_model(MODEL_PATHS[choice])
    
    run = st.checkbox('Nyalakan Kamera')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)

    while run:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame, conf=0.5, verbose=False)
        res_plotted = results[0].plot()
        FRAME_WINDOW.image(res_plotted)
    else:
        cam.release()

elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Cari Berdasarkan Manfaat")
    query = st.text_input("Contoh: batuk, kolesterol, diabetes")
    if query:
        found = False
        for leaf, info in class_info.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                st.markdown(f"<div class='card'><h3>🌿 {leaf}</h3><p>{', '.join(info['benefits'])}</p></div>", unsafe_allow_html=True)
                found = True
        if not found: st.info("Tidak ditemukan hasil.")
