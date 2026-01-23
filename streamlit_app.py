import streamlit as st
import yaml
import time
import os
import numpy as np
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

IS_CLOUD = "STREAMLIT_CLOUD" in os.environ

# =====================================================
# LOAD YAML
# =====================================================
@st.cache_data
def load_yaml():
    with open("data-baru.yaml", "r") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data.get("names", [])
CLASS_INFO = yaml_data.get("info", {})

# =====================================================
# LOAD YOLO
# =====================================================
@st.cache_resource
def load_model(path):
    from ultralytics import YOLO
    return YOLO(path)

MODEL_PATHS = {
    "YOLOv11 Nano": "models/bestnano.pt",
    "YOLOv11 Small": "models/bestsmall.pt",
    "YOLOv11 Medium": "best_m.pt"
}

# =====================================================
# WARNA PER KELAS
# =====================================================
np.random.seed(42)
CLASS_COLORS = [
    tuple(int(c) for c in np.random.randint(0, 255, 3))
    for _ in CLASS_NAMES
]

# =====================================================
# DRAW LABEL
# =====================================================
def draw_label(img, text, x, y, color):
    import cv2
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y-h-8), (x+w+6, y), color, -1)
    cv2.putText(img, text, (x+3, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# =====================================================
# DETEKSI
# =====================================================
def run_detection(image_np, model):
    import cv2
    start = time.time()
    results = model.predict(image_np, verbose=False)
    infer_time = time.time() - start

    output = image_np.copy()
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for i, box in enumerate(r.boxes.xyxy):
            x1,y1,x2,y2 = map(int, box)
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100

            if cls_id >= len(CLASS_NAMES):
                continue

            name = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]
            info = CLASS_INFO.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf,2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {})
            })

            cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
            draw_label(output,f"{name} {conf:.1f}%",x1,y1,color)

    return output, detections, infer_time

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌿 MENU")
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda","📷 Deteksi Gambar","🎥 Deteksi Webcam","💊 Rekomendasi Manfaat"]
)

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.title("🌿 HerbaSmartAI")
    st.info("Deteksi daun herbal berbasis YOLO + rekomendasi tradisional")

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal")

    model = load_model(st.selectbox("⚙️ Pilih Varian YOLO", MODEL_PATHS.values()))

    img_file = st.file_uploader("Upload gambar",["jpg","png","jpeg"])
    cam = st.camera_input("Atau ambil foto")

    if img_file or cam:
        img = Image.open(img_file or cam).convert("RGB")
        out, det, t = run_detection(np.array(img), model)

        st.image(out, use_container_width=True)
        st.success(f"⏱️ {t:.3f} detik | 📊 {len(det)} daun terdeteksi")

# =====================================================
# DETEKSI WEBCAM (FIXED)
# =====================================================
elif menu == "🎥 Deteksi Webcam":
    st.title("🎥 Deteksi Webcam")

    model = load_model(st.selectbox("⚙️ Pilih Varian YOLO", MODEL_PATHS.values()))

    if IS_CLOUD:
        st.info("📸 Mode snapshot (Cloud)")
        snap = st.camera_input("Ambil gambar")
        if snap:
            img = Image.open(snap).convert("RGB")
            out, det, _ = run_detection(np.array(img), model)
            st.image(out, use_container_width=True)
            st.info(f"📊 {len(det)} daun terdeteksi")
    else:
        import cv2
        run = st.checkbox("▶️ Aktifkan Webcam")
        frame_box = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out, det, _ = run_detection(rgb, model)
                frame_box.image(out, channels="RGB", use_container_width=True)
            cap.release()

# =====================================================
# REKOMENDASI MANFAAT (RESEP FIX)
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Cari Daun Berdasarkan Manfaat")

    q = st.text_input("Masukkan gejala (contoh: batuk, demam)")

    if q:
        found = False
        for leaf, info in CLASS_INFO.items():
            if any(q.lower() in b.lower() for b in info.get("benefits", [])):
                found = True
                with st.expander(f"🌿 {leaf}"):
                    st.write("**Kandungan:**", ", ".join(info.get("components", [])))

                    st.write("**Manfaat:**")
                    for b in info.get("benefits", []):
                        st.write(f"- {b}")

                    st.write("**Resep Tradisional:**")
                    for manfaat, resep in info.get("recipes", {}).items():
                        st.markdown(f"**{manfaat}**")
                        st.write("Bahan:")
                        for i in resep.get("ingredients", []):
                            st.write(f"• {i}")
                        st.write("Langkah:")
                        for j, step in enumerate(resep.get("steps", []),1):
                            st.write(f"{j}. {step}")
        if not found:
            st.warning("❌ Tidak ditemukan")
