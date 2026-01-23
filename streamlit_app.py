# =====================================================
# HERBASMARTAI - FINAL SIDANG VERSION
# =====================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import yaml
import time
import cv2
import numpy as np
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# LOAD YAML DATA (EXTERNAL KNOWLEDGE BASE)
# =====================================================
@st.cache_data
def load_yaml():
    with open("data/data-baru.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

yaml_data   = load_yaml()
CLASS_NAMES = yaml_data["names"]
CLASS_INFO  = yaml_data["info"]

# =====================================================
# LOAD YOLO MODEL (CACHE RESOURCE)
# =====================================================
@st.cache_resource
def load_model(path):
    return YOLO(path)

MODEL_PATHS = {
    "YOLOv8 Nano (Cepat)": "models/bestnano.pt",
    "YOLOv8 Small (Seimbang)": "models/bestsmall.pt",
    "YOLOv8 Medium (Akurat)": "models/yolo_medium.pt",
}

# =====================================================
# GENERATE COLOR PER CLASS
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
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 6, y), color, -1)
    cv2.putText(
        img, text, (x + 3, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

# =====================================================
# IMAGE DETECTION FUNCTION
# =====================================================
def detect_image(model, image: Image.Image):
    start = time.time()

    img_np = np.array(image)
    results = model.predict(img_np, verbose=False)

    infer_time = time.time() - start
    output = img_np.copy()
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
            cls_id = int(r.boxes.cls[i])
            conf   = float(r.boxes.conf[i]) * 100

            if cls_id >= len(CLASS_NAMES):
                continue

            name  = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]
            info  = CLASS_INFO.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {})
            })

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            draw_label(output, f"{name} {conf:.1f}%", x1, y1, color)

    return output, detections, infer_time

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌿 MENU")
menu = st.sidebar.radio(
    "Navigasi",
    ["Beranda", "Deteksi Gambar", "Deteksi Webcam", "Rekomendasi Manfaat"]
)

# =====================================================
# BERANDA
# =====================================================
if menu == "Beranda":
    st.title("🌿 HerbaSmartAI")
    st.info(
        "Sistem deteksi daun herbal berbasis **YOLOv8** "
        "yang mampu menampilkan nama daun, confidence, manfaat, dan resep tradisional."
    )

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "Deteksi Gambar":
    st.header("📷 Deteksi Daun Herbal dari Gambar")

    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])

    uploaded = st.file_uploader("Upload gambar daun", ["jpg", "jpeg", "png"])
    camera   = st.camera_input("Atau ambil foto langsung")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        st.image(image, caption="Gambar Input", use_container_width=True)

        with st.spinner("🔍 Mendeteksi..."):
            output_img, detections, infer_time = detect_image(model, image)

        st.image(output_img, caption="Hasil Deteksi", use_container_width=True)

        st.success(
            f"⏱ Waktu inferensi: {infer_time:.3f} detik | "
            f"🌿 Jumlah daun terdeteksi: {len(detections)}"
        )

        for d in detections:
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                st.markdown("**🧪 Kandungan:**")
                st.write(", ".join(d["components"]))

                st.markdown("**💊 Manfaat:**")
                for b in d["benefits"]:
                    st.write(f"- {b}")

                if d["recipes"]:
                    st.markdown("### 🍵 Resep Tradisional")
                    st.markdown("**Bahan:**")
                    for b in d["recipes"].get("bahan", []):
                        st.write(f"- {b}")

                    st.markdown("**Cara Pembuatan:**")
                    for i, step in enumerate(d["recipes"].get("cara", []), 1):
                        st.write(f"{i}. {step}")

# =====================================================
# DETEKSI WEBCAM (LOKAL ONLY)
# =====================================================
elif menu == "Deteksi Webcam":
    st.header("🎥 Deteksi Real-Time Webcam")

    if "STREAMLIT_CLOUD" in os.environ:
        st.warning("🚫 Webcam tidak didukung di Streamlit Cloud.")
        st.stop()

    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])

    run = st.checkbox("▶️ Aktifkan Webcam")
    frame_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                    cls_id = int(r.boxes.cls[i])

                    if cls_id >= len(CLASS_NAMES):
                        continue

                    name  = CLASS_NAMES[cls_id]
                    color = CLASS_COLORS[cls_id]

                    cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                    draw_label(rgb, name, x1, y1, color)

            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

        cap.release()

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "Rekomendasi Manfaat":
    st.header("💊 Cari Daun Berdasarkan Manfaat")

    query = st.text_input("Masukkan gejala (contoh: batuk, demam)")

    if query:
        found = False
        for leaf, info in CLASS_INFO.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                found = True
                with st.expander(f"🌿 {leaf}"):
                    st.markdown("**🧪 Kandungan:**")
                    st.write(", ".join(info.get("components", [])))

                    st.markdown("**💊 Manfaat:**")
                    for b in info.get("benefits", []):
                        st.write(f"- {b}")

                    if info.get("recipes"):
                        st.markdown("### 🍵 Resep Tradisional")
                        st.markdown("**Bahan:**")
                        for b in info["recipes"].get("bahan", []):
                            st.write(f"- {b}")

                        st.markdown("**Cara Pembuatan:**")
                        for i, step in enumerate(info["recipes"].get("cara", []), 1):
                            st.write(f"{i}. {step}")

        if not found:
            st.warning("❌ Tidak ditemukan daun dengan manfaat tersebut.")
