# =====================================================
# HERBASMARTAI - STREAMLIT APP (FINAL SIDANG VERSION)
# =====================================================

import streamlit as st
from PIL import Image
import yaml
import time
import os
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# CEK LINGKUNGAN (WEB / CLOUD)
# =====================================================
IS_CLOUD = "STREAMLIT_CLOUD" in os.environ

# =====================================================
# LOAD YAML DATA (ROBUST PATH)
# =====================================================
@st.cache_data
def load_yaml():
    paths = [
        "data-baru.yaml",
        "./data-baru.yaml",
        "data/data-baru.yaml"
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    st.error("❌ File data-baru.yaml tidak ditemukan")
    st.stop()

yaml_data = load_yaml()
CLASS_NAMES = yaml_data["names"]
CLASS_INFO  = yaml_data["info"]

# =====================================================
# LOAD YOLO MODEL (LAZY & AMAN)
# =====================================================
@st.cache_resource
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

MODEL_PATHS = {
    "YOLOv8 Nano": "models/bestnano.pt",
    "YOLOv8 Small": "models/bestsmall.pt",
    "YOLOv8 Medium": "models/yolo_medium.pt"
}

# =====================================================
# WARNA BBOX TIAP KELAS
# =====================================================
np.random.seed(42)
CLASS_COLORS = {
    i: tuple(np.random.randint(0, 255, 3).tolist())
    for i in range(len(CLASS_NAMES))
}

# =====================================================
# FUNGSI DETEKSI GAMBAR
# =====================================================
def detect_image(model, image_pil):
    import cv2

    img = np.array(image_pil)
    start = time.time()
    results = model.predict(img, verbose=False)
    infer_time = time.time() - start

    detections = []
    count = 0

    for r in results:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100

            if cls_id >= len(CLASS_NAMES):
                continue

            name = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]
            info  = CLASS_INFO.get(name, {})

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                f"{name} {conf:.1f}%",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {})
            })

            count += 1

    return img, detections, count, infer_time

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
    st.markdown("""
    **HerbaSmartAI** adalah sistem deteksi daun herbal berbasis  
    **Deep Learning (YOLOv8)** yang mampu:
    - Mengidentifikasi jenis daun herbal
    - Menampilkan manfaat & kandungan
    - Memberikan resep tradisional
    """)

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "Deteksi Gambar":
    st.header("📷 Deteksi Daun Herbal")

    model_choice = st.selectbox(
        "Pilih Model YOLO",
        MODEL_PATHS.keys()
    )
    model = load_model(MODEL_PATHS[model_choice])

    uploaded = st.file_uploader(
        "Upload gambar daun",
        ["jpg", "jpeg", "png"]
    )
    camera = st.camera_input("Atau ambil foto")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        with st.spinner("🔍 Mendeteksi..."):
            output, detections, count, infer_time = detect_image(model, image)

        st.image(output, use_container_width=True)
        st.success(f"🌱 Jumlah daun terdeteksi: **{count}**")
        st.info(f"⏱️ Waktu inferensi: {infer_time:.3f} detik")

        for d in detections:
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                st.markdown("**🧪 Kandungan:**")
                st.write(", ".join(d["components"]))

                st.markdown("**💊 Manfaat:**")
                for b in d["benefits"]:
                    st.write(f"- {b}")

                if d["recipes"]:
                    st.markdown("**🍵 Resep Tradisional:**")
                    for k, v in d["recipes"].items():
                        st.write(f"**{k.capitalize()}**:")
                        for step in v:
                            st.write(f"- {step}")

# =====================================================
# DETEKSI WEBCAM
# =====================================================
elif menu == "Deteksi Webcam":
    st.header("🎥 Deteksi Real-Time")

    if IS_CLOUD:
        st.warning("🚫 Webcam tidak didukung di Streamlit Cloud")
        st.stop()

    import cv2

    model_choice = st.selectbox(
        "Pilih Model YOLO",
        MODEL_PATHS.keys()
    )
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
                    name = CLASS_NAMES[cls_id]
                    color = CLASS_COLORS[cls_id]

                    cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        rgb, name, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

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
                        st.markdown("**🍵 Resep Tradisional:**")
                        for k, v in info["recipes"].items():
                            st.write(f"**{k.capitalize()}**:")
                            for step in v:
                                st.write(f"- {step}")

        if not found:
            st.warning("❌ Tidak ditemukan daun untuk gejala tersebut")
