import streamlit as st
import yaml
import time
import numpy as np
import cv2
import av

from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# LOAD DATA YAML
# =====================================================
@st.cache_data
def load_yaml():
    with open("data-baru.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data["names"]
CLASS_INFO  = yaml_data["info"]

# =====================================================
# YOLO MODEL PATHS
# =====================================================
MODEL_PATHS = {
    "YOLO Nano (Cepat)": "models/bestnano.pt",
    "YOLO Small (Seimbang)": "models/bestsmall.pt",
    "YOLO Medium (Akurat)": "models/bestmedium.pt"
}

@st.cache_resource
def load_model(path):
    return YOLO(path)

# =====================================================
# CLASS COLORS (FIXED)
# =====================================================
np.random.seed(42)
CLASS_COLORS = [
    tuple(int(c) for c in np.random.randint(50, 255, 3))
    for _ in CLASS_NAMES
]

# =====================================================
# IMAGE DETECTION FUNCTION
# =====================================================
def detect_image(model, image: Image.Image):
    img = np.array(image)
    start = time.time()

    results = model.predict(img, conf=0.3, verbose=False)
    infer_time = time.time() - start

    detections = []
    total = 0

    for r in results:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
            cls_id = int(r.boxes.cls[i])
            conf   = float(r.boxes.conf[i]) * 100

            name  = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]
            info  = CLASS_INFO.get(name, {})

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
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

            total += 1

    return img, detections, infer_time, total

# =====================================================
# REALTIME VIDEO PROCESSOR
# =====================================================
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=0.3, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                cls_id = int(r.boxes.cls[i])
                conf   = float(r.boxes.conf[i]) * 100

                name  = CLASS_NAMES[cls_id]
                color = CLASS_COLORS[cls_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{name} {conf:.1f}%",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =====================================================
# SIDEBAR MENU
# =====================================================
st.sidebar.title("🌿 HerbaSmartAI")
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
    **HerbaSmartAI** adalah sistem pendeteksi daun herbal berbasis **YOLO**
    yang mampu mengenali daun secara otomatis dan menampilkan:
    - Nama daun
    - Kandungan kimia
    - Manfaat kesehatan
    - Resep tradisional
    """)

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal (Gambar)")

    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])

    uploaded = st.file_uploader("Upload gambar daun", ["jpg", "png", "jpeg"])
    camera   = st.camera_input("Atau ambil foto")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        with st.spinner("Mendeteksi daun..."):
            output, detections, infer_time, total = detect_image(model, image)

        st.image(output, use_container_width=True)
        st.success(f"🌿 Total daun terdeteksi: {total}")
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
                    for title, rec in d["recipes"].items():
                        st.markdown(f"**{title}**")
                        st.markdown("*Bahan:*")
                        for ing in rec.get("ingredients", []):
                            st.write(f"- {ing}")
                        st.markdown("*Langkah:*")
                        for step in rec.get("steps", []):
                            st.write(f"- {step}")

# =====================================================
# DETEKSI WEBCAM REALTIME
# =====================================================
elif menu == "Deteksi Webcam":
    st.title("🎥 Deteksi Daun Herbal Real-Time")

    st.info("Gunakan browser Chrome / Edge dan izinkan akses kamera.")

    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])

    webrtc_streamer(
        key="herba-realtime",
        video_processor_factory=lambda: YOLOVideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "Rekomendasi Manfaat":
    st.title("💊 Cari Daun Berdasarkan Manfaat")

    query = st.text_input("Masukkan gejala (contoh: diabetes, batuk)")

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
                        for title, rec in info["recipes"].items():
                            st.markdown(f"**{title}**")
                            st.markdown("*Bahan:*")
                            for ing in rec.get("ingredients", []):
                                st.write(f"- {ing}")
                            st.markdown("*Langkah:*")
                            for step in rec.get("steps", []):
                                st.write(f"- {step}")

        if not found:
            st.warning("❌ Tidak ditemukan daun untuk manfaat tersebut")
