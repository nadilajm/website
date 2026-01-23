# =========================================================
# STREAMLIT APP - DETEKSI DAUN HERBAL (YOLOv8 + WEBCAM RT)
# =========================================================

import streamlit as st
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import random
from pathlib import Path

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Deteksi Daun Herbal",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# PATH & KONFIGURASI
# =========================================================
MODEL_PATH = "best.pt"                # ganti jika nama model berbeda
YAML_PATH  = "data-baru.yaml"         # YAML di root folder

# =========================================================
# LOAD YAML
# =========================================================
@st.cache_data
def load_yaml():
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data["names"]
INFO_DATA   = yaml_data["info"]

# =========================================================
# WARNA BBOX (UNIK TIAP KELAS)
# =========================================================
random.seed(42)
CLASS_COLORS = {
    i: (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )
    for i in range(len(CLASS_NAMES))
}

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# =========================================================
# VIDEO PROCESSOR (WEBCAM REALTIME)
# =========================================================
class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.4, verbose=False)

        total = 0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                total += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100

                label = CLASS_NAMES[cls_id]
                color = CLASS_COLORS[cls_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{label} {conf:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.putText(
            img,
            f"Total Daun Terdeteksi: {total}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🌿 Menu")
menu = st.sidebar.radio(
    "Pilih Fitur",
    ["🏠 Beranda", "📷 Deteksi Gambar", "🎥 Deteksi Webcam Realtime"]
)

# =========================================================
# HALAMAN BERANDA
# =========================================================
if menu == "🏠 Beranda":
    st.title("🌿 Sistem Deteksi Daun Herbal")
    st.write(
        """
        Aplikasi ini menggunakan **YOLO (CNN)** untuk mendeteksi
        **daun herbal Indonesia** serta menampilkan:
        - Nama daun
        - Kandungan
        - Manfaat
        - Resep tradisional
        """
    )
    st.success("Siap digunakan untuk demo & sidang skripsi 🎓")

# =========================================================
# DETEKSI GAMBAR
# =========================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun dari Gambar")

    file = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        results = model.predict(img_np, conf=0.4)

        total = 0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                total += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                label = CLASS_NAMES[cls_id]
                color = CLASS_COLORS[cls_id]

                cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img_np,
                    f"{label} {conf:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        st.image(img_np, caption=f"Total Daun Terdeteksi: {total}")

        # INFO DETAIL
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = CLASS_NAMES[cls_id]
                info = INFO_DATA.get(name, {})

                st.subheader(f"🌿 {name}")

                st.markdown("**🧪 Kandungan:**")
                for c in info.get("components", []):
                    st.write("-", c)

                st.markdown("**💊 Manfaat:**")
                for b in info.get("benefits", []):
                    st.write("-", b)

                st.markdown("**🍵 Resep Tradisional:**")
                recipes = info.get("recipes", {})
                for title, detail in recipes.items():
                    st.markdown(f"**{title}**")
                    st.write("**Bahan:**")
                    for i in detail.get("ingredients", []):
                        st.write("-", i)
                    st.write("**Langkah:**")
                    for s in detail.get("steps", []):
                        st.write("-", s)

# =========================================================
# WEBCAM REALTIME (WAJIB WEBRTC)
# =========================================================
elif menu == "🎥 Deteksi Webcam Realtime":
    st.title("🎥 Deteksi Daun Realtime (Webcam)")

    st.info(
        "Gunakan browser Chrome/Edge dan izinkan akses kamera.\n"
        "Mode ini **REALTIME**, bukan foto."
    )

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
