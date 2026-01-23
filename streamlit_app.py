# =========================================================
# STREAMLIT APP - DETEKSI DAUN HERBAL (YOLOv8 + WEBCAM RT)
# =========================================================

import streamlit as st
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from PIL import Imageimport streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import yaml
import cv2
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# LOAD YAML
# =====================================================
@st.cache_data
def load_yaml():
    with open("data-baru.yaml", "r") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data["names"]
CLASS_INFO = yaml_data["info"]

# =====================================================
# LOAD MODEL (DINAMIS)
# =====================================================
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

MODEL_PATHS = {
    "YOLOv11 Nano": "models/bestnano.pt",
    "YOLOv11 Small": "models/bestsmall.pt",
    "YOLOv11 Medium": "best_m.pt"
}

# =====================================================
# SIDEBAR NAVIGATION (TETAP SEPERTI AWAL)
# =====================================================
st.sidebar.title("🌿 MENU")
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "📷 Deteksi Gambar", "🎥 Deteksi Webcam", "💊 Rekomendasi Manfaat"]
)

# =====================================================
# FUNGSI DETEKSI GAMBAR
# =====================================================
def detect_image(image: Image.Image, model):
    start = time.time()
    results = model.predict(image, verbose=False)
    infer_time = time.time() - start

    draw = ImageDraw.Draw(image)
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100
            name = CLASS_NAMES[cls_id]
            info = CLASS_INFO.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {}),
                "gambar": info.get("gambar", "")
            })

            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1 - 20), f"{name} {conf:.1f}%", fill="green")

    return image, detections, infer_time

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.markdown("## 🌿 HerbaSmartAI")
    st.info("""
    Sistem deteksi daun herbal berbasis **YOLOv8**
    untuk menampilkan nama daun, kandungan, manfaat,
    dan resep tradisional.
    """)

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal")

    # 🔽 PILIH VARIAN YOLO (KHUSUS HALAMAN INI)
    yolo_choice = st.selectbox(
        "⚙️ Pilih Varian YOLO",
        list(MODEL_PATHS.keys())
    )

    model = load_model(MODEL_PATHS[yolo_choice])
    st.caption(f"Model aktif: **{yolo_choice}**")

    uploaded = st.file_uploader("Upload gambar daun", type=["jpg","png","jpeg"])
    camera = st.camera_input("Atau ambil foto")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        with st.spinner("🔍 Mendeteksi..."):
            result_img, detections, infer_time = detect_image(image.copy(), model)

        st.image(result_img, use_container_width=True)
        st.success(f"⏱️ Waktu inferensi: {infer_time:.3f} detik")

        for d in detections:
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                if d["gambar"]:
                    st.image(d["gambar"], use_container_width=True)

                st.markdown("**🧪 Kandungan:**")
                st.write(", ".join(d["components"]))

                st.markdown("**💊 Manfaat:**")
                for b in d["benefits"]:
                    st.write(f"- {b}")

# =====================================================
# DETEKSI WEBCAM
# =====================================================
elif menu == "🎥 Deteksi Webcam":
    st.title("🎥 Deteksi Real-Time")

    # 🔽 PILIH VARIAN YOLO (KHUSUS HALAMAN INI)
    yolo_choice = st.selectbox(
        "⚙️ Pilih Varian YOLO",
        list(MODEL_PATHS.keys())
    )

    model = load_model(MODEL_PATHS[yolo_choice])
    st.caption(f"Model aktif: **{yolo_choice}**")

    run = st.checkbox("▶️ Aktifkan Webcam")
    frame_slot = st.empty()

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
                for i, box in enumerate(r.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(r.boxes.cls[i])
                    name = CLASS_NAMES[cls_id]
                    cv2.rectangle(rgb, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(rgb, name, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            frame_slot.image(rgb, channels="RGB", use_container_width=True)

        cap.release()

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Cari Daun Berdasarkan Manfaat")

    query = st.text_input("Masukkan gejala (contoh: batuk, demam)")

    if query:
        found = False
        for leaf, info in CLASS_INFO.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                found = True
                with st.expander(f"🌿 {leaf}"):
                    if info.get("gambar"):
                        st.image(info["gambar"], use_container_width=True)

                    st.write("**Kandungan:**", ", ".join(info.get("components", [])))
                    st.write("**Manfaat:**")
                    for b in info.get("benefits", []):
                        st.write(f"- {b}")

        if not found:
            st.warning("❌ Tidak ditemukan daun untuk gejala tersebut")

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
