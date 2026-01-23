# =====================================================
# HERBASMARTAI - FINAL STREAMLIT APP (SIDANG VERSION)
# =====================================================

import streamlit as st
from PIL import Image, ImageDraw
import yaml
import time
import os
import random

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# ENV CHECK
# =====================================================
IS_CLOUD = "STREAMLIT_CLOUD" in os.environ

# =====================================================
# LOAD YAML DATA
# =====================================================
@st.cache_data
def load_yaml():
    with open("data-baru.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

yaml_data   = load_yaml()
CLASS_NAMES = yaml_data["names"]
CLASS_INFO  = yaml_data["info"]

# =====================================================
# COLOR MAP (BBOX WARNA TIAP KELAS)
# =====================================================
random.seed(42)
COLOR_MAP = {
    name: (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )
    for name in CLASS_NAMES
}

# =====================================================
# LOAD YOLO MODEL (LAZY LOAD - AMAN CLOUD)
# =====================================================
@st.cache_resource
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

MODEL_PATHS = {
    "YOLOv11 Nano"  : "models/bestnano.pt",
    "YOLOv11 Small" : "models/bestsmall.pt",
    "YOLOv11 Medium": "models/best_m.pt"
}

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌿 MENU")
menu = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Beranda",
        "📷 Deteksi Gambar",
        "🎥 Deteksi Webcam",
        "💊 Rekomendasi Manfaat"
    ]
)

# =====================================================
# UTIL: RENDER RESEP (FIX FINAL)
# =====================================================
def render_recipes(recipes: dict):
    st.markdown("### 🍵 Resep Tradisional")
    for title, recipe in recipes.items():
        st.markdown(f"**{title}**")

        if recipe.get("ingredients"):
            st.markdown("**Bahan:**")
            for item in recipe["ingredients"]:
                st.write(f"- {item}")

        if recipe.get("steps"):
            st.markdown("**Cara Pembuatan:**")
            for i, step in enumerate(recipe["steps"], 1):
                st.write(f"{i}. {step}")

        st.markdown("---")

# =====================================================
# IMAGE DETECTION
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
            conf   = float(r.boxes.conf[i]) * 100
            name   = CLASS_NAMES[cls_id]

            color = COLOR_MAP[name]

            info = CLASS_INFO.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "info": info
            })

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text(
                (x1, y1 - 15),
                f"{name} {conf:.1f}%",
                fill=color
            )

    return image, detections, infer_time

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.title("🌿 HerbaSmartAI")
    st.info(
        "Sistem **Deteksi Daun Herbal berbasis YOLO** "
        "yang menampilkan **nama daun, kandungan, manfaat, "
        "dan resep tradisional**."
    )

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal")

    yolo_choice = st.selectbox(
        "⚙️ Pilih Varian YOLO",
        list(MODEL_PATHS.keys())
    )

    model = load_model(MODEL_PATHS[yolo_choice])

    uploaded = st.file_uploader(
        "Upload gambar daun",
        type=["jpg", "png", "jpeg"]
    )

    camera = st.camera_input("Atau ambil foto")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        with st.spinner("🔍 Mendeteksi..."):
            result_img, detections, infer_time = detect_image(
                image.copy(), model
            )

        st.image(result_img, use_container_width=True)
        st.success(f"⏱️ Waktu inferensi: {infer_time:.3f} detik")
        st.info(f"📊 Jumlah daun terdeteksi: **{len(detections)}**")

        for d in detections:
            info = d["info"]
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                if info.get("gambar") and os.path.exists(info["gambar"]):
                    st.image(info["gambar"], use_container_width=True)

                st.markdown("**🧪 Kandungan:**")
                st.write(", ".join(info.get("components", [])))

                st.markdown("**💊 Manfaat:**")
                for b in info.get("benefits", []):
                    st.write(f"- {b}")

                if info.get("recipes"):
                    render_recipes(info["recipes"])

# =====================================================
# DETEKSI WEBCAM
# =====================================================
elif menu == "🎥 Deteksi Webcam":
    st.title("🎥 Deteksi Webcam")

    st.info(
        "Webcam berjalan menggunakan **Streamlit Camera Input** "
        "(frame-by-frame, stabil, dan aman untuk deployment)."
    )

    yolo_choice = st.selectbox(
        "⚙️ Pilih Varian YOLO",
        list(MODEL_PATHS.keys())
    )
    model = load_model(MODEL_PATHS[yolo_choice])

    camera_frame = st.camera_input("📸 Ambil frame dari webcam")

    if camera_frame:
        image = Image.open(camera_frame).convert("RGB")

        with st.spinner("🔍 Mendeteksi dari webcam..."):
            result_img, detections, infer_time = detect_image(
                image.copy(), model
            )

        st.image(result_img, use_container_width=True)
        st.success(f"⏱️ Waktu inferensi: {infer_time:.3f} detik")
        st.info(f"📊 Jumlah daun terdeteksi: {len(detections)}")

        for d in detections:
            info = d["info"]
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                st.markdown("**🧪 Kandungan:**")
                st.write(", ".join(info.get("components", [])))

                st.markdown("**💊 Manfaat:**")
                for b in info.get("benefits", []):
                    st.write(f"- {b}")

                if info.get("recipes"):
                    render_recipes(info["recipes"])

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Rekomendasi Daun Berdasarkan Manfaat")

    query = st.text_input("Masukkan gejala (contoh: batuk, diabetes)")

    if query:
        found = False
        for leaf, info in CLASS_INFO.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                found = True
                with st.expander(f"🌿 {leaf}"):
                    if info.get("gambar") and os.path.exists(info["gambar"]):
                        st.image(info["gambar"], use_container_width=True)

                    st.markdown("**🧪 Kandungan:**")
                    st.write(", ".join(info.get("components", [])))

                    st.markdown("**💊 Manfaat:**")
                    for b in info.get("benefits", []):
                        st.write(f"- {b}")

                    if info.get("recipes"):
                        render_recipes(info["recipes"])

        if not found:
            st.warning("❌ Tidak ditemukan daun untuk gejala tersebut")
