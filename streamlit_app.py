import streamlit as st
from PIL import Image, ImageDraw
import yaml
import time
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# CEK LINGKUNGAN (CLOUD / LOCAL)
# =====================================================
IS_CLOUD = "STREAMLIT_CLOUD" in os.environ

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
# LAZY LOAD YOLO MODEL (INI KUNCI FIX ERROR)
# =====================================================
@st.cache_resource
def load_model(model_path):
    from ultralytics import YOLO   # ⬅️ LAZY IMPORT
    return YOLO(model_path)

MODEL_PATHS = {
    "YOLOv11 Nano": "models/bestnano.pt",
    "YOLOv11 Small": "models/bestsmall.pt",
    "YOLOv11 Medium": "best_m.pt"
}

# =====================================================
# SIDEBAR NAVIGATION (TETAP)
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
    Sistem deteksi daun herbal berbasis **YOLO**
    untuk menampilkan nama daun, kandungan,
    manfaat, dan resep tradisional.
    """)

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
    st.caption(f"Model aktif: **{yolo_choice}**")

    uploaded = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])
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

    if IS_CLOUD:
        st.warning("🚫 Webcam tidak didukung di Streamlit Cloud")
        st.stop()

    import cv2  # ⬅️ BARU BOLEH DI SINI

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
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        rgb, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

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
