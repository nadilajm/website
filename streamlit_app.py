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

# =====================================================
# CEK LINGKUNGAN
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
CLASS_NAMES = yaml_data.get("names", [])
CLASS_INFO = yaml_data.get("info", {})

# =====================================================
# LOAD YOLO (LAZY)
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
# WARNA PER KELAS (AUTO GENERATE)
# =====================================================
def generate_colors(n):
    np.random.seed(42)
    return [
        tuple(int(c) for c in np.random.randint(0, 255, 3))
        for _ in range(n)
    ]

CLASS_COLORS = generate_colors(len(CLASS_NAMES))

# =====================================================
# DRAW LABEL (NUMPY + OPENCV)
# =====================================================
def draw_label(img, text, x, y, color):
    import cv2

    (w, h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    cv2.rectangle(
        img,
        (x, y - h - 8),
        (x + w + 6, y),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x + 3, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

# =====================================================
# DETEKSI
# =====================================================
def run_detection(image_np, model):
    import cv2

    start = time.time()
    results = model.predict(image_np, verbose=False)
    infer_time = time.time() - start

    output_img = image_np.copy()
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100

            if cls_id >= len(CLASS_NAMES):
                continue

            name = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]
            info = CLASS_INFO.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "gambar": info.get("gambar", "")
            })

            # bounding box
            cv2.rectangle(
                output_img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{name} {conf:.1f}%"
            draw_label(output_img, label, x1, y1, color)

    return output_img, detections, infer_time

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌿 MENU")
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "📷 Deteksi Gambar", "🎥 Deteksi Webcam", "💊 Rekomendasi Manfaat"]
)

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.markdown("## 🌿 HerbaSmartAI")
    st.info(
        "Aplikasi deteksi daun herbal berbasis **YOLO**.\n\n"
        "✔ Bounding box berwarna per kelas\n"
        "✔ Nama daun & confidence\n"
        "✔ Jumlah daun terdeteksi\n"
        "✔ Mode gambar & webcam"
    )

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal")

    yolo_choice = st.selectbox("⚙️ Pilih Varian YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[yolo_choice])

    uploaded = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
    camera = st.camera_input("Atau ambil foto")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image is not None:
        image_np = np.array(image)

        with st.spinner("🔍 Mendeteksi..."):
            out_img, detections, infer_time = run_detection(image_np, model)

        st.image(out_img, use_container_width=True)
        st.success(f"⏱️ Inferensi: {infer_time:.3f} detik")
        st.info(f"📊 Total daun terdeteksi: **{len(detections)}**")

        for d in detections:
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)"):
                img = d.get("gambar")
                if isinstance(img, str) and (img.startswith("http") or os.path.exists(img)):
                    st.image(img, use_container_width=True)

                st.markdown("**🧪 Kandungan:**")
                st.write(", ".join(d.get("components", [])))

                st.markdown("**💊 Manfaat:**")
                for b in d.get("benefits", []):
                    st.write(f"- {b}")

# =====================================================
# DETEKSI WEBCAM
# =====================================================
elif menu == "🎥 Deteksi Webcam":
    st.title("🎥 Deteksi Webcam")

    if IS_CLOUD:
        st.warning("🚫 Webcam tidak tersedia di Streamlit Cloud")
        st.stop()

    import cv2

    yolo_choice = st.selectbox("⚙️ Pilih Varian YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[yolo_choice])

    run = st.checkbox("▶️ Aktifkan Webcam")
    frame_box = st.empty()
    counter_box = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_img, detections, _ = run_detection(rgb, model)

            frame_box.image(out_img, channels="RGB", use_container_width=True)
            counter_box.info(f"📊 Daun terdeteksi: **{len(detections)}**")

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
                    img = info.get("gambar")
                    if isinstance(img, str) and (img.startswith("http") or os.path.exists(img)):
                        st.image(img, use_container_width=True)

                    st.write("**Kandungan:**", ", ".join(info.get("components", [])))
                    st.write("**Manfaat:**")
                    for b in info.get("benefits", []):
                        st.write(f"- {b}")

        if not found:
            st.warning("❌ Tidak ditemukan daun untuk gejala tersebut")
