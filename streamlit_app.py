import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
import time
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD YAML DATA
# =====================================================
@st.cache_data
def load_yaml():
    if not os.path.exists("data-baru.yaml"):
        st.error("File 'data-baru.yaml' tidak ditemukan!")
        return {"names": [], "info": {}}
    with open("data-baru.yaml", "r") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data.get("names", [])
CLASS_INFO = yaml_data.get("info", {})

# =====================================================
# LOAD YOLO MODEL
# =====================================================
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

MODEL_PATHS = {
    "YOLOv11 Nano": "models/bestnano.pt",
    "YOLOv11 Small": "models/bestsmall.pt",
    "YOLOv11 Medium": "best_m.pt"
}

# =====================================================
# FUNGSI DETEKSI GAMBAR
# =====================================================
def detect_image(image: Image.Image, model):
    start = time.time()
    results = model.predict(image, verbose=False)
    infer_time = time.time() - start

    draw = ImageDraw.Draw(image)
    detections = []
    
    # Mencoba memuat font agar teks lebih terbaca
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
            info = CLASS_INFO.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {}),
                "gambar": info.get("gambar", "")
            })

            # Gambar Bounding Box
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=5)
            # Gambar Label
            draw.text((x1, y1 - 35), f"{name} {conf:.1f}%", fill="#00FF00", font=font)

    return image, detections, infer_time

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    st.markdown("### 🌿 MENU NAVIGASI")
    menu = st.radio(
        "Pilih Halaman:",
        ["🏠 Beranda", "📷 Deteksi Gambar", "💊 Rekomendasi Manfaat"]
    )

# =====================================================
# HALAMAN: BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.title("🌿 HerbaSmartAI")
    st.markdown("""
    ### Selamat Datang di Sistem Identifikasi Daun Herbal
    Aplikasi ini menggunakan teknologi **Computer Vision (YOLOv11)** untuk membantu Anda mengenali berbagai jenis tanaman obat melalui foto.
    
    **Fitur Utama:**
    * **Identifikasi Otomatis:** Deteksi jenis daun secara instan.
    * **Informasi Kandungan:** Mengetahui senyawa kimia alami di dalam daun.
    * **Manfaat Kesehatan:** Informasi khasiat untuk pengobatan alami.
    * **Pencarian Manfaat:** Cari daun berdasarkan gejala kesehatan Anda.
    """)

# =====================================================
# HALAMAN: DETEKSI GAMBAR
# =====================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal")

    # Pilih Model
    yolo_choice = st.selectbox("⚙️ Pilih Varian Model YOLO", list(MODEL_PATHS.keys()))
    model_path = MODEL_PATHS[yolo_choice]
    model = load_model(model_path)

    if model is None:
        st.error(f"File model `{model_path}` tidak ditemukan! Pastikan file ada di folder yang benar.")
    else:
        uploaded = st.file_uploader("Upload gambar daun (JPG, PNG, JPEG)", type=["jpg","png","jpeg"])
        camera = st.camera_input("Atau ambil foto langsung")

        image = None
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
        elif camera:
            image = Image.open(camera).convert("RGB")

        if image:
            with st.spinner("🔍 Sedang menganalisis gambar..."):
                result_img, detections, infer_time = detect_image(image.copy(), model)

            st.image(result_img, use_container_width=True, caption="Hasil Deteksi")
            st.success(f"⏱️ Waktu deteksi: {infer_time:.3f} detik")

            if not detections:
                st.warning("⚠️ Tidak ada daun herbal yang terdeteksi.")
            else:
                st.markdown("### 📋 Hasil Analisis")
                for d in detections:
                    with st.expander(f"🌿 {d['name']} ({d['confidence']}%)", expanded=True):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if d["gambar"] and os.path.exists(d["gambar"]):
                                st.image(d["gambar"], use_container_width=True)
                            else:
                                st.info("Gambar referensi tidak tersedia")
                        
                        with col2:
                            st.markdown("**🧪 Kandungan Kimia:**")
                            st.write(", ".join(d["components"]) if d["components"] else "Informasi tidak tersedia")

                            st.markdown("**💊 Manfaat Kesehatan:**")
                            if d["benefits"]:
                                for b in d["benefits"]:
                                    st.write(f"- {b}")
                            else:
                                st.write("Informasi tidak tersedia")

# =====================================================
# HALAMAN: REKOMENDASI MANFAAT
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Cari Daun Berdasarkan Manfaat")
    st.markdown("Masukkan keluhan atau manfaat yang Anda cari (contoh: *batuk*, *demam*, *asam urat*).")

    query = st.text_input("🔍 Kata Kunci Manfaat:")

    if query:
        found = False
        for leaf, info in CLASS_INFO.items():
            # Cek apakah query ada di dalam daftar manfaat
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                found = True
                with st.expander(f"🌿 {leaf}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if info.get("gambar") and os.path.exists(info["gambar"]):
                            st.image(info["gambar"], use_container_width=True)
                    
                    with col2:
                        st.write("**Kandungan:**", ", ".join(info.get("components", [])))
                        st.write("**Daftar Manfaat:**")
                        for b in info.get("benefits", []):
                            st.write(f"- {b}")

        if not found:
            st.warning(f"❌ Tidak ditemukan tanaman herbal dengan manfaat terkait '{query}'")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("HerbaSmartAI © 2024 - Pengenal Daun Herbal Pintar")
