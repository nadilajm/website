import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
import time
import cv2
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Deteksi Daun Herbal",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS STYLING (Tema Hijau Herbal)
# =====================================================
st.markdown("""
<style>
/* Background utama hijau */
.stApp {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

/* Container utama */
.block-container {
    max-width: 1400px;
    padding: 3rem 2rem;
}

/* ===============================
   SIDEBAR - HIJAU dengan TEKS PUTIH
================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #059669 0%, #047857 100%) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
    font-weight: 700;
}

/* ===============================
   AREA VISUALISASI - TERANG/PUTIH
================================ */
.card, .detect-card {
    background: white !important;
    padding: 2rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}

/* Judul */
h1, h2, h3 {
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.card h1, .card h2, .card h3,
.detect-card h1, .detect-card h2, .detect-card h3 {
    color: #111827 !important;
    text-shadow: none;
}

/* ===============================
   INPUT & UPLOAD - HIJAU dengan TEKS PUTIH
================================ */
section[data-testid="stFileUploader"],
section[data-testid="stCameraInput"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    padding: 2rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}

.main label {
    color: white !important;
    font-weight: 700;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

section[data-testid="stFileUploader"] label,
section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] p,
section[data-testid="stFileUploader"] div,
section[data-testid="stFileUploader"] small {
    color: white !important;
    font-weight: 600;
}

section[data-testid="stCameraInput"] label,
section[data-testid="stCameraInput"] span,
section[data-testid="stCameraInput"] p,
section[data-testid="stCameraInput"] div,
section[data-testid="stCameraInput"] small {
    color: white !important;
    font-weight: 600;
}

section[data-testid="stTextInput"] input {
    background: white !important;
    color: #111827 !important;
    border: 2px solid white !important;
    border-radius: 12px;
    font-weight: 600;
}

section[data-testid="stTextInput"] input::placeholder {
    color: #6b7280 !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.2) !important;
    border: 2px solid white !important;
    color: white !important;
}

/* Confidence badge */
.confidence {
    display: inline-block;
    background: #047857;
    color: white !important;
    padding: 0.5rem 1.5rem;
    border-radius: 999px;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* Button */
.stButton > button {
    background: white !important;
    color: #047857 !important;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.stButton > button:hover {
    background: #f0fdf4 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}

.stSuccess {
    background: rgba(255,255,255,0.95) !important;
    color: #047857 !important;
    border-radius: 12px;
}

/* Hero section */
.hero-title {
    font-size: 4rem;
    font-weight: bold;
    color: #639872;
    text-shadow: 
        3px 3px 0 #0B6A43, 
        6px 6px 0 #0B6A43;
    text-align: center;
    margin-bottom: 2rem;
}

.hero-content {
    background: rgba(255,255,255,0.95);
    padding: 3rem;
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* Recipe accordion */
.recipe-section {
    background: #f0fdf4;
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD YAML DATA
# =====================================================
@st.cache_data
def load_yaml():
    with open("data-baru.yaml", "r") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
class_names = yaml_data["names"]
class_info = yaml_data["info"]

# =====================================================
# LOAD YOLO MODEL
# =====================================================
@st.cache_resource
def load_model(path):
    return YOLO(path)

MODEL_PATHS = {
    "YOLOv8n (Nano)": "models/bestnano.pt",
    "YOLOv8s (Small)": "models/bestsmall.pt",
    "YOLOv8m (Medium)": "models/yolo_medium.pt",
    "Best Model": "best.pt"
}

# =====================================================
# DETECTION + DRAW BOUNDING BOX
# =====================================================
def detect_and_draw(model, image):
    start = time.time()
    results = model.predict(image, verbose=False)
    infer_time = time.time() - start

    detections = []
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100
            name = class_names[cls_id]
            info = class_info.get(name, {})

            # Get recipes with conversion
            recipes_raw = info.get('recipes', {})
            converted_recipes = {}
            for benefit, recipe in recipes_raw.items():
                converted_recipes[benefit] = {
                    'bahan': recipe.get('ingredients', []),
                    'langkah': recipe.get('steps', [])
                }

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": converted_recipes,
                "gambar": info.get("gambar", "")
            })

            # Bounding box (PUTIH)
            x1, y1, x2, y2 = [int(c) for c in box]
            draw.rectangle([x1, y1, x2, y2], outline="white", width=4)

            # Label
            label = f"{name} {conf:.1f}%"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            w = text_bbox[2] - text_bbox[0]
            h = text_bbox[3] - text_bbox[1]

            draw.rectangle(
                [x1, y1 - h - 6, x1 + w + 6, y1],
                fill="white"
            )
            draw.text(
                (x1 + 3, y1 - h - 3),
                label,
                fill="#065f46",
                font=font
            )

    return image, detections, infer_time

# =====================================================
# WEBCAM DETECTION FUNCTION
# =====================================================
def detect_webcam():
    cap = cv2.VideoCapture(0)
    
    stframe = st.empty()
    stop_button = st.button("Stop Webcam")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        results = model.predict(frame_rgb, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    class_id = int(boxes.cls[i].item())
                    class_name = class_names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, class_name, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)
    
    cap.release()

# =====================================================
# SIDEBAR MENU
# =====================================================
with st.sidebar:
    st.markdown("### 🌿 MENU NAVIGASI")
    menu = st.radio(
        "",
        ["🏠 Beranda", "🔍 Deteksi Gambar", "📸 Deteksi Webcam", "💊 Rekomendasi Manfaat"],
        label_visibility="collapsed"
    )

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.markdown('<h1 class="hero-title">🌿 DETEKSI DAUN HERBAL</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-content">
        <h2 style="color: #047857; text-align: center;">Sistem Deteksi Daun Herbal Berbasis Deep Learning</h2>
        <p style="color: #111827; text-align: center; font-size: 1.1rem; line-height: 1.8;">
            Aplikasi ini menggunakan teknologi <strong>YOLOv8 (You Only Look Once)</strong> untuk mendeteksi 
            berbagai jenis daun herbal secara real-time. Sistem ini dapat mengidentifikasi kandungan, 
            manfaat kesehatan, dan memberikan rekomendasi resep pengobatan tradisional.
        </p>
        
        <hr style="border-color: #10b981; margin: 2rem 0;">
        
        <h3 style="color: #047857; margin-top: 2rem;">✨ Fitur Utama:</h3>
        <ul style="color: #111827; font-size: 1rem; line-height: 2;">
            <li><strong>Deteksi Gambar:</strong> Upload foto daun untuk analisis instant</li>
            <li><strong>Deteksi Real-time:</strong> Gunakan webcam untuk deteksi langsung</li>
            <li><strong>Informasi Lengkap:</strong> Kandungan kimia, manfaat kesehatan, dan resep pengobatan</li>
            <li><strong>Rekomendasi Cerdas:</strong> Cari daun herbal berdasarkan manfaat yang diinginkan</li>
        </ul>
        
        <h3 style="color: #047857; margin-top: 2rem;">🎯 Cara Menggunakan:</h3>
        <ol style="color: #111827; font-size: 1rem; line-height: 2;">
            <li>Pilih menu di sidebar sesuai kebutuhan Anda</li>
            <li>Upload gambar atau aktifkan webcam untuk deteksi</li>
            <li>Lihat hasil deteksi beserta informasi lengkap daun herbal</li>
            <li>Gunakan menu Rekomendasi untuk mencari berdasarkan manfaat</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "🔍 Deteksi Gambar":
    st.title("🔍 Deteksi Daun Herbal dari Gambar")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
        model = load_model(MODEL_PATHS[model_choice])

    uploaded = st.file_uploader("Upload gambar daun", ["jpg", "jpeg", "png"])
    camera = st.camera_input("Atau ambil foto langsung")

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        with st.spinner("🔄 Mendeteksi daun herbal..."):
            image_box, detections, infer_time = detect_and_draw(model, image.copy())

        st.markdown("""
        <div class="card">
            <h3>Hasil Deteksi</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(image_box, use_container_width=True)
        st.success(f"✅ Deteksi selesai dalam {infer_time:.3f} detik")

        if detections:
            st.markdown("---")
            st.markdown("### 📋 Detail Deteksi")
            
            for idx, d in enumerate(detections, 1):
                with st.expander(f"🌿 {d['name']} - Confidence: {d['confidence']}%", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if d['gambar']:
                            try:
                                st.image(d['gambar'], caption=d['name'], use_container_width=True)
                            except:
                                st.info("Gambar tidak tersedia")
                    
                    with col2:
                        st.markdown(f"**Tingkat Kepercayaan:** `{d['confidence']}%`")
                        
                        if d['components']:
                            st.markdown("**🧪 Kandungan Kimia:**")
                            st.write(", ".join(d['components']))
                        
                        if d['benefits']:
                            st.markdown("**💊 Manfaat Kesehatan:**")
                            for benefit in d['benefits']:
                                st.write(f"• {benefit}")
                        
                        if d['recipes']:
                            st.markdown("**📖 Resep Pengobatan:**")
                            for benefit, recipe in d['recipes'].items():
                                st.markdown(f"**{benefit}:**")
                                
                                st.markdown("*Bahan:*")
                                for bahan in recipe['bahan']:
                                    st.write(f"  • {bahan}")
                                
                                st.markdown("*Langkah:*")
                                for i, langkah in enumerate(recipe['langkah'], 1):
                                    st.write(f"  {i}. {langkah}")
                                
                                st.markdown("---")
        else:
            st.warning("⚠️ Tidak ada daun herbal yang terdeteksi")

# =====================================================
# DETEKSI WEBCAM
# =====================================================
elif menu == "📸 Deteksi Webcam":
    st.title("📸 Deteksi Daun Herbal Real-time")
    
    st.markdown("""
    <div class="card">
        <h3>Deteksi menggunakan Webcam</h3>
        <p>Klik tombol di bawah untuk memulai deteksi real-time menggunakan webcam Anda.</p>
    </div>
    """, unsafe_allow_html=True)
    
    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys(), key="webcam_model")
    model = load_model(MODEL_PATHS[model_choice])
    
    if st.button("🎥 Mulai Deteksi Webcam"):
        detect_webcam()
    
    st.info("💡 **Tips:** Pastikan pencahayaan cukup dan daun berada dalam jarak yang jelas dari kamera.")

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Rekomendasi Berdasarkan Manfaat")
    
    st.markdown("""
    <div class="card">
        <h3>Cari Daun Herbal Berdasarkan Manfaat</h3>
        <p>Masukkan gejala atau kondisi kesehatan yang ingin Anda obati (contoh: batuk, demam, diabetes)</p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("🔍 Masukkan manfaat yang dicari:", placeholder="Contoh: batuk, demam, diabetes")

    if query:
        results = []
        for leaf, info in class_info.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                results.append({
                    "name": leaf,
                    "info": info
                })
        
        if results:
            st.success(f"✅ Ditemukan {len(results)} daun herbal untuk '{query}'")
            st.markdown("---")
            
            for idx, result in enumerate(results, 1):
                with st.expander(f"🌿 {result['name']}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if result['info'].get('gambar'):
                            try:
                                st.image(result['info']['gambar'], caption=result['name'], use_container_width=True)
                            except:
                                st.info("Gambar tidak tersedia")
                    
                    with col2:
                        if result['info'].get('components'):
                            st.markdown("**🧪 Kandungan:**")
                            st.write(", ".join(result['info']['components']))
                        
                        if result['info'].get('benefits'):
                            st.markdown("**💊 Manfaat:**")
                            for benefit in result['info']['benefits']:
                                st.write(f"• {benefit}")
                        
                        if result['info'].get('recipes'):
                            st.markdown("**📖 Resep Pengobatan:**")
                            for benefit, recipe in result['info']['recipes'].items():
                                st.markdown(f"**{benefit}:**")
                                
                                st.markdown("*Bahan:*")
                                for bahan in recipe.get('ingredients', []):
                                    st.write(f"  • {bahan}")
                                
                                st.markdown("*Langkah:*")
                                for i, langkah in enumerate(recipe.get('steps', []), 1):
                                    st.write(f"  {i}. {langkah}")
                                
                                st.markdown("---")
        else:
            st.warning(f"⚠️ Tidak ditemukan daun herbal untuk manfaat '{query}'")
            st.info("💡 Coba kata kunci lain seperti: batuk, demam, diabetes, hipertensi, kolesterol")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p>🌿 <strong>Sistem Deteksi Daun Herbal</strong> 🌿</p>
    <p>Powered by YOLOv8 & Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)
