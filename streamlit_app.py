import streamlit as st
from ultralytics import YOLO
from PIL import Image
import yaml
import time
import cv2
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="HerbaSmartAI - Deteksi Daun Herbal",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# ENHANCED MODERN STYLE
# =====================================================
st.markdown("""
<style>

/* =====================================================
   BACKGROUND DENGAN OVERLAY GRADIENT
===================================================== */
.stApp {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.12) 100%),
                url("static/assets/Untitled design.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* =====================================================
   MAIN CONTAINER
===================================================== */
.block-container {
    max-width: 1400px;
    padding: 3rem 2rem;
}

/* =====================================================
   HERO SECTION
===================================================== */
.hero-section {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    padding: 4rem 3rem;
    border-radius: 28px;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 20px 60px rgba(5, 150, 105, 0.3);
}

.hero-title {
    color: #ffffff !important;
    font-size: 3.5rem !important;
    font-weight: 900;
    margin-bottom: 1rem !important;
    text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
}

.hero-subtitle {
    color: #d1fae5 !important;
    font-size: 1.3rem !important;
    font-weight: 500;
    margin: 0;
}

/* =====================================================
   FEATURE CARDS
===================================================== */
.feature-card {
    background: rgba(255, 255, 255, 0.98);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    border: 2px solid rgba(16, 185, 129, 0.1);
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.12);
    border-color: #10b981;
}

.feature-icon {
    font-size: 3.5rem;
    margin-bottom: 1.5rem;
    filter: grayscale(100%);
    opacity: 0.6;
}

.feature-card h3 {
    color: #065f46 !important;
    font-size: 1.5rem !important;
    font-weight: 800;
    margin-bottom: 1rem !important;
}

.feature-card p {
    color: #374151 !important;
    font-size: 1rem;
    line-height: 1.7;
}

/* =====================================================
   INFO CARD
===================================================== */
.info-card {
    background: rgba(255, 255, 255, 0.98);
    padding: 3rem;
    border-radius: 24px;
    margin-top: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid rgba(16, 185, 129, 0.2);
}

.section-title {
    color: #065f46 !important;
    font-size: 2rem !important;
    font-weight: 800;
    margin-bottom: 2rem !important;
    text-align: center;
}

.feature-list {
    display: grid;
    gap: 1.5rem;
}

.feature-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    border-radius: 12px;
    border-left: 4px solid #10b981;
    transition: all 0.3s ease;
}

.feature-item:hover {
    transform: translateX(8px);
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
}

.feature-item .bullet {
    color: #10b981 !important;
    font-size: 1.5rem;
    font-weight: 900;
    line-height: 1;
}

.feature-item span {
    color: #1f2937 !important;
    font-size: 1.05rem;
    font-weight: 500;
    line-height: 1.7;
}

/* =====================================================
   PAGE TITLE
===================================================== */
.page-title {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900;
    font-size: 3rem !important;
    margin-bottom: 2rem !important;
    text-align: center;
}

/* =====================================================
   CARD BASE
===================================================== */
.card {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    border: 2px solid rgba(16, 185, 129, 0.1);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    border-color: #10b981;
}

.upload-card {
    border: 2px dashed #10b981 !important;
    background: linear-gradient(135deg, rgba(240, 253, 244, 0.5), rgba(255, 255, 255, 0.95)) !important;
}

.upload-card:hover {
    background: linear-gradient(135deg, rgba(240, 253, 244, 0.8), rgba(255, 255, 255, 0.95)) !important;
    border-color: #059669 !important;
}

/* =====================================================
   LABEL TEXT
===================================================== */
.label-text {
    color: #065f46 !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* =====================================================
   RESULT CARD
===================================================== */
.result-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid #d1fae5;
    transition: all 0.3s ease;
}

.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
    border-color: #10b981;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #d1fae5;
    flex-wrap: wrap;
    gap: 1rem;
}

.plant-name {
    color: #065f46 !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    margin: 0 !important;
}

.confidence-badge {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff !important;
    padding: 0.75rem 2rem;
    border-radius: 50px;
    font-weight: 800;
    font-size: 1.1rem;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    transition: all 0.3s ease;
}

.confidence-badge:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
}

.result-content {
    margin-top: 1.5rem;
}

.result-content p {
    color: #1f2937 !important;
    font-size: 1.05rem;
    line-height: 1.8;
    margin-bottom: 1rem !important;
}

.info-label {
    color: #065f46 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
}

/* =====================================================
   RECIPE CONTAINER
===================================================== */
.recipe-container {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    padding: 2.5rem;
    border-radius: 20px;
    border-left: 6px solid #10b981;
    margin-top: 1rem;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.15);
}

.recipe-title {
    color: #065f46 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #10b981;
}

.recipe-section-title {
    color: #047857 !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.recipe-item,
.recipe-step {
    color: #1f2937 !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
    margin-bottom: 0.75rem !important;
    padding-left: 0.5rem;
}

.recipe-divider {
    border: none;
    border-top: 2px solid #10b981;
    margin: 2rem 0;
    opacity: 0.3;
}

/* =====================================================
   SELECTBOX
===================================================== */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.98) !important;
    border-radius: 16px !important;
    border: 2px solid #d1fae5 !important;
    padding: 1rem 1.25rem !important;
    transition: all 0.3s ease;
}

div[data-baseweb="select"] > div:hover {
    border-color: #10b981 !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
}

div[data-baseweb="select"] span {
    color: #1f2937 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}

ul[data-baseweb="menu"] {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(10px);
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    border: 2px solid #d1fae5;
}

ul[data-baseweb="menu"] li {
    color: #1f2937 !important;
    font-weight: 600;
    font-size: 1.05rem !important;
    padding: 1rem 1.25rem !important;
    transition: all 0.2s ease;
}

ul[data-baseweb="menu"] li:hover {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
    color: #065f46 !important;
    font-weight: 700 !important;
}

/* =====================================================
   FILE UPLOADER
===================================================== */
section[data-testid="stFileUploader"],
section[data-testid="stCameraInput"] {
    border: 2px dashed #10b981 !important;
    background: linear-gradient(135deg, rgba(240, 253, 244, 0.4), rgba(255, 255, 255, 0.95)) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
}

section[data-testid="stFileUploader"]:hover,
section[data-testid="stCameraInput"]:hover {
    border-color: #059669 !important;
    background: linear-gradient(135deg, rgba(240, 253, 244, 0.7), rgba(255, 255, 255, 0.95)) !important;
}

section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] small,
section[data-testid="stFileUploader"] p,
section[data-testid="stCameraInput"] span,
section[data-testid="stCameraInput"] small,
section[data-testid="stCameraInput"] p {
    color: #1f2937 !important;
    font-weight: 700 !important;
}

/* =====================================================
   SIDEBAR
===================================================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
    border-right: 3px solid #d1fae5;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #065f46 !important;
    font-weight: 900 !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #1f2937 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: 2px solid transparent;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    transform: translateX(6px);
    border-color: #10b981;
}

/* =====================================================
   IMAGE
===================================================== */
img {
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    transition: transform 0.3s ease;
}

img:hover {
    transform: scale(1.02);
}

/* =====================================================
   BUTTONS
===================================================== */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 1rem 2.5rem;
    font-weight: 800;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
}

/* =====================================================
   EXPANDER
===================================================== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    font-weight: 800;
    font-size: 1.1rem !important;
    color: #065f46 !important;
    border: 2px solid #d1fae5;
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    border-color: #10b981;
    background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
}

/* =====================================================
   MESSAGES
===================================================== */
.stSuccess,
.stWarning,
.stInfo {
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    font-weight: 700;
    font-size: 1.05rem;
}

/* =====================================================
   SPINNER
===================================================== */
.stSpinner > div {
    border-color: #10b981 !important;
    border-width: 4px !important;
}

/* =====================================================
   INPUT FIELDS
===================================================== */
.stTextInput > div > div > input {
    border-radius: 14px;
    border: 2px solid #d1fae5;
    padding: 1rem 1.25rem;
    transition: all 0.3s ease;
    font-size: 1.05rem;
    font-weight: 600;
    color: #1f2937 !important;
}

.stTextInput > div > div > input:focus {
    border-color: #10b981;
    box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.15);
}

.stTextInput > div > div > input::placeholder {
    color: #9ca3af !important;
    font-weight: 500;
}

/* =====================================================
   SCROLLBAR
===================================================== */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

::-webkit-scrollbar-track {
    background: #f0fdf4;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #059669;
}

/* =====================================================
   TEXT DEFAULTS
===================================================== */
p, li, label, span, div {
    color: #1f2937 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #065f46 !important;
}

/* =====================================================
   ANIMATIONS
===================================================== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.card,
.result-card,
.feature-card {
    animation: fadeIn 0.6s ease-out;
}

/* =====================================================
   CHECKBOX
===================================================== */
.stCheckbox {
    padding: 1rem;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    border: 2px solid #d1fae5;
}

.stCheckbox:hover {
    border-color: #10b981;
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
}

# =====================================================
# GENERATE CLASS COLORS
# =====================================================
np.random.seed(42)
CLASS_COLORS = [
    tuple(int(c) for c in np.random.randint(0, 255, 3))
    for _ in class_names
]

# =====================================================
# DRAW LABEL FUNCTION
# =====================================================
def draw_label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y-h-8), (x+w+6, y), color, -1)
    cv2.putText(img, text, (x+3, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# =====================================================
# DETECTION FUNCTION WITH BOUNDING BOX
# =====================================================
def detect_image(model, image):
    start = time.time()
    
    # Convert PIL to numpy
    img_np = np.array(image)
    
    # Run prediction
    results = model.predict(img_np, verbose=False)
    infer_time = time.time() - start

    # Copy image for drawing
    output = img_np.copy()
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            # Get bounding box coordinates
            box = r.boxes.xyxy[i]
            x1, y1, x2, y2 = map(int, box)
            
            # Get class info
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100
            
            if cls_id >= len(class_names):
                continue
                
            name = class_names[cls_id]
            color = CLASS_COLORS[cls_id]
            info = class_info.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {})
            })

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            draw_label(output, f"{name} {conf:.1f}%", x1, y1, color)

    return output, detections, infer_time

# =====================================================
# SIDEBAR MENU
# =====================================================
st.sidebar.markdown('<h2 style="text-align: center;">MENU NAVIGASI</h2>', unsafe_allow_html=True)
menu = st.sidebar.radio(
    "",
    ["Beranda", "Deteksi Gambar", "Deteksi Webcam", "Rekomendasi Manfaat"],
    label_visibility="collapsed"
)

# =====================================================
# BERANDA
# =====================================================
if menu == "Beranda":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">HerbaSmartAI</h1>
        <p class="hero-subtitle">Sistem Deteksi Daun Herbal Berbasis AI dengan Teknologi YOLO</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Deteksi Akurat</h3>
            <p>Menggunakan model YOLO v8 dengan akurasi tinggi untuk mengenali berbagai jenis daun herbal dengan presisi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Proses Cepat</h3>
            <p>Analisis gambar dalam hitungan detik dengan teknologi deep learning dan computer vision terkini</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h3>Informasi Lengkap</h3>
            <p>Database lengkap kandungan kimia, manfaat kesehatan, dan resep pengolahan tradisional</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3 class="section-title">Fitur Unggulan Aplikasi</h3>
        <div class="feature-list">
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Deteksi otomatis menggunakan teknologi YOLO v8 dengan bounding box visualization</span>
            </div>
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Pilihan model Nano, Small, dan Medium sesuai kebutuhan kecepatan dan akurasi</span>
            </div>
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Confidence score untuk menunjukkan tingkat kepercayaan hasil deteksi</span>
            </div>
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Database lengkap kandungan kimia dan manfaat kesehatan setiap tanaman</span>
            </div>
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Resep tradisional lengkap dengan bahan dan cara pembuatan</span>
            </div>
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Deteksi realtime menggunakan webcam untuk identifikasi langsung</span>
            </div>
            <div class="feature-item">
                <span class="bullet">●</span>
                <span>Pencarian berdasarkan manfaat kesehatan yang diinginkan</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "Deteksi Gambar":
    st.markdown('<h1 class="page-title">Deteksi Daun Herbal dari Gambar</h1>', unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<p class="label-text">Pilih Model YOLO</p>', unsafe_allow_html=True)
    model_choice = st.selectbox("", MODEL_PATHS.keys(), label_visibility="collapsed")
    model = load_model(MODEL_PATHS[model_choice])
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='card upload-card'>", unsafe_allow_html=True)
        st.markdown('<p class="label-text">Upload Gambar Daun</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", ["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card upload-card'>", unsafe_allow_html=True)
        st.markdown('<p class="label-text">Ambil Foto Langsung</p>', unsafe_allow_html=True)
        camera = st.camera_input("", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif camera:
        image = Image.open(camera).convert("RGB")

    if image:
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown('<p class="label-text">Gambar Input</p>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Sedang menganalisis gambar..."):
            output_img, detections, infer_time = detect_image(model, image)

        # Show annotated image
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown('<p class="label-text">Hasil Deteksi</p>', unsafe_allow_html=True)
        st.image(output_img
