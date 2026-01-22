import streamlit as st
from ultralytics import YOLO
from PIL import Image
import yaml
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Deteksi Daun Herbal",
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
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%),
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
   CARD DENGAN GLASSMORPHISM EFFECT
===================================================== */
.card,
.detect-card,
section[data-testid="stFileUploader"],
section[data-testid="stCameraInput"] {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1),
                0 2px 8px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.8);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover,
.detect-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15),
                0 4px 12px rgba(0, 0, 0, 0.08);
}

/* =====================================================
   HEADINGS DENGAN GRADIENT
===================================================== */
h1 {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900;
    font-size: 3rem !important;
    margin-bottom: 1.5rem !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

h2 {
    color: #065f46 !important;
    font-weight: 800;
    font-size: 2rem !important;
    margin-top: 1rem !important;
    margin-bottom: 1rem !important;
}

h3 {
    color: #047857 !important;
    font-weight: 700;
    font-size: 1.5rem !important;
    margin-bottom: 1rem !important;
}

/* =====================================================
   TEXT STYLING
===================================================== */
p, li, label {
    color: #1f2937 !important;
    font-size: 1rem;
    line-height: 1.7;
    font-weight: 500;
}

span {
    color: #374151 !important;
}

/* =====================================================
   CONFIDENCE BADGE MODERN
===================================================== */
.confidence {
    display: inline-block;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff !important;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    transition: transform 0.2s ease;
}

.confidence:hover {
    transform: scale(1.05);
}

/* =====================================================
   RECIPE BOX DENGAN BORDER ANIMASI
===================================================== */
.recipe-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    padding: 1.5rem;
    border-radius: 16px;
    border-left: 5px solid #10b981;
    margin-top: 1rem;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
}

.recipe-box h3 {
    color: #065f46 !important;
    margin-top: 0.5rem !important;
}

.recipe-box p,
.recipe-box li {
    color: #1f2937 !important;
}

/* =====================================================
   SELECTBOX MODERN
===================================================== */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 16px !important;
    border: 2px solid #d1fae5 !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease;
}

div[data-baseweb="select"] > div:hover {
    border-color: #10b981 !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
}

div[data-baseweb="select"] span {
    color: #1f2937 !important;
    font-weight: 600;
}

ul[data-baseweb="menu"] {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(10px);
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    border: 1px solid #d1fae5;
}

ul[data-baseweb="menu"] li {
    color: #1f2937 !important;
    font-weight: 600;
    padding: 0.75rem 1rem !important;
    transition: background 0.2s ease;
}

ul[data-baseweb="menu"] li:hover {
    background: #f0fdf4 !important;
}

/* =====================================================
   FILE UPLOADER ENHANCED
===================================================== */
section[data-testid="stFileUploader"],
section[data-testid="stCameraInput"] {
    border: 2px dashed #10b981 !important;
    background: rgba(240, 253, 244, 0.5) !important;
}

section[data-testid="stFileUploader"]:hover,
section[data-testid="stCameraInput"]:hover {
    border-color: #059669 !important;
    background: rgba(240, 253, 244, 0.8) !important;
}

section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] small,
section[data-testid="stFileUploader"] p,
section[data-testid="stCameraInput"] span,
section[data-testid="stCameraInput"] small,
section[data-testid="stCameraInput"] p {
    color: #1f2937 !important;
    font-weight: 600;
}

/* =====================================================
   SIDEBAR MODERN
===================================================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
    border-right: 2px solid #d1fae5;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #065f46 !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #1f2937 !important;
    font-weight: 600;
}

/* Radio buttons di sidebar */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.8);
    padding: 0.75rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #f0fdf4;
    transform: translateX(4px);
}

/* =====================================================
   IMAGE STYLING
===================================================== */
img {
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
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
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 700;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
}

/* =====================================================
   EXPANDER
===================================================== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-radius: 12px;
    padding: 1rem;
    font-weight: 700;
    color: #065f46 !important;
    border: 2px solid #d1fae5;
}

.streamlit-expanderHeader:hover {
    border-color: #10b981;
}

/* =====================================================
   SUCCESS/WARNING/INFO MESSAGES
===================================================== */
.stSuccess,
.stWarning,
.stInfo {
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

/* =====================================================
   SPINNER
===================================================== */
.stSpinner > div {
    border-color: #10b981 !important;
}

/* =====================================================
   LIST STYLING
===================================================== */
.card ul {
    list-style: none;
    padding-left: 0;
}

.card ul li {
    padding: 0.75rem 0;
    padding-left: 2rem;
    position: relative;
    color: #1f2937 !important;
    font-size: 1.1rem;
}

.card ul li:before {
    content: "🌿";
    position: absolute;
    left: 0;
    font-size: 1.2rem;
}

/* =====================================================
   INPUT FIELDS
===================================================== */
.stTextInput > div > div > input {
    border-radius: 12px;
    border: 2px solid #d1fae5;
    padding: 0.75rem 1rem;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #10b981;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

/* =====================================================
   SCROLLBAR CUSTOM
===================================================== */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
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
   ANIMATION UTILITIES
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
.detect-card {
    animation: fadeIn 0.5s ease-out;
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
# DETECTION FUNCTION
# =====================================================
def detect_image(model, image):
    start = time.time()
    results = model.predict(image, verbose=False)
    infer_time = time.time() - start

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100
            name = class_names[cls_id]
            info = class_info.get(name, {})

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "recipes": info.get("recipes", {})
            })

    return detections, infer_time

# =====================================================
# SIDEBAR MENU
# =====================================================
menu = st.sidebar.radio(
    "🌿 Menu Navigasi",
    ["🏠 Beranda", "🔍 Deteksi Daun", "💊 Rekomendasi Manfaat"]
)

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.title("🌿 Sistem Deteksi Daun Herbal")

    st.markdown("""
    <div class="card">
        <h3 style="color: #065f46 !important; margin-top: 0;">Selamat Datang di Sistem Deteksi Daun Herbal</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">
            Aplikasi ini menggunakan teknologi AI untuk mengenali daun herbal dan memberikan informasi lengkap tentang manfaat serta resep tradisionalnya.
        </p>
        <ul>
            <li>Deteksi otomatis daun herbal menggunakan YOLO</li>
            <li>Pilihan model Nano, Small, dan Medium sesuai kebutuhan</li>
            <li>Menampilkan confidence score hasil deteksi</li>
            <li>Informasi kandungan, manfaat, dan resep tradisional</li>
            <li>Rekomendasi daun herbal berdasarkan manfaat kesehatan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DETEKSI DAUN
# =====================================================
elif menu == "🔍 Deteksi Daun":
    st.title("🔍 Deteksi Daun Herbal")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Pilih Model YOLO")
    model_choice = st.selectbox("", MODEL_PATHS.keys(), label_visibility="collapsed")
    model = load_model(MODEL_PATHS[model_choice])
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload Gambar")
        uploaded = st.file_uploader("", ["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📷 Ambil Foto")
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
            st.markdown("### 🖼️ Gambar Input")
            st.image(image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("🔄 Mendeteksi daun..."):
            detections, infer_time = detect_image(model, image)

        st.success(f"✅ Deteksi selesai! Waktu inferensi: {round(infer_time, 3)} detik")

        if detections:
            for idx, d in enumerate(detections):
                st.markdown(f"""
                <div class="detect-card">
                    <h3>🌿 {d['name']}</h3>
                    <div class="confidence">Confidence: {d['confidence']}%</div>
                    <p><b>🧪 Kandungan:</b> {", ".join(d['components'])}</p>
                    <p><b>💊 Manfaat:</b> {", ".join(d['benefits'])}</p>
                </div>
                """, unsafe_allow_html=True)

                if d["recipes"]:
                    with st.expander("📖 Resep Tradisional", expanded=False):
                        st.markdown("<div class='recipe-box'>", unsafe_allow_html=True)

                        for manfaat, resep in d["recipes"].items():
                            st.markdown(f"### 🍃 {manfaat}")

                            st.markdown("**🥄 Bahan:**")
                            for b in resep.get("ingredients", []):
                                st.markdown(f"- {b}")

                            st.markdown("**👨‍🍳 Cara Membuat:**")
                            for i, step in enumerate(resep.get("steps", []), 1):
                                st.markdown(f"{i}. {step}")

                            if manfaat != list(d["recipes"].keys())[-1]:
                                st.markdown("---")

                        st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("⚠️ Daun tidak terdeteksi. Silakan coba dengan gambar lain.")

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Rekomendasi Berdasarkan Manfaat")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔎 Cari Daun Herbal")
    query = st.text_input("Masukkan manfaat yang dicari (contoh: batuk, maag, demam)", placeholder="Ketik manfaat di sini...")
    st.markdown("</div>", unsafe_allow_html=True)

    if query:
        found = False

        for leaf, info in class_info.items():
            benefits = info.get("benefits", [])
            recipes = info.get("recipes", {})

            if any(query.lower() in b.lower() for b in benefits):
                found = True

                st.markdown(f"""
                <div class="detect-card">
                    <h3>🌿 {leaf}</h3>
                    <p><b>🧪 Kandungan:</b> {", ".join(info.get("components", []))}</p>
                    <p><b>💊 Manfaat:</b> {", ".join(benefits)}</p>
                </div>
                """, unsafe_allow_html=True)

                if recipes:
                    matching_recipes = {k: v for k, v in recipes.items() if query.lower() in k.lower()}
                    
                    if matching_recipes:
                        with st.expander("📖 Resep Tradisional", expanded=False):
                            st.markdown("<div class='recipe-box'>", unsafe_allow_html=True)

                            for manfaat, resep in matching_recipes.items():
                                st.markdown(f"### 🍃 {manfaat}")

                                st.markdown("**🥄 Bahan:**")
                                for b in resep.get("ingredients", []):
                                    st.markdown(f"- {b}")

                                st.markdown("**👨‍🍳 Cara Membuat:**")
                                for i, step in enumerate(resep.get("steps", []), 1):
                                    st.markdown(f"{i}. {step}")

                                if manfaat != list(matching_recipes.keys())[-1]:
                                    st.markdown("---")

                            st.markdown("</div>", unsafe_allow_html=True)

        if not found:
            st.info("ℹ️ Tidak ditemukan daun dengan manfaat tersebut. Coba kata kunci lain.")