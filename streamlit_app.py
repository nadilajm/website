import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
import time

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
# CSS STYLING
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.block-container {
    max-width: 1400px;
    padding: 3rem 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #059669 0%, #047857 100%) !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
    font-weight: 700;
}

.card {
    background: white;
    padding: 2rem;
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    margin-bottom: 1.5rem;
}

h1, h2, h3 {
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.card h1, .card h2, .card h3 {
    color: #111827 !important;
    text-shadow: none;
}

section[data-testid="stFileUploader"],
section[data-testid="stCameraInput"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    padding: 2rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
}

.main label {
    color: white !important;
    font-weight: 700;
}

section[data-testid="stFileUploader"] label,
section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] p,
section[data-testid="stFileUploader"] small,
section[data-testid="stCameraInput"] label,
section[data-testid="stCameraInput"] span,
section[data-testid="stCameraInput"] p,
section[data-testid="stCameraInput"] small {
    color: white !important;
    font-weight: 600;
}

section[data-testid="stTextInput"] input {
    background: white !important;
    color: #111827 !important;
    border: 2px solid white !important;
    border-radius: 12px;
}

.stButton > button {
    background: white !important;
    color: #047857 !important;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 700;
}

.stButton > button:hover {
    background: #f0fdf4 !important;
}

.hero-title {
    font-size: 4rem;
    font-weight: bold;
    color: #639872;
    text-shadow: 3px 3px 0 #0B6A43, 6px 6px 0 #0B6A43;
    text-align: center;
    margin-bottom: 2rem;
}

.hero-content {
    background: rgba(255,255,255,0.95);
    padding: 3rem;
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

.upload-box {
    background-color: #10b981;
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    font-weight: 700;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD YAML DATA
# =====================================================
@st.cache_data
def load_yaml():
    try:
        with open("data-baru.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except:
        return {"names": [], "info": {}}

yaml_data = load_yaml()
class_names = yaml_data.get("names", [])
class_info = yaml_data.get("info", {})

# =====================================================
# LOAD YOLO MODEL
# =====================================================
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except:
        return None

MODEL_PATHS = {
    "Best Model": "best.pt",
    "YOLOv8n (Nano)": "models/bestnano.pt",
    "YOLOv8s (Small)": "models/bestsmall.pt",
    "YOLOv8m (Medium)": "models/yolo_medium.pt"
}

# =====================================================
# DETECTION FUNCTION
# =====================================================
def detect_and_draw(model, image):
    if model is None:
        return image, [], 0
    
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

            x1, y1, x2, y2 = [int(c) for c in box]
            draw.rectangle([x1, y1, x2, y2], outline="white", width=4)

            label = f"{name} {conf:.1f}%"
            try:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                w = text_bbox[2] - text_bbox[0]
                h = text_bbox[3] - text_bbox[1]
            except:
                w, h = 100, 20

            draw.rectangle([x1, y1 - h - 6, x1 + w + 6, y1], fill="white")
            draw.text((x1 + 3, y1 - h - 3), label, fill="#065f46", font=font)

    return image, detections, infer_time

# =====================================================
# REMOVE DUPLICATES
# =====================================================
def remove_duplicate_detections(detections):
    unique = {}
    for d in detections:
        if d['name'] not in unique:
            unique[d['name']] = d
    return list(unique.values())

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### 🌿 MENU NAVIGASI")
    menu = st.radio(
        "",
        ["🏠 Beranda", "📸 Deteksi Kamera", "📤 Deteksi Upload", "💊 Rekomendasi Manfaat"],
        label_visibility="collapsed"
    )

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.markdown('<h1 class="hero-title">🌿 HERBAL LEAF DETECTION</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-content">
        <h2 style="color: #047857; text-align: center;">Sistem Deteksi Daun Herbal Berbasis Deep Learning</h2>
        <p style="color: #111827; text-align: center; font-size: 1.1rem; line-height: 1.8;">
            <strong>HerbaSmartAI</strong> adalah website yang dirancang untuk membantu mengidentifikasi 
            dan mengklasifikasikan daun herbal melalui deteksi gambar menggunakan teknologi 
            <strong>YOLOv8 (You Only Look Once)</strong>.
        </p>
        
        <hr style="border-color: #10b981; margin: 2rem 0;">
        
        <h3 style="color: #047857;">✨ Fitur Utama:</h3>
        <ul style="color: #111827; font-size: 1rem; line-height: 2;">
            <li><strong>Deteksi Kamera:</strong> Ambil foto langsung dari kamera</li>
            <li><strong>Deteksi Upload:</strong> Upload foto daun untuk analisis</li>
            <li><strong>Informasi Lengkap:</strong> Kandungan, manfaat, dan resep pengobatan</li>
            <li><strong>Rekomendasi:</strong> Cari daun berdasarkan manfaat kesehatan</li>
        </ul>
        
        <h3 style="color: #047857; margin-top: 2rem;">🎯 Cara Menggunakan:</h3>
        <ol style="color: #111827; font-size: 1rem; line-height: 2;">
            <li>Pilih menu di sidebar</li>
            <li>Ambil foto atau upload gambar</li>
            <li>Lihat hasil deteksi dan informasi lengkap</li>
            <li>Klik tombol resep untuk detail pengobatan</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.9); padding: 2rem; border-radius: 16px;">
        <h3 style="color: #047857; text-shadow: none;">📖 About Us</h3>
        <p style="color: #111827; font-size: 1rem; line-height: 1.8;">
            HerbaSmartAI is a website designed to help identify and classify herbal leaves through 
            image detection using YOLOv8 deep learning technology. We provide comprehensive information 
            about medicinal plants including their chemical components, health benefits, and traditional 
            healing recipes.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DETEKSI KAMERA
# =====================================================
elif menu == "📸 Deteksi Kamera":
    st.title("📸 Deteksi Daun Herbal Menggunakan YOLOv8")
    
    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])
    
    if model is None:
        st.error("❌ Model tidak dapat dimuat")
        st.stop()
    
    st.markdown('<div class="upload-box">📸 Ambil foto daun herbal</div>', unsafe_allow_html=True)
    
    camera = st.camera_input("Ambil Foto", label_visibility="collapsed")
    
    if camera:
        image = Image.open(camera).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Gambar Asli")
            st.image(image, use_container_width=True)
        
        with st.spinner("🔄 Mendeteksi..."):
            result_img, detections, infer_time = detect_and_draw(model, image.copy())
        
        with col2:
            st.markdown("### Hasil Deteksi")
            st.image(result_img, use_container_width=True)
        
        st.success(f"✅ Deteksi selesai dalam {infer_time:.3f} detik")
        
        if detections:
            unique_detections = remove_duplicate_detections(detections)
            
            st.markdown("---")
            st.markdown("### 📋 Nama Daun yang Terdeteksi")
            
            for idx, d in enumerate(unique_detections, 1):
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h4 style="color: #047857; margin: 0;">🌿 {d['name']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if d['gambar']:
                        try:
                            st.image(d['gambar'], use_container_width=True)
                        except:
                            st.info("📷 Gambar tidak tersedia")
                
                with col2:
                    if d['components']:
                        st.markdown("**🧪 Komponen:**")
                        st.write(", ".join(d['components']))
                    
                    if d['benefits']:
                        st.markdown("**💊 Manfaat:**")
                        st.write(", ".join(d['benefits']))
                    
                    if d['recipes']:
                        st.markdown("**📖 Resep Pengobatan:**")
                        for benefit, recipe in d['recipes'].items():
                            if st.button(f"📋 Resep: {benefit}", key=f"cam_{d['name']}_{benefit}_{idx}"):
                                st.info(f"**Resep untuk {benefit}**")
                                st.markdown("**🥣 Bahan:**")
                                for bahan in recipe['bahan']:
                                    st.write(f"• {bahan}")
                                st.markdown("**👨‍🍳 Langkah:**")
                                for i, langkah in enumerate(recipe['langkah'], 1):
                                    st.write(f"{i}. {langkah}")
                
                st.markdown("---")

# =====================================================
# DETEKSI UPLOAD
# =====================================================
elif menu == "📤 Deteksi Upload":
    st.title("📤 Deteksi Daun Herbal Menggunakan YOLOv8 (Upload)")
    
    model_choice = st.selectbox("Pilih Model YOLO", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])
    
    if model is None:
        st.error("❌ Model tidak dapat dimuat")
        st.stop()
    
    st.markdown('<div class="upload-box">📤 Upload gambar daun herbal</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Gambar Asli")
            st.image(image, use_container_width=True)
        
        with st.spinner("🔄 Mendeteksi..."):
            result_img, detections, infer_time = detect_and_draw(model, image.copy())
        
        with col2:
            st.markdown("### Hasil Deteksi")
            st.image(result_img, use_container_width=True)
        
        st.success(f"✅ Deteksi selesai dalam {infer_time:.3f} detik")
        
        if detections:
            unique_detections = remove_duplicate_detections(detections)
            
            st.markdown("---")
            st.markdown("### 📋 Nama Daun yang Terdeteksi")
            
            for idx, d in enumerate(unique_detections, 1):
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h4 style="color: #047857; margin: 0;">🌿 {d['name']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if d['gambar']:
                        try:
                            st.image(d['gambar'], use_container_width=True)
                        except:
                            st.info("📷 Gambar tidak tersedia")
                
                with col2:
                    if d['components']:
                        st.markdown("**🧪 Komponen:**")
                        st.write(", ".join(d['components']))
                    
                    if d['benefits']:
                        st.markdown("**💊 Manfaat:**")
                        st.write(", ".join(d['benefits']))
                    
                    if d['recipes']:
                        st.markdown("**📖 Resep Pengobatan:**")
                        for benefit, recipe in d['recipes'].items():
                            if st.button(f"📋 Resep: {benefit}", key=f"upl_{d['name']}_{benefit}_{idx}"):
                                st.info(f"**Resep untuk {benefit}**")
                                st.markdown("**🥣 Bahan:**")
                                for bahan in recipe['bahan']:
                                    st.write(f"• {bahan}")
                                st.markdown("**👨‍🍳 Langkah:**")
                                for i, langkah in enumerate(recipe['langkah'], 1):
                                    st.write(f"{i}. {langkah}")
                
                st.markdown("---")

# =====================================================
# REKOMENDASI
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Rekomendasi Daun Herbal")
    
    st.markdown("""
    <div class="card">
        <h3>Cari Daun Herbal Berdasarkan Manfaat</h3>
        <p>Masukkan gejala atau kondisi kesehatan</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("", placeholder="Ketik gejala atau penyakit...", label_visibility="collapsed")
    with col2:
        search_btn = st.button("🔍 Cari", use_container_width=True)
    
    if query or search_btn:
        results = []
        for leaf, info in class_info.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                results.append({"name": leaf, "info": info})
        
        if results:
            st.success(f"✅ Ditemukan {len(results)} rekomendasi daun untuk \"{query}\"")
            st.markdown("---")
            
            for idx, result in enumerate(results, 1):
                st.markdown(f"""
                <div style="background: white; padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h4 style="color: #047857;">🌿 {result['name']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if result['info'].get('gambar'):
                        try:
                            st.image(result['info']['gambar'], use_container_width=True)
                        except:
                            st.info("📷 Gambar tidak tersedia")
                
                with col2:
                    if result['info'].get('components'):
                        st.markdown("**🧪 Komponen:**")
                        st.write(", ".join(result['info']['components']))
                    
                    if result['info'].get('benefits'):
                        st.markdown("**💊 Manfaat:**")
                        st.write(", ".join(result['info']['benefits']))
                    
                    if result['info'].get('recipes'):
                        st.markdown("**📖 Resep Pengobatan:**")
                        for benefit, recipe in result['info']['recipes'].items():
                            if st.button(f"📋 {benefit}", key=f"rec_{result['name']}_{benefit}_{idx}"):
                                st.info(f"**Resep untuk {benefit}**")
                                st.markdown("**🥣 Bahan-bahan:**")
                                for bahan in recipe.get('ingredients', []):
                                    st.write(f"• {bahan}")
                                st.markdown("**👨‍🍳 Cara Membuat:**")
                                for i, langkah in enumerate(recipe.get('steps', []), 1):
                                    st.write(f"{i}. {langkah}")
                
                st.markdown("---")
        else:
            st.warning(f"⚠️ Tidak ditemukan rekomendasi untuk \"{query}\"")
            st.info("💡 Coba: batuk, demam, diabetes, hipertensi, kolesterol")

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
