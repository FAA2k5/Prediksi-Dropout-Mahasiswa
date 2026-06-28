import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CEK DAN BUAT FOLDER MODEL
# ============================================
def ensure_model_folder():
    """Pastikan folder model ada"""
    if not os.path.exists('model'):
        os.makedirs('model')
        st.warning("⚠️ Folder 'model' belum ada, telah dibuat otomatis.")
        st.info("📌 Harap upload file model ke folder 'model/'")
        return False
    return True

# ============================================
# LOAD MODEL DARI BEBERAPA LOKASI
# ============================================
@st.cache_resource
def load_models():
    """Load model dari berbagai kemungkinan lokasi"""
    
    # Daftar lokasi yang mungkin
    possible_paths = [
        'model/best_model.pkl',
        'best_model.pkl',
        './model/best_model.pkl',
        '../model/best_model.pkl'
    ]
    
    model = None
    scaler = None
    selector = None
    encoder_target = None
    encoders = {}
    selected_features = []
    
    # Coba cari model di semua lokasi
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error("❌ Model tidak ditemukan! Pastikan file model ada.")
        st.info("📌 Upload file model ke folder yang sesuai.")
        return None, None, None, None, None, None
    
    try:
        # Load model
        model = joblib.load(model_path)
        st.success(f"✅ Model ditemukan di: {model_path}")
        
        # Cari file lainnya
        base_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        
        # Load scaler
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.success("✅ Scaler ditemukan")
        
        # Load selector
        selector_path = os.path.join(base_dir, 'selector.pkl')
        if os.path.exists(selector_path):
            selector = joblib.load(selector_path)
            st.success("✅ Selector ditemukan")
        
        # Load encoder target
        encoder_target_path = os.path.join(base_dir, 'encoder_target.pkl')
        if os.path.exists(encoder_target_path):
            encoder_target = joblib.load(encoder_target_path)
            st.success("✅ Encoder target ditemukan")
        
        # Load semua encoder fitur
        for file in os.listdir(base_dir):
            if file.startswith('encoder_') and file != 'encoder_target.pkl' and file.endswith('.pkl'):
                col_name = file.replace('encoder_', '').replace('.pkl', '')
                encoders[col_name] = joblib.load(os.path.join(base_dir, file))
        
        if encoders:
            st.success(f"✅ {len(encoders)} encoder fitur ditemukan")
        
        # Load selected features
        selected_path = os.path.join(base_dir, 'selected_features.txt')
        if os.path.exists(selected_path):
            with open(selected_path, 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
            st.success(f"✅ {len(selected_features)} fitur terpilih ditemukan")
        else:
            if selector is not None and hasattr(selector, 'feature_names_in_'):
                selected_features = list(selector.feature_names_in_)
                st.info(f"ℹ️ Menggunakan {len(selected_features)} fitur dari selector")
        
        return model, scaler, selector, encoder_target, encoders, selected_features
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None, None

# ============================================
# LOAD MODEL
# ============================================
ensure_model_folder()
model, scaler, selector, encoder_target, encoders, selected_features = load_models()

# ============================================
# FUNGSI PREPROCESSING
# ============================================
def preprocess_input(data):
    """Preprocess input data sebelum diprediksi"""
    df = data.copy()
    
    # Hapus kolom yang tidak diperlukan
    if 'Status_encoded' in df.columns:
        df.drop('Status_encoded', axis=1, inplace=True)
    
    # Pastikan semua fitur ada
    if selected_features:
        df_final = pd.DataFrame()
        for col in selected_features:
            if col in df.columns:
                df_final[col] = df[col]
            else:
                df_final[col] = 0
        df = df_final
    
    # Encode fitur kategorikal
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except:
                df[col] = 0
    
    # Scaling
    if scaler is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Feature selection
    if selector is not None:
        try:
            X_selected = selector.transform(df)
            return X_selected
        except:
            return df.values
    else:
        return df.values

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict(data):
    """Lakukan prediksi"""
    try:
        X_preprocessed = preprocess_input(data)
        prediction = model.predict(X_preprocessed)
        prediction_proba = model.predict_proba(X_preprocessed)
        result = encoder_target.inverse_transform(prediction)
        return result[0], prediction_proba[0]
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None

# ============================================
# TAMPILAN UTAMA
# ============================================
st.title("🎓 Sistem Prediksi Dropout Mahasiswa")

# Cek status model
if model is None:
    st.error("""
    ### ❌ Model Tidak Ditemukan!
    
    **Cara Mengatasi:**
    1. Pastikan folder `model/` ada di repository
    2. Upload file model ke folder `model/`
    3. File yang dibutuhkan:
       - `best_model.pkl`
       - `scaler.pkl`
       - `selector.pkl`
       - `encoder_target.pkl`
       - `encoder_*.pkl`
       - `selected_features.txt`
    
    **Jika menggunakan GitHub:**
    - Pastikan file sudah di-commit dan push
    - Periksa struktur folder di GitHub
    """
    )
    st.stop()

# Lanjutkan dengan tampilan normal...
st.markdown("""
<style>
.info-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid #667eea;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    font-size: 18px;
    padding: 12px 24px;
    border-radius: 10px;
    border: none;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>📋 Jaya Jaya Institut</b><br>
    Sistem deteksi dini untuk mengidentifikasi mahasiswa yang berisiko dropout.
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.markdown("""
    **Status Prediksi:**
    - 🟢 Graduate: Lulus
    - 🟡 Enrolled: Aktif
    - 🔴 Dropout: Berisiko
    
    **Faktor Risiko:**
    1. Nilai Semester 1 rendah
    2. SKS terlalu banyak
    3. Tidak ada beasiswa
    4. Memiliki hutang
    """)
    
    if model is not None:
        st.markdown("---")
        st.markdown("✅ **Model Siap Digunakan**")
    
    st.markdown("---")
    st.caption("© 2024 Jaya Jaya Institut")

# ... (lanjutkan dengan form input dan prediksi seperti sebelumnya)
