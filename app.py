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
# LOAD MODEL DARI ROOT DIRECTORY
# ============================================
@st.cache_resource
def load_models():
    """Load semua model dari root directory"""
    try:
        # Cari file di root directory (tempat app.py berada)
        base_dir = '.'
        
        # Load model
        model = joblib.load(os.path.join(base_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
        selector = joblib.load(os.path.join(base_dir, 'selector.pkl'))
        encoder_target = joblib.load(os.path.join(base_dir, 'encoder_target.pkl'))
        
        # Load selected features
        selected_features = []
        try:
            with open(os.path.join(base_dir, 'selected_features.txt'), 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
        except:
            if hasattr(selector, 'feature_names_in_'):
                selected_features = list(selector.feature_names_in_)
        
        # Load encoders
        encoders = {}
        for file in os.listdir(base_dir):
            if file.startswith('encoder_') and file != 'encoder_target.pkl' and file.endswith('.pkl'):
                col_name = file.replace('encoder_', '').replace('.pkl', '')
                encoders[col_name] = joblib.load(os.path.join(base_dir, file))
        
        return model, scaler, selector, encoder_target, encoders, selected_features
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None, None

# Load model
model, scaler, selector, encoder_target, encoders, selected_features = load_models()

# ============================================
# FUNGSI PREPROCESSING
# ============================================
def preprocess_input(data):
    """Preprocess input data"""
    df = data.copy()
    
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
            return selector.transform(df)
        except:
            return df.values
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
        return None, None

# ============================================
# FUNGSI REKOMENDASI
# ============================================
def get_recommendations(status, probabilities, data):
    """Berikan rekomendasi"""
    recommendations = []
    max_prob = max(probabilities)
    
    if status == 'Dropout':
        recommendations.append("⚠️ **RISIKO TINGGI DROPOUT**")
        recommendations.append(f"📊 Probabilitas Dropout: {max_prob*100:.1f}%")
        recommendations.append("")
        recommendations.append("📌 Rekomendasi Intervensi:")
        
        if 'Curricular_units_1st_sem_grade' in data.columns:
            grade = data['Curricular_units_1st_sem_grade'].values[0]
            if grade < 10:
                recommendations.append("   • Program bimbingan belajar intensif")
                recommendations.append("   • Konseling akademik rutin")
        
        if 'Scholarship_holder' in data.columns and data['Scholarship_holder'].values[0] == 0:
            recommendations.append("   • Pertimbangkan program bantuan biaya pendidikan")
        
        if 'Debtor' in data.columns and data['Debtor'].values[0] == 1:
            recommendations.append("   • Bantuan konsultasi keuangan")
        
        if 'Age_at_enrollment' in data.columns and data['Age_at_enrollment'].values[0] > 25:
            recommendations.append("   • Program fleksibel untuk mahasiswa dewasa")
        
        recommendations.append("")
        recommendations.append("📞 Segera hubungi konselor akademik!")
        
    elif status == 'Graduate':
        recommendations.append("✅ **MAHASISWA BERPRESTASI**")
        recommendations.append(f"📊 Probabilitas Lulus: {max_prob*100:.1f}%")
        recommendations.append("")
        recommendations.append("📌 Rekomendasi:")
        recommendations.append("   • Pertahankan prestasi akademik")
        recommendations.append("   • Ikuti program pengembangan karir")
        
    else:
        recommendations.append("📚 **MAHASISWA AKTIF**")
        recommendations.append(f"📊 Probabilitas Aktif: {max_prob*100:.1f}%")
        recommendations.append("")
        recommendations.append("📌 Rekomendasi:")
        recommendations.append("   • Lanjutkan studi dengan baik")
        recommendations.append("   • Ikuti program mentoring")
    
    return recommendations

# ============================================
# TAMPILAN UTAMA
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px 20px;
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
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Sistem Prediksi Dropout Mahasiswa</h1>
    <p style="font-size: 18px; opacity: 0.9;">Jaya Jaya Institut</p>
</div>
""", unsafe_allow_html=True)

# Cek model
if model is None:
    st.error("""
    ### ❌ Model Tidak Ditemukan!
    
    **Cara Mengatasi:**
    1. Upload semua file model ke root directory (satu folder dengan app.py)
    2. File yang dibutuhkan:
       - `best_model.pkl`
       - `scaler.pkl`
       - `selector.pkl`
       - `encoder_target.pkl`
       - `encoder_*.pkl` (semua file encoder)
       - `selected_features.txt`
    
    **Cara Upload di GitHub:**
    1. Buka repository di GitHub
    2. Klik `Add file` → `Upload files`
    3. Drag and drop semua file model
    4. Klik `Commit changes`
    """)
    st.stop()

# Info singkat
st.markdown("""
<div class="info-box">
    <b>📋 Deteksi Dini Risiko Dropout</b><br>
    Sistem ini membantu mengidentifikasi mahasiswa yang berisiko dropout 
    berdasarkan data akademik, demografi, dan sosial-ekonomi.
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
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
    st.markdown("---")
    st.caption("© 2024 Jaya Jaya Institut")

# ============================================
# FORM INPUT DATA
# ============================================
st.header("📝 Input Data Mahasiswa")

# Buat 3 kolom untuk input
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Data Demografi")
    
    gender = st.selectbox(
        "Jenis Kelamin",
        options=[0, 1],
        format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki"
    )
    
    age = st.number_input(
        "Usia saat Mendaftar",
        min_value=17,
        max_value=70,
        value=20,
        step=1
    )
    
    marital_status = st.selectbox(
        "Status Pernikahan",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            1: "Single", 2: "Married", 3: "Widower",
            4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
        }.get(x, "Unknown")
    )
    
    displaced = st.selectbox(
        "Pindah Domisili",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    
    international = st.selectbox(
        "Mahasiswa Internasional",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

with col2:
    st.subheader("📚 Data Akademik")
    
    course = st.selectbox(
        "Program Studi",
        options=[33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
        format_func=lambda x: {
            33: "Biofuel Production", 171: "Animation & Multimedia",
            8014: "Social Service (evening)", 9003: "Agronomy",
            9070: "Communication Design", 9085: "Veterinary Nursing",
            9119: "Informatics Engineering", 9130: "Equinculture",
            9147: "Management", 9238: "Social Service",
            9254: "Tourism", 9500: "Nursing",
            9556: "Oral Hygiene", 9670: "Advertising & Marketing",
            9773: "Journalism", 9853: "Basic Education",
            9991: "Management (evening)"
        }.get(x, "Other")
    )
    
    app_mode = st.selectbox(
        "Mode Pendaftaran",
        options=[1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57]
    )
    
    app_order = st.selectbox(
        "Urutan Pilihan",
        options=list(range(10)),
        format_func=lambda x: f"Pilihan ke-{x+1}"
    )
    
    prev_qual = st.selectbox(
        "Kualifikasi Sebelumnya",
        options=[1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
        format_func=lambda x: {
            1: "Secondary", 2: "Bachelor's", 3: "Degree",
            4: "Master's", 5: "Doctorate", 6: "Higher education",
            9: "12th year - not completed", 10: "11th year - not completed",
            12: "Other - 11th year", 14: "10th year",
            15: "10th year - not completed", 19: "Basic education 3rd cycle",
            38: "Basic education 2nd cycle", 39: "Technological specialization",
            40: "Higher education - degree", 42: "Professional higher technical",
            43: "Higher education - master"
        }.get(x, "Other")
    )
    
    prev_grade = st.number_input(
        "Nilai Kualifikasi Sebelumnya",
        min_value=0.0,
        max_value=200.0,
        value=100.0,
        step=0.5
    )
    
    admission_grade = st.number_input(
        "Nilai Masuk",
        min_value=0.0,
        max_value=200.0,
        value=120.0,
        step=0.5
    )
    
    attendance = st.selectbox(
        "Jadwal Kuliah",
        options=[0, 1],
        format_func=lambda x: "Evening" if x == 0 else "Daytime"
    )

with col3:
    st.subheader("💰 Data Sosial-Ekonomi")
    
    mother_qual = st.selectbox(
        "Kualifikasi Pendidikan Ibu",
        options=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 19, 22, 26, 27, 29, 30, 31, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    )
    
    father_qual = st.selectbox(
        "Kualifikasi Pendidikan Ayah",
        options=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 19, 22, 26, 27, 29, 30, 31, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    )
    
    mother_occ = st.number_input(
        "Kode Pekerjaan Ibu",
        min_value=0,
        max_value=200,
        value=100
    )
    
    father_occ = st.number_input(
        "Kode Pekerjaan Ayah",
        min_value=0,
        max_value=200,
        value=100
    )
    
    scholarship = st.selectbox(
        "Penerima Beasiswa",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    
    debtor = st.selectbox(
        "Memiliki Hutang",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    
    tuition = st.selectbox(
        "Biaya Kuliah Lunas",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
    
    special_needs = st.selectbox(
        "Kebutuhan Khusus",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

# ============================================
# DATA AKADEMIK SEMESTER
# ============================================
st.header("📊 Data Akademik Semester")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 Semester 1")
    s1_credited = st.number_input("SKS Dikreditkan S1", min_value=0, max_value=20, value=0)
    s1_enrolled = st.number_input("SKS Diambil S1", min_value=0, max_value=12, value=6)
    s1_evaluations = st.number_input("Jumlah Evaluasi S1", min_value=0, max_value=20, value=6)
    s1_approved = st.number_input("SKS Lulus S1", min_value=0, max_value=12, value=5)
    s1_grade = st.number_input("Nilai Rata-rata S1", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    s1_without_eval = st.number_input("SKS Tanpa Evaluasi S1", min_value=0, max_value=10, value=0)

with col2:
    st.subheader("📗 Semester 2")
    s2_credited = st.number_input("SKS Dikreditkan S2", min_value=0, max_value=20, value=0)
    s2_enrolled = st.number_input("SKS Diambil S2", min_value=0, max_value=12, value=6)
    s2_evaluations = st.number_input("Jumlah Evaluasi S2", min_value=0, max_value=20, value=6)
    s2_approved = st.number_input("SKS Lulus S2", min_value=0, max_value=12, value=5)
    s2_grade = st.number_input("Nilai Rata-rata S2", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    s2_without_eval = st.number_input("SKS Tanpa Evaluasi S2", min_value=0, max_value=10, value=0)

# ============================================
# DATA EKONOMI MAKRO
# ============================================
st.header("📈 Data Ekonomi Makro")

col1, col2, col3 = st.columns(3)

with col1:
    unemployment_rate = st.number_input(
        "Tingkat Pengangguran (%)",
        min_value=0.0,
        max_value=30.0,
        value=7.2,
        step=0.1
    )

with col2:
    inflation_rate = st.number_input(
        "Tingkat Inflasi (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.5,
        step=0.1
    )

with col3:
    gdp = st.number_input(
        "GDP",
        min_value=0.0,
        max_value=100000.0,
        value=25000.0,
        step=100.0
    )

# ============================================
# TOMBOL PREDIKSI
# ============================================
st.divider()
predict_btn = st.button("🔮 Prediksi Status Mahasiswa", use_container_width=True, type="primary")

# ============================================
# PROSES PREDIKSI
# ============================================
if predict_btn:
    if model is None:
        st.error("❌ Model tidak tersedia. Pastikan file model ada di root directory.")
    else:
        with st.spinner("🔄 Memproses data..."):
            try:
                # Kumpulkan data
                input_data = pd.DataFrame({
                    'Marital_status': [marital_status],
                    'Application_mode': [app_mode],
                    'Application_order': [app_order],
                    'Course': [course],
                    'Daytime_evening_attendance': [attendance],
                    'Previous_qualification': [prev_qual],
                    'Previous_qualification_grade': [prev_grade],
                    'Mothers_qualification': [mother_qual],
                    'Fathers_qualification': [father_qual],
                    'Mothers_occupation': [mother_occ],
                    'Fathers_occupation': [father_occ],
                    'Admission_grade': [admission_grade],
                    'Displaced': [displaced],
                    'Educational_special_needs': [special_needs],
                    'Debtor': [debtor],
                    'Tuition_fees_up_to_date': [tuition],
                    'Gender': [gender],
                    'Scholarship_holder': [scholarship],
                    'Age_at_enrollment': [age],
                    'International': [international],
                    'Curricular_units_1st_sem_credited': [s1_credited],
                    'Curricular_units_1st_sem_enrolled': [s1_enrolled],
                    'Curricular_units_1st_sem_evaluations': [s1_evaluations],
                    'Curricular_units_1st_sem_approved': [s1_approved],
                    'Curricular_units_1st_sem_grade': [s1_grade],
                    'Curricular_units_1st_sem_without_evaluations': [s1_without_eval],
                    'Curricular_units_2nd_sem_credited': [s2_credited],
                    'Curricular_units_2nd_sem_enrolled': [s2_enrolled],
                    'Curricular_units_2nd_sem_evaluations': [s2_evaluations],
                    'Curricular_units_2nd_sem_approved': [s2_approved],
                    'Curricular_units_2nd_sem_grade': [s2_grade],
                    'Curricular_units_2nd_sem_without_evaluations': [s2_without_eval],
                    'Unemployment_rate': [unemployment_rate],
                    'Inflation_rate': [inflation_rate],
                    'GDP': [gdp]
                })
                
                # Prediksi
                status, probabilities = predict(input_data)
                
                if status:
                    st.divider()
                    st.header("📊 Hasil Prediksi")
                    
                    # Tampilkan status
                    if status == 'Dropout':
                        st.error(f"🔴 **Status: {status}**")
                        st.warning("⚠️ Mahasiswa teridentifikasi berisiko tinggi dropout. Segera lakukan intervensi!")
                    elif status == 'Graduate':
                        st.success(f"🟢 **Status: {status}**")
                        st.info("✅ Mahasiswa berpotensi lulus tepat waktu.")
                    else:
                        st.warning(f"🟡 **Status: {status}**")
                        st.info("📚 Mahasiswa masih aktif. Pantau perkembangan secara rutin.")
                    
                    # Probabilitas
                    st.subheader("📈 Probabilitas Prediksi")
                    
                    prob_df = pd.DataFrame({
                        'Status': encoder_target.classes_,
                        'Probabilitas': probabilities
                    })
                    
                    colors = ['#e74c3c' if x == 'Dropout' else '#2ecc71' if x == 'Graduate' else '#f39c12' 
                              for x in prob_df['Status']]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(prob_df['Status'], prob_df['Probabilitas'], color=colors)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probabilitas')
                    ax.set_title('Probabilitas Prediksi per Status')
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Rekomendasi
                    st.divider()
                    st.header("💡 Rekomendasi")
                    
                    recommendations = get_recommendations(status, probabilities, input_data)
                    for rec in recommendations:
                        if rec.startswith("⚠️") or rec.startswith("✅") or rec.startswith("📚"):
                            st.markdown(f"### {rec}")
                        elif rec.startswith("📊"):
                            st.info(rec)
                        elif rec.startswith("📌"):
                            st.markdown(f"**{rec}**")
                        elif rec.startswith("   •"):
                            st.markdown(f"- {rec.strip()}")
                        else:
                            st.write(rec)
                    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("""
**Disclaimer:** Sistem ini hanya untuk tujuan prediksi dan rekomendasi awal. 
Keputusan akhir tetap pada institusi dan konselor akademik.
""")
