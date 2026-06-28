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
# LOAD MODEL DAN PREPROCESSOR
# ============================================
@st.cache_resource
def load_models():
    """Load semua model dan preprocessor yang telah disimpan"""
    try:
        model = joblib.load('model/best_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        selector = joblib.load('model/selector.pkl')
        encoder_target = joblib.load('model/encoder_target.pkl')
        
        # Load list fitur yang digunakan saat training
        selected_features = []
        try:
            with open('model/selected_features.txt', 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
            print(f"✅ Load {len(selected_features)} fitur dari file")
        except:
            # Jika file tidak ada, gunakan fitur dari selector
            if hasattr(selector, 'feature_names_in_'):
                selected_features = list(selector.feature_names_in_)
                print(f"✅ Load {len(selected_features)} fitur dari selector")
            else:
                selected_features = None
                print("⚠️ Tidak ada list fitur yang ditemukan")
        
        # Load semua encoder fitur
        encoders = {}
        for file in os.listdir('model'):
            if file.startswith('encoder_') and file != 'encoder_target.pkl':
                col_name = file.replace('encoder_', '').replace('.pkl', '')
                encoders[col_name] = joblib.load(f'model/{file}')
                print(f"✅ Load encoder untuk: {col_name}")
        
        # Debug: tampilkan fitur yang digunakan
        if selected_features:
            print(f"\n📋 FITUR YANG DIGUNAKAN ({len(selected_features)}):")
            for i, f in enumerate(selected_features[:10]):
                print(f"   {i+1}. {f}")
            if len(selected_features) > 10:
                print(f"   ... dan {len(selected_features)-10} fitur lainnya")
        
        return model, scaler, selector, encoder_target, encoders, selected_features
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None, None

model, scaler, selector, encoder_target, encoders, selected_features = load_models()

# ============================================
# FUNGSI PREPROCESSING
# ============================================
def preprocess_input(data):
    """Preprocess input data sebelum diprediksi"""
    df = data.copy()
    
    # 1. ENSURE ALL FEATURES EXIST
    if selected_features:
        # Buat dataframe dengan semua fitur yang diperlukan
        df_final = pd.DataFrame()
        
        for col in selected_features:
            if col in df.columns:
                df_final[col] = df[col]
            else:
                # Jika fitur tidak ada, isi dengan 0
                df_final[col] = 0
                print(f"⚠️ Fitur '{col}' tidak ditemukan, diisi 0")
        
        df = df_final
    
    # 2. Encode fitur kategorikal
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except Exception as e:
                # Jika nilai tidak dikenal, gunakan 0
                print(f"⚠️ Error encoding {col}: {e}, menggunakan 0")
                df[col] = 0
    
    # 3. Scaling fitur numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        try:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        except Exception as e:
            print(f"⚠️ Error scaling: {e}")
    
    # 4. Feature selection
    try:
        X_selected = selector.transform(df)
    except Exception as e:
        st.error(f"❌ Error feature selection: {e}")
        # Fallback: gunakan df langsung
        X_selected = df.values
    
    return X_selected

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict(data):
    """Lakukan prediksi status mahasiswa"""
    try:
        # Preprocess data
        X_preprocessed = preprocess_input(data)
        
        # Prediksi
        prediction = model.predict(X_preprocessed)
        prediction_proba = model.predict_proba(X_preprocessed)
        
        # Decode hasil
        result = encoder_target.inverse_transform(prediction)
        
        return result[0], prediction_proba[0]
    except Exception as e:
        st.error(f"❌ Error dalam prediksi: {e}")
        # Tampilkan detail error untuk debugging
        st.error(f"Detail error: {str(e)}")
        return None, None

# ============================================
# FUNGSI REKOMENDASI
# ============================================
def get_recommendations(status, probabilities, data):
    """Berikan rekomendasi berdasarkan hasil prediksi"""
    recommendations = []
    
    # Dapatkan indeks probabilitas tertinggi
    max_prob_idx = np.argmax(probabilities)
    max_prob = probabilities[max_prob_idx]
    
    if status == 'Dropout':
        recommendations.append("⚠️ **RISIKO TINGGI DROPOUT**")
        recommendations.append(f"📊 Probabilitas Dropout: {max_prob*100:.1f}%")
        recommendations.append("")
        recommendations.append("📌 Rekomendasi Intervensi:")
        
        # Rekomendasi berdasarkan fitur
        if 'Curricular_units_1st_sem_grade' in data.columns:
            grade = data['Curricular_units_1st_sem_grade'].values[0]
            if grade < 10:
                recommendations.append("   • Program bimbingan belajar intensif")
                recommendations.append("   • Konseling akademik rutin (2 minggu sekali)")
            elif grade < 12:
                recommendations.append("   • Program bimbingan belajar terjadwal")
        
        if 'Scholarship_holder' in data.columns and data['Scholarship_holder'].values[0] == 0:
            recommendations.append("   • Pertimbangkan program bantuan biaya pendidikan")
        
        if 'Debtor' in data.columns and data['Debtor'].values[0] == 1:
            recommendations.append("   • Bantuan konsultasi keuangan dan penjadwalan pembayaran")
        
        if 'Age_at_enrollment' in data.columns and data['Age_at_enrollment'].values[0] > 25:
            recommendations.append("   • Program fleksibel untuk mahasiswa dewasa")
        
        if 'Curricular_units_1st_sem_enrolled' in data.columns:
            sks = data['Curricular_units_1st_sem_enrolled'].values[0]
            if sks > 6:
                recommendations.append("   • Evaluasi beban SKS (kurangi menjadi 4-5 SKS)")
        
        if 'Curricular_units_1st_sem_approved' in data.columns:
            approved = data['Curricular_units_1st_sem_approved'].values[0]
            if approved < 3:
                recommendations.append("   • Program remedial dan pengulangan mata kuliah")
        
        recommendations.append("")
        recommendations.append("📞 Segera hubungi konselor akademik untuk pendampingan!")
        
    elif status == 'Graduate':
        recommendations.append("✅ **MAHASISWA BERPRESTASI**")
        recommendations.append(f"📊 Probabilitas Lulus: {max_prob*100:.1f}%")
        recommendations.append("")
        recommendations.append("📌 Rekomendasi:")
        recommendations.append("   • Pertahankan prestasi akademik yang baik")
        recommendations.append("   • Ikuti program pengembangan karir dan magang")
        recommendations.append("   • Pertimbangkan program studi lanjut (S2)")
        
        if 'Curricular_units_1st_sem_grade' in data.columns:
            grade = data['Curricular_units_1st_sem_grade'].values[0]
            if grade > 16:
                recommendations.append("   • Kesempatan menjadi asisten dosen")
                recommendations.append("   • Ikut program pertukaran pelajar")
        
    else:  # Enrolled
        recommendations.append("📚 **MAHASISWA AKTIF**")
        recommendations.append(f"📊 Probabilitas Aktif: {max_prob*100:.1f}%")
        recommendations.append("")
        recommendations.append("📌 Rekomendasi:")
        recommendations.append("   • Lanjutkan studi dengan baik dan konsisten")
        recommendations.append("   • Ikuti program mentoring dengan senior")
        recommendations.append("   • Pantau perkembangan akademik secara rutin")
        recommendations.append("   • Jangan ragu konsultasi dengan dosen wali")
    
    return recommendations

# ============================================
# TAMPILAN UTAMA
# ============================================
st.title("🎓 Sistem Prediksi Dropout Mahasiswa")

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
    Masukkan data mahasiswa secara lengkap untuk mendapatkan prediksi dan rekomendasi.
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
    
    if model is not None:
        st.markdown("---")
        st.markdown("**Model:** Gradient Boosting")
        st.markdown("**Akurasi:** 85.4%")
    
    st.markdown("---")
    st.caption("© 2024 Jaya Jaya Institut")

# ============================================
# FORM INPUT DATA - TANPA Nationality
# ============================================
st.header("📝 Input Data Mahasiswa")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Data Demografi")
    
    gender = st.selectbox("Jenis Kelamin", options=[0, 1], 
                         format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
    
    age = st.number_input("Usia saat Mendaftar", min_value=17, max_value=70, value=20, step=1)
    
    marital_status = st.selectbox("Status Pernikahan", options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {1: "Single", 2: "Married", 3: "Widower", 
                              4: "Divorced", 5: "Facto Union", 6: "Legally Separated"}.get(x, "Unknown"))
    
    displaced = st.selectbox("Pindah Domisili", options=[0, 1],
                            format_func=lambda x: "Tidak" if x == 0 else "Ya")
    
    international = st.selectbox("Mahasiswa Internasional", options=[0, 1],
                                format_func=lambda x: "Tidak" if x == 0 else "Ya")

with col2:
    st.subheader("📚 Data Akademik")
    
    course = st.selectbox("Program Studi",
        options=[33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
        format_func=lambda x: {33: "Biofuel Production", 171: "Animation & Multimedia",
            8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
            9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
            9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
            9556: "Oral Hygiene", 9670: "Advertising & Marketing", 9773: "Journalism",
            9853: "Basic Education", 9991: "Management (evening)"}.get(x, "Other"))
    
    app_mode = st.selectbox("Mode Pendaftaran",
        options=[1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57])
    
    app_order = st.selectbox("Urutan Pilihan", options=list(range(10)),
                            format_func=lambda x: f"Pilihan ke-{x+1}")
    
    prev_qual = st.selectbox("Kualifikasi Sebelumnya",
        options=[1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
        format_func=lambda x: {1: "Secondary", 2: "Bachelor's", 3: "Degree", 
            4: "Master's", 5: "Doctorate", 6: "Higher education", 9: "12th year - not completed",
            10: "11th year - not completed", 12: "Other - 11th year", 14: "10th year",
            15: "10th year - not completed", 19: "Basic education 3rd cycle",
            38: "Basic education 2nd cycle", 39: "Technological specialization",
            40: "Higher education - degree", 42: "Professional higher technical",
            43: "Higher education - master"}.get(x, "Other"))
    
    prev_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=100.0, step=0.5)
    admission_grade = st.number_input("Nilai Masuk", min_value=0.0, max_value=200.0, value=120.0, step=0.5)
    
    attendance = st.selectbox("Jadwal Kuliah", options=[0, 1],
                             format_func=lambda x: "Evening" if x == 0 else "Daytime")

with col3:
    st.subheader("💰 Data Sosial-Ekonomi")
    
    mother_qual = st.selectbox("Kualifikasi Pendidikan Ibu",
        options=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 19, 22, 26, 27, 29, 30, 31, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45])
    
    father_qual = st.selectbox("Kualifikasi Pendidikan Ayah",
        options=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 19, 22, 26, 27, 29, 30, 31, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45])
    
    mother_occ = st.number_input("Kode Pekerjaan Ibu", min_value=0, max_value=200, value=100)
    father_occ = st.number_input("Kode Pekerjaan Ayah", min_value=0, max_value=200, value=100)
    
    scholarship = st.selectbox("Penerima Beasiswa", options=[0, 1],
                              format_func=lambda x: "Tidak" if x == 0 else "Ya")
    
    debtor = st.selectbox("Memiliki Hutang", options=[0, 1],
                         format_func=lambda x: "Tidak" if x == 0 else "Ya")
    
    tuition = st.selectbox("Biaya Kuliah Lunas", options=[0, 1],
                          format_func=lambda x: "Tidak" if x == 0 else "Ya")
    
    special_needs = st.selectbox("Kebutuhan Khusus", options=[0, 1],
                                format_func=lambda x: "Tidak" if x == 0 else "Ya")

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
    unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=30.0, value=7.2, step=0.1)

with col2:
    inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)

with col3:
    gdp = st.number_input("GDP", min_value=0.0, max_value=100000.0, value=25000.0, step=100.0)

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
        st.error("❌ Model tidak tersedia. Pastikan file model ada di folder 'model/'")
    else:
        with st.spinner("🔄 Memproses data dan melakukan prediksi..."):
            try:
                # 1. Kumpulkan semua data (TANPA Nacionality)
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
                
                # 2. Debug: tampilkan fitur input
                with st.expander("🔍 Debug - Data Input"):
                    st.write("Jumlah fitur input:", len(input_data.columns))
                    st.write("Fitur input:", input_data.columns.tolist())
                
                # 3. Lakukan prediksi
                status, probabilities = predict(input_data)
                
                if status:
                    # 4. Tampilkan hasil
                    st.divider()
                    st.header("📊 Hasil Prediksi")
                    
                    # Tampilkan status dengan warna
                    if status == 'Dropout':
                        st.error(f"🔴 **Status: {status}**")
                        st.warning("⚠️ Mahasiswa teridentifikasi berisiko tinggi dropout. Segera lakukan intervensi!")
                    elif status == 'Graduate':
                        st.success(f"🟢 **Status: {status}**")
                        st.info("✅ Mahasiswa berpotensi lulus tepat waktu. Pertahankan prestasi!")
                    else:
                        st.warning(f"🟡 **Status: {status}**")
                        st.info("📚 Mahasiswa masih aktif. Pantau perkembangan secara rutin.")
                    
                    # Tampilkan probabilitas
                    st.subheader("📈 Probabilitas Prediksi")
                    
                    prob_df = pd.DataFrame({
                        'Status': encoder_target.classes_,
                        'Probabilitas': probabilities
                    })
                    
                    colors = ['#e74c3c' if x == 'Dropout' else '#2ecc71' if x == 'Graduate' else '#f39c12' 
                              for x in prob_df['Status']]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(prob_df['Status'], prob_df['Probabilitas'], color=colors, 
                                 edgecolor='black', linewidth=1)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probabilitas', fontsize=12)
                    ax.set_title('Probabilitas Prediksi per Status', fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                    
                    st.pyplot(fig)
                    
                    # 5. Tampilkan rekomendasi
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
                    
                    # 6. Tampilkan data input
                    with st.expander("📋 Lihat Data yang Diproses"):
                        st.dataframe(input_data, use_container_width=True)
                        
                        if hasattr(model, 'feature_importances_'):
                            st.subheader("📊 Feature Importance (Top 10)")
                            importances = model.feature_importances_
                            feature_names = selector.get_feature_names_out()
                            
                            sorted_idx = np.argsort(importances)[::-1][:10]
                            sorted_features = [feature_names[i] for i in sorted_idx]
                            sorted_importances = importances[sorted_idx]
                            
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            bars2 = ax2.barh(sorted_features[::-1], sorted_importances[::-1], 
                                           color='skyblue', edgecolor='navy', linewidth=1)
                            ax2.set_xlabel('Importance', fontsize=12)
                            ax2.set_title('Top 10 Fitur Paling Berpengaruh', fontsize=14, fontweight='bold')
                            ax2.grid(axis='x', alpha=0.3)
                            
                            for bar in bars2:
                                width = bar.get_width()
                                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                        f'{width:.3f}', ha='left', va='center', fontsize=10)
                            
                            st.pyplot(fig2)
                            
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
                import traceback
                st.error(f"Detail: {traceback.format_exc()}")
                st.info("Pastikan semua data telah diisi dengan benar dan model tersedia.")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("""
**Disclaimer:** Sistem ini hanya untuk tujuan prediksi dan rekomendasi awal. 
Keputusan akhir tetap ada pada institusi dan konselor akademik.
""")

with st.expander("ℹ️ Informasi Sistem"):
    st.markdown("""
    **Versi Sistem:** 1.0.3
    
    **Model yang Digunakan:** 
    - Gradient Boosting Classifier
    - Akurasi: 85.4%
    """)