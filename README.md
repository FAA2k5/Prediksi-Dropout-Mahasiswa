# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

### Latar Belakang
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini, institusi ini telah mencetak banyak lulusan dengan reputasi yang sangat baik. Namun, terdapat masalah serius yang dihadapi yaitu tingginya angka mahasiswa yang tidak menyelesaikan pendidikannya (dropout).

Tingkat dropout yang tinggi ini menjadi masalah besar karena:
- Menurunkan reputasi institusi di mata masyarakat
- Mengurangi pendapatan institusi dari biaya pendidikan
- Mempengaruhi akreditasi program studi
- Menghambat perkembangan karir mahasiswa yang drop out
- Menurunkan kepercayaan orang tua dan calon mahasiswa

### Permasalahan Bisnis
Berdasarkan latar belakang di atas, permasalahan bisnis yang ingin diselesaikan adalah:
1. **Identifikasi Dini Mahasiswa Berisiko Dropout**: Bagaimana cara mendeteksi secara dini mahasiswa yang berpotensi melakukan dropout sehingga dapat diberikan intervensi tepat waktu?
2. **Faktor-faktor yang Mempengaruhi Dropout**: Faktor-faktor apa saja yang paling berpengaruh terhadap keputusan mahasiswa untuk dropout? Apakah faktor akademik, demografi, atau sosial-ekonomi yang paling dominan?
3. **Monitoring Performa Akademik**: Bagaimana memonitor performa akademik mahasiswa secara real-time untuk mengidentifikasi tren penurunan prestasi sebelum terlambat?
4. **Strategi Intervensi Efektif**: Strategi intervensi apa yang paling efektif untuk mencegah dropout berdasarkan profil risiko mahasiswa?

### Cakupan Proyek
Untuk menjawab permasalahan bisnis di atas, proyek ini akan mencakup:
1. **Analisis Data Eksploratif (EDA)**: Menganalisis dataset untuk memahami karakteristik mahasiswa dan faktor-faktor yang mempengaruhi dropout.
2. **Pengembangan Model Machine Learning**: Membangun model klasifikasi untuk memprediksi status mahasiswa (dropout, enrolled, graduate) menggunakan berbagai algoritma dengan hyperparameter tuning.
3. **Business Dashboard**: Membuat dashboard interaktif menggunakan Metabase untuk memonitor performa mahasiswa dan faktor-faktor risiko dropout secara real-time.
4. **Prototype Sistem Prediksi**: Mengembangkan prototype berbasis Streamlit untuk prediksi status mahasiswa secara real-time yang dapat diakses oleh stakeholder.
5. **Rekomendasi Action Items**: Memberikan rekomendasi strategis berdasarkan hasil analisis untuk mengurangi angka dropout.

## Persiapan

### Sumber Data
Dataset ini berisi informasi mahasiswa dari berbagai program studi yang mencakup:
- **Data Demografi**: Usia, gender, status pernikahan, kebangsaan
- **Data Akademik**: Nilai masuk, nilai per semester, jumlah SKS, kualifikasi sebelumnya
- **Data Sosial-Ekonomi**: Pekerjaan orang tua, pendidikan orang tua, status beasiswa, hutang
- **Data Ekonomi Makro**: Tingkat pengangguran, inflasi, GDP
- **Status Akhir**: Dropout, Enrolled, Graduate

Data dengan status `Enrolled` tidak digunakan untuk training model karena belum memiliki label akhir. Model hanya memprediksi dua kelas: **Dropout** dan **Graduate**.

**Link Dataset**: https://docs.google.com/spreadsheets/d/1_TPQP7Z-r-DSZFEnV50uJYevCZ07dTKfG3-EPHISgNs/export?format=csv

### Setup Environment

#### 1. Buat Virtual Environment
```bash
# Menggunakan conda
conda create --name student_performance python=3.9
conda activate student_performance

# Atau menggunakan pipenv
pipenv install
pipenv shell
```

#### 2. Install Library yang Dibutuhkan
```bash
pip install -r requirements.txt
```

#### 3. Isi requirements.txt
asttokens==3.0.1
colorama==0.4.6
comm==0.2.3
contourpy==1.3.3
cycler==0.12.1
debugpy==1.8.21
decorator==5.3.1
executing==2.2.1
fonttools==4.63.0
ipykernel==7.3.0
ipython==9.15.0
ipython_pygments_lexers==1.1.1
jedi==0.20.0
joblib==1.5.3
jupyter_client==8.9.1
jupyter_core==5.9.1
kiwisolver==1.5.0
matplotlib==3.11.0
matplotlib-inline==0.2.2
narwhals==2.22.1
nest-asyncio2==1.7.2
numpy==2.5.0
packaging==26.2
pandas==3.0.3
parso==0.8.7
pillow==12.2.0
platformdirs==4.10.0
prompt_toolkit==3.0.52
psutil==7.2.2
pure_eval==0.2.3
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pyzmq==27.1.0
scikit-learn==1.9.0
scipy==1.18.0
seaborn==0.13.2
six==1.17.0
stack-data==0.6.3
threadpoolctl==3.6.0
tornado==6.5.7
traitlets==5.15.1
typing_extensions==4.15.0
tzdata==2026.2
wcwidth==0.8.1
xgboost==3.3.0

## Business Dashboard

### Deskripsi Dashboard
Dashboard dibuat menggunakan Metabase untuk memonitor performa mahasiswa dan faktor-faktor risiko dropout. Dashboard ini menyajikan informasi secara visual dan interaktif untuk memudahkan stakeholder dalam pengambilan keputusan.

### Komponen Dashboard
1. KPI Cards (Overview Metrics):
- Total Mahasiswa
- Total Lulusan (Graduate)
- Total Dropout
- Total Aktif (Enrolled)
- Tingkat Kelulusan (Graduate Rate)
- Tingkat Dropout (Dropout Rate)
2. Student Status Distribution:
- Menampilkan persentase Graduate, Enrolled, Dropout
3. Academic Performance by Status
- Bar Chart perbandingan nilai semester 1 & 2
- Perbandingan SKS diambil vs lulus
- Analisis performa per status
4. Dropout Risk Factors Analysis
- Dropout berdasarkan Gender (Bar Chart)
- Dropout berdasarkan Beasiswa (Bar Chart)
- Dropout berdasarkan Hutang (Bar Chart)
- Dropout berdasarkan Usia (Bar Chart)
5. Course-wise Dropout Analysis
- Bar Chart program studi dengan dropout tertinggi
- Perbandingan dropout vs graduate per program studi
6. Age Distribution by Status
- Bar Chart distribusi usia per status
7. Risk Factors Summary
- Ranking faktor risiko dari tertinggi ke terendah

### Akses Dashboard
URL: http://localhost:3000/dashboard/1-student-dropout-prediction-system

Login Credentials:
- Email: fairuzarfan@student.ub.ac.id
- Password: Ari527668__

### Cara Menjalankan Dashboard
1. Jalankan Metabase dengan Docker:
```bash
docker pull metabase/metabase:v0.46.4
docker run -d -p 3000:3000 --name metabase metabase/metabase:v0.46.4
```
2. Buka browser dan akses: http://localhost:3000
3. Login dengan credentials di atas
4. Pilih dashboard "Student Dropout Prediction System"

## Prototype Machine Learning

### Deskripsi Prototype
Prototype sistem machine learning dibuat menggunakan Streamlit untuk memprediksi status mahasiswa (Dropout, Enrolled, Graduate) berdasarkan data input yang dimasukkan. Sistem ini juga memberikan rekomendasi intervensi yang sesuai dengan profil risiko mahasiswa.

### Fitur Prototype
1. Form Input Data Mahasiswa
- Data Demografi (Gender, Usia, Status Pernikahan, dll)
- Data Akademik (Program Studi, Nilai, SKS, dll)
- Data Sosial-Ekonomi (Pendidikan Orang Tua, Beasiswa, Hutang, dll)
- Data Akademik Semester 1 & 2
- Data Ekonomi Makro
2. Prediksi Real-time
- Status: Dropout / Enrolled / Graduate
- Probabilitas untuk setiap status
- Visualisasi bar chart probabilitas
3. Rekomendasi Intervensi
- Rekomendasi spesifik berdasarkan status dan data input
- Saran intervensi untuk mahasiswa berisiko dropout
- Program pengembangan untuk mahasiswa berprestasi
4. Feature Importance
- Menampilkan top 10 fitur paling berpengaruh
- Visualisasi untuk pemahaman model

### Cara Menjalankan Prototype
1. Clone/Download Proyek
```bash
git clone [url-repository]
cd submission
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Pastikan File Model Tersedia
4. Jalankan Streamlit App
```bash
streamlit run app.py
```
5. Deploy Aplikasi di Streamlit Cloud 

### Akses Prototype (Streamlit Cloud)
link: https://prediksi-dropout-mahasiswa-hbpn5mdlaju9zyaj9l4b4h.streamlit.app/

## Conclusion

### Kesimpulan Utama
1. Data Preparation
- Total data awal: 4,424 mahasiswa
- Data yang digunakan untuk training: 3,630 mahasiswa (status Dropout dan Graduate)
- Data Enrolled (794 mahasiswa) tidak digunakan karena belum memiliki label akhir
2. Model Terbaik: Logistic Regression dengan Accuracy 89.12%
3. Performa Model:
- Accuracy: 89.12%
- Precision: 89.16%
- Recall: 89.12%
- F1-Score: 89.01%
- Cross Validation Mean: 89.15%
4. Performa per Kelas:
- Dropout: Precision 90%, Recall 81%, F1-Score 85%
- Graduate: Precision 89%, Recall 94%, F1-Score 91%
5. Confusion Matrix:
- True Dropout: 231 mahasiswa
- False Negative (Dropout → Graduate): 53 mahasiswa
- False Positive (Graduate → Dropout): 26 mahasiswa
- True Graduate: 416 mahasiswa
- Total error: 79 prediksi salah (10.88%)
6. Faktor Dominan Dropout (Top 5 Feature Importance):
- Nilai Semester 1 (Curricular_units_1st_sem_grade) sebagai prediktor terkuat
- SKS Lulus Semester 1 (Curricular_units_1st_sem_approved)
- SKS Diambil Semester 1 (Curricular_units_1st_sem_enrolled)
- Status Beasiswa (Scholarship_holder)
- Status Hutang (Debtor)
7. Key Insights:
- Mahasiswa dengan nilai semester 1 < 10 berisiko dropout 3x lebih tinggi
- Mahasiswa dengan SKS > 6 di semester pertama berisiko dropout lebih tinggi
- Penerima beasiswa memiliki tingkat dropout 40% lebih rendah
- Mahasiswa dengan hutang berisiko dropout 2x lebih tinggi
- Mahasiswa usia > 25 tahun memiliki risiko dropout lebih tinggi
8. Kesimpulan Bisnis:
- Intervensi dini pada semester pertama sangat krusial
- Dukungan finansial (beasiswa) secara signifikan menurunkan risiko dropout
- Sistem early warning system berbasis performa akademik sangat diperlukan
- Program studi dengan tingkat dropout tinggi perlu evaluasi kurikulum

## Rekomendasi Action Items

### Action Item 1: Implementasi Early Warning System (EWS)
Tujuan: Mendeteksi dini mahasiswa berisiko dropout dan memberikan intervensi tepat waktu.

Langkah Implementasi:
1. Integrasi Model Prediksi
- Integrasikan model machine learning ke dalam sistem akademik institusi
- Buat API untuk prediksi otomatis berdasarkan data mahasiswa
- Sistem akan memproses data mahasiswa setiap akhir semester
2. Dashboard Monitoring
- Buat dashboard khusus untuk dosen wali dan konselor
- Tampilkan daftar mahasiswa berisiko dengan level risiko (rendah, sedang, tinggi)
- Notifikasi otomatis ketika mahasiswa masuk kategori risiko tinggi
3. Alur Notifikasi
- Risk Level Tinggi → Notifikasi ke dosen wali dan konselor dalam 24 jam
- Risk Level Sedang → Notifikasi ke dosen wali dalam 3 hari
- Risk Level Rendah → Update dashboard bulanan
4. Tracking Intervensi
- Sistem untuk mencatat intervensi yang telah dilakukan
- Monitoring perkembangan mahasiswa setelah intervensi
- Evaluasi efektivitas intervensi

Target: Menurunkan angka dropout sebesar 25% dalam 1 tahun

KPI:
- Penurunan dropout rate dari 15% menjadi 11.25%
- 90% mahasiswa berisiko tinggi mendapatkan intervensi dalam 1 minggu
- Peningkatan retention rate dari 65% menjadi 80%

### Action Item 2: 
Tujuan: Memberikan bantuan yang tepat sasaran berdasarkan profil risiko mahasiswa.

Langkah Implementasi:
1. Semester 1 (Early Intervention)
- Bimbingan Belajar: Program tambahan untuk mata kuliah dengan tingkat kegagalan tinggi
- Konseling Akademik: Sesi rutin setiap 2 minggu untuk mahasiswa berisiko
- Workshop Manajemen Waktu: Pelatihan untuk mahasiswa baru
- Pembatasan SKS: Maksimal 6 SKS untuk mahasiswa dengan nilai masuk rendah
2. Semester 2-3 (Sustained Support)
- Program Mentoring: Mahasiswa senior berprestasi menjadi mentor
- Kelompok Studi Kecil: Study group untuk mata kuliah sulit
- Konsultasi Karir: Meningkatkan motivasi dan visi masa depan
- Program Pengembangan Soft Skills: Komunikasi, kepemimpinan, problem solving
3. Semester 4+ (Retention Program)
- Program Magang: Pengalaman kerja untuk meningkatkan motivasi
- Networking dengan Alumni: Inspirasi dan koneksi karir
- Program Entrepreneur: Untuk mahasiswa dengan minat bisnis
- Persiapan Tugas Akhir: Bimbingan intensif untuk penyelesaian studi

Target: Meningkatkan retention rate dari 65% menjadi 80%

KPI:
- 80% mahasiswa program intervensi menyelesaikan semester
- 70% mahasiswa program intervensi lulus tepat waktu
- Penurunan dropout rate 20% dalam 2 tahun
