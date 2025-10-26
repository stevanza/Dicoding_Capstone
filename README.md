Sistem Deteksi Penipuan Transaksi dengan XAI

Proyek ini adalah aplikasi web interaktif yang dibangun menggunakan Streamlit untuk mendeteksi penipuan transaksi keuangan. Keunggulan utamanya adalah penggunaan Explainable AI (XAI) melalui SHAP untuk memberikan penjelasan yang transparan dan dapat dipahami manusia di balik setiap prediksi model.

Proyek ini dibuat sebagai bagian dari Dicoding Capstone Project (Tim DB8-PG019).

Tampilan Aplikasi

Berikut adalah tampilan dari dasbor aplikasi Streamlit. Pengguna dapat memasukkan parameter transaksi secara manual di sidebar, dan model akan memberikan prediksi instan beserta penjelasan visualnya.

<img width="1512" height="824" alt="Screenshot 2025-10-26 at 20 34 48" src="https://github.com/user-attachments/assets/86e03cb2-27f8-463f-98d9-555e61d40127" />

Fitur Utama

Deteksi Penipuan: Mengklasifikasikan transaksi sebagai "PENIPUAN" atau "Bukan Penipuan" menggunakan model XGBoost yang telah dilatih.

Penjelasan Model (XAI): Menampilkan SHAP Force Plot untuk setiap prediksi, yang menunjukkan faktor-faktor apa saja (fitur) yang paling berkontribusi terhadap keputusan model.

Antarmuka Interaktif: Sidebar yang mudah digunakan memungkinkan simulasi transaksi baru untuk dianalisis secara real-time.

Teknologi yang Digunakan

Proyek ini dibangun menggunakan tumpukan teknologi berikut:

Python 3.9+

Streamlit: Untuk membangun dan menjalankan aplikasi web interaktif.

XGBoost: Sebagai algoritma machine learning utama untuk klasifikasi.

SHAP: Untuk menghitung dan memvisualisasikan penjelasan model (XAI).

Scikit-learn: Untuk pra-pemrosesan data dan evaluasi model.

Pandas & Numpy: Untuk manipulasi dan pemrosesan data.

Joblib: Untuk menyimpan dan memuat model yang telah dilatih.

Matplotlib: Untuk merender plot SHAP di dalam Streamlit.

replicating Cara Meniru Proyek Ini (Replikasi Langkah)

Berikut adalah panduan langkah demi langkah untuk menjalankan proyek ini di komputer lokal Anda.

1. Dapatkan Kode

Kloning repository ini ke komputer Anda:

git clone [https://github.com/stevanza/Dicoding_Capstone.git](https://github.com/stevanza/Dicoding_Capstone.git)
cd Dicoding_Capstone


2. Dapatkan Dataset (Untuk Pelatihan Ulang)

Model yang disediakan (.joblib) sudah dilatih. Namun, jika Anda ingin melatih ulang model dari awal, Anda memerlukan dataset aslinya.

Model ini dilatih menggunakan dataset PaySim - Synthetic Financial Transactions dari Kaggle.

Unduh dataset: Kaggle: PaySim Dataset

Letakkan file archive.zip yang telah diunduh ke dalam folder proyek Anda.

3. Struktur Folder

.

├── app.py                  
├── fraud_detection_notebook.py 
├── xgboost_model.joblib    
├── model_columns.joblib    
└── requirements.txt        


4. Latih Model (Opsional - Lewati jika menggunakan model bawaan)

Jika Anda ingin melatih model Anda sendiri:

Buka file fraud_detection_notebook.py menggunakan Google Colab atau Jupyter Notebook.

Unggah file archive.zip (dataset) saat diminta oleh sel pertama.

Jalankan semua sel di dalam notebook secara berurutan.

Sel terakhir akan menyimpan file xgboost_model.joblib dan model_columns.joblib yang baru. (Jika di Colab, pastikan untuk mengunduhnya).

5. Instalasi Dependensi

Sangat disarankan untuk membuat virtual environment agar tidak mengganggu instalasi Python utama Anda.

# Buat virtual environment (opsional)
python3 -m venv venv

# Aktifkan (di macOS/Linux)
source venv/bin/activate
# (di Windows)
venv\Scripts\activate

# Instal semua library yang dibutuhkan
pip install -r requirements.txt


6. Jalankan Aplikasi Streamlit

Setelah semua dependensi terinstal, jalankan perintah berikut di terminal Anda:

streamlit run app.py


Aplikasi Anda sekarang akan terbuka secara otomatis di browser Anda (biasanya di http://localhost:8501).
