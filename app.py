import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from streamlit_shap import st_shap
import matplotlib.pyplot as plt # Impor pyplot

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Penipuan Transaksi",
    page_icon="🛡️",
    layout="wide"
)

# --- Kustomisasi Tema (Warm Background) ---
# Menambahkan CSS kustom untuk latar belakang yang hangat
st.markdown(
    """
    <style>
    /* Target the main app container's content block */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #FDFCF5 !important; /* Paksa latar belakang hangat */
    }
    
    /* Target semua elemen teks di dalam .main agar kontras */
    /* Ini lebih spesifik daripada .stApp */
    .main, 
    .main h1, 
    .main h2, 
    .main h3, 
    .main p, 
    .main li, 
    .main .stMarkdown, 
    .main .stWrite,
    .main .stError, /* Atur teks error agar gelap */
    .main .stSuccess { /* Atur teks sukses agar gelap */
        color: #0E1117 !important; /* Paksa teks menjadi gelap */
    }

    /* Pastikan label expander juga memiliki teks gelap */
    .main [data-testid="stExpander"] summary {
        color: #0E1117 !important;
    }

    /* Khusus untuk st.error dan st.success, kita mungkin ingin latarnya
       tetap ada, tapi teksnya gelap. Selektor di atas sudah menangani teks.
       Jika Anda ingin mengubah latar st.error/st.success juga, 
       tambahkan aturan di sini. */

    /* Biarkan sidebar dan header menggunakan tema default (gelap) */
    /* [data-testid="stSidebar"] { ... } */
    /* [data-testid="stHeader"] { ... } */
    
    </style>
    """,
    unsafe_allow_html=True
)

# --- Judul dan Deskripsi ---
st.title("🛡️ Sistem Deteksi Penipuan Transaksi Keuangan")
st.write("""
Aplikasi ini menggunakan model Machine Learning (XGRequest) dan Explainable AI (SHAP) 
untuk memprediksi apakah sebuah transaksi merupakan penipuan dan menjelaskan alasannya.
Masukkan detail transaksi di sidebar kiri untuk memulai analisis.
""")

# --- Fungsi Caching untuk Memuat Model dan Explainer ---
# Menggunakan cache_resource agar model dan kolom hanya dimuat sekali.

@st.cache_resource
def load_model_artifacts():
    """
    Memuat model XGBoost dan daftar kolom dari file .joblib
    """
    try:
        model = joblib.load('xgboost_model.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, model_columns
    except FileNotFoundError:
        st.error("Error: File 'xgboost_model.joblib' atau 'model_columns.joblib' tidak ditemukan.")
        st.error("Pastikan Anda telah menjalankan sel 'Simpan Model' di notebook Anda dan file-filenya berada di folder yang sama dengan app.py.")
        return None, None

@st.cache_resource
def load_explainer(_model):
    """
    Membuat dan menyimpan SHAP explainer
    """
    # FIX: Menggabungkan shap.TreeExplainer ke satu baris
    return shap.TreeExplainer(_model)

# Memuat artefak
model, model_columns = load_model_artifacts()
if model:
    explainer = load_explainer(model)

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Masukkan Detail Transaksi:")

# Input berdasarkan studi kasus di Sel 7
type_transaksi = st.sidebar.selectbox("Jenis Transaksi", ['TRANSFER', 'CASH_OUT'])
amount = st.sidebar.number_input("Jumlah (Amount)", min_value=0.0, value=5000000.0, step=1000.0)
step = st.sidebar.number_input("Waktu (Step)", min_value=1, value=10, step=1)
oldbalanceOrg = st.sidebar.number_input("Saldo Awal Pengirim", min_value=0.0, value=5000000.0, step=1000.0)
newbalanceOrig = st.sidebar.number_input("Saldo Akhir Pengirim", min_value=0.0, value=0.0, step=1000.0)
oldbalanceDest = st.sidebar.number_input("Saldo Awal Penerima", min_value=0.0, value=10000.0, step=1000.0)
newbalanceDest = st.sidebar.number_input("Saldo Akhir Penerima", min_value=0.0, value=5010000.0, step=1000.0)

# Tombol untuk memicu analisis
if st.sidebar.button("Analisis Transaksi"):
    if model and model_columns:
        # 1. Membuat DataFrame dari input
        data_input = {
            'step': step,
            'type': type_transaksi,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }
        input_df_raw = pd.DataFrame([data_input])
        
        # 2. Melakukan Rekayasa Fitur (sesuai Sel 3 & 7)
        input_df = input_df_raw.copy()
        input_df['type_TRANSFER'] = 1 if type_transaksi == 'TRANSFER' else 0
        input_df = input_df.drop('type', axis=1)
        
        # Hitung fitur rekayasa
        input_df['errorBalanceOrg'] = input_df['newbalanceOrig'] + input_df['amount'] - input_df['oldbalanceOrg']
        input_df['errorBalanceDest'] = input_df['oldbalanceDest'] + input_df['amount'] - input_df['newbalanceDest']
        
        # Pastikan urutan kolom sama persis dengan saat pelatihan
        try:
            input_df = input_df[model_columns]
        except KeyError as e:
            st.error(f"Error: Kolom tidak cocok. {e}")
            st.stop()

        # 3. Membuat Prediksi
        prediksi = model.predict(input_df)[0]
        probabilitas = model.predict_proba(input_df)[0]
        
        st.header("Hasil Prediksi Model")
        if prediksi == 1:
            st.error(f"**Prediksi: PENIPUAN** (Probabilitas: {probabilitas[1]*100:.2f}%)")
        else:
            st.success(f"**Prediksi: Bukan Penipuan** (Probabilitas: {probabilitas[0]*100:.2f}%)")

        # 4. Menampilkan Penjelasan SHAP
        st.header("Penjelasan Model (XAI)")
        st.write("Plot di bawah ini menunjukkan bagaimana setiap fitur 'mendorong' atau 'menarik' prediksi dari nilai dasar ke hasil akhir.")
        
        shap_values = explainer.shap_values(input_df)
        expected_value = explainer.expected_value
        
        # FIX: Bypass st_shap dan gunakan st.pyplot secara langsung
        # 1. Buat plot SHAP dan dapatkan objek Figure (fig)
        # Kita tambahkan plt.clf() untuk membersihkan figur sebelumnya
        plt.clf()
        fig = shap.force_plot(
            expected_value,
            shap_values[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False # Penting: agar mengembalikan fig, bukan menampilkannya
        )
        
        # 2. Tampilkan fig menggunakan st.pyplot dengan use_container_width
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Tidak dapat membuat plot SHAP.")
        
        # Menampilkan detail fitur
        with st.expander("Lihat Data Input yang Telah Diproses Model"):
            st.dataframe(input_df)

    else:
        st.error("Model belum dimuat. Periksa pesan error di atas.")

