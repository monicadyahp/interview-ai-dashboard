import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 1. Tambahkan import os di paling atas

st.set_page_config(page_title="Dashboard Interview-AI", page_icon="🎤", layout="wide")

st.title("🎤 Dashboard Analisis Emosi: Interview-AI")
st.markdown("""
Aplikasi ini adalah prototipe untuk memantau emosi kandidat saat wawancara kerja. 
Data di bawah ini merupakan hasil ekstraksi dan pembersihan (*Data Wrangling*) dari dataset **FER-2013**.
""")
st.divider() 

@st.cache_data
def load_data():
    # 2. Masukkan logika path absolut ke dalam fungsi load_data
    # Ini memastikan Python mencari file di folder yang sama dengan script ini berada
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(CURRENT_DIR, 'fer2013_cleaned_data.csv')
    
    # 3. Baca file menggunakan path absolut yang sudah dibentuk
    df = pd.read_csv(CSV_PATH)
    return df

try:
    df_cleaned = load_data()
except FileNotFoundError:
    # 4. Modifikasi pesan error agar menampilkan path lengkap (berguna untuk debugging)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(CURRENT_DIR, 'fer2013_cleaned_data.csv')
    st.error(f"File tidak ditemukan! Python mencoba mencari di: {CSV_PATH}")
    st.stop()

# --- Sisa kode ke bawah tetap sama ---
col1, col2 = st.columns([1, 2]) 

with col1:
    st.subheader("📁 Preview Data Bersih")
    st.write("Tabel ini berisi path gambar dan label emosi yang sudah di-undersampling.")
    st.dataframe(df_cleaned.head(10), use_container_width=True)
    st.info(f"Total Data Siap Pakai: {len(df_cleaned)} gambar")

with col2:
    st.subheader("📊 Distribusi Emosi (Sesudah Cleaning)")
    st.write("Grafik distribusi kelas emosi yang sudah seimbang.")
    
    df_counts = df_cleaned['Label'].value_counts().rename_axis('Kategori Emosi').reset_index(name='Jumlah Gambar')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Kategori Emosi', y='Jumlah Gambar', data=df_counts, hue='Kategori Emosi', palette='Greens_r', legend=False, ax=ax)
    st.pyplot(fig)

st.divider()
st.subheader("📸 Galeri Sampel Wajah Kandidat")
st.write("Berikut adalah beberapa contoh wajah dari dataset yang digunakan untuk melatih AI.")

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'sample_images')

if os.path.exists(SAMPLE_DIR):
    images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if images:
        cols = st.columns(min(4, len(images)))
        for idx, img_name in enumerate(images[:4]):
            with cols[idx]:
                img_path = os.path.join(SAMPLE_DIR, img_name)
                label_name = img_name.split('_')[0].upper()
                st.image(img_path, caption=f"Emosi: {label_name}", use_container_width=True)
    else:
        st.warning("Folder 'sample_images' ada, tapi tidak ada gambar di dalamnya.")
else:
    st.error("Folder 'sample_images' tidak ditemukan. Pastikan folder ini sudah di-upload ke GitHub.")

st.success("Tugas Data Science untuk persiapan data telah selesai!")