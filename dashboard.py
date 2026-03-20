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
st.write("Contoh gambar wajah dari dataset. (Catatan: Gambar mungkin tidak muncul di Cloud karena batasan ukuran upload GitHub)")

# Ambil sampel data
sample_data = df_cleaned.sample(min(4, len(df_cleaned)))
cols = st.columns(4)

for index, (i, row) in enumerate(sample_data.iterrows()):
    with cols[index]:
        img_path = row['File_Path']
        
        # CEK: Apakah file gambarnya benar-benar ada di folder?
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Emosi: {row['Label'].upper()}", use_container_width=True)
        else:
            # Jika tidak ada (karena di-gitignore), tampilkan placeholder atau pesan
            st.warning(f"File {row['Label']} tidak ditemukan di server.")
            st.caption("Fungsionalitas lokal aman.")

st.success("Tugas Data Science untuk persiapan data telah selesai!")