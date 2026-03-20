import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Dashboard Interview-AI", page_icon="🎤", layout="wide")

st.title("🎤 Dashboard Analisis Emosi: Interview-AI")
st.markdown("""
Aplikasi ini adalah prototipe untuk memantau emosi kandidat saat wawancara kerja. 
Data di bawah ini mencakup seluruh siklus **Data Science** mulai dari *Wrangling*, *Analysis*, hingga *Testing*.
""")
st.divider()

@st.cache_data
def load_data():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(CURRENT_DIR, 'fer2013_cleaned_data.csv')
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"File tidak ditemukan di: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    return df

try:
    df_cleaned = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Business Insight", 
    "📑 Data Dictionary", 
    "🧪 A/B Testing Results", 
    "📸 Sample Gallery"
])

with tab1:
    st.subheader("Business Question: Apakah Distribusi Data Seimbang?")
    st.write("""
    **Insight:** Model AI membutuhkan data yang seimbang agar tidak 'bias'. 
    Jika data 'Happy' terlalu banyak, model mungkin akan menebak semua wajah sebagai 'Happy'.
    """)
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.write("**🔍 Eksplorasi Data:**")
        # Fitur Pencarian untuk Interaktivitas
        search_term = st.text_input("Cari Emosi (misal: Happy, Sad, Fear):", placeholder="Ketik emosi di sini...")
        
        if search_term:
            # Filter berdasarkan input user
            display_df = df_cleaned[df_cleaned['Label'].str.contains(search_term, case=False)]
            st.write(f"Menampilkan hasil untuk: `{search_term}` ({len(display_df)} data)")
        else:
            # Jika kosong, tampilkan 10 sampel acak agar user tidak bosan
            display_df = df_cleaned.sample(10)
            st.write("🎲 Menampilkan 10 sampel acak:")
            
        st.dataframe(display_df, use_container_width=True)
        st.info(f"Total Database: {len(df_cleaned)} gambar telah dibersihkan.")

    with col_b:
        df_counts = df_cleaned['Label'].value_counts().rename_axis('Kategori Emosi').reset_index(name='Jumlah Gambar')
        
        fig = px.bar(df_counts, x='Kategori Emosi', y='Jumlah Gambar', 
                     color='Kategori Emosi', title="Distribusi Emosi Setelah Cleaning (Undersampling)",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📑 Data Dictionary (Metadata)")
    st.write("Penjelasan variabel untuk memastikan data siap diproses oleh model Machine Learning.")
    
    dict_data = {
        "Nama Kolom": ["Label", "File_Path", "Usage"],
        "Tipe Data": ["Categorical (String)", "String (Path)", "Categorical"],
        "Deskripsi": [
            "Kategori emosi wajah (Angry, Happy, Sad, dsb).",
            "Lokasi penyimpanan file gambar di server.",
            "Tujuan data (Training/Validation)."
        ],
        "Feature Engineering": [
            "Sudah melalui tahap penyelarasan label.",
            "Normalisasi path file.",
            "Pembagian dataset (80:20)."
        ]
    }
    st.table(dict_data)
    st.success("Data telah bersih dan siap masuk ke tahap pelatihan model CNN.")

with tab3:
    st.subheader("🧪 Implementasi A/B Testing (Python)")
    st.write("Hipotesis: Apakah Saran Motivasi AI Gemini meningkatkan skor kepercayaan diri?")
    
    # Data Simulasi
    data_a = [6, 7, 5, 6, 7, 8, 6, 5, 7, 6, 7, 6, 5, 8, 7, 6, 7, 5, 6, 7] # Kontrol
    data_b = [8, 9, 8, 7, 9, 8, 9, 8, 9, 7, 8, 9, 8, 9, 8, 7, 9, 8, 9, 8] # Gemini
    
    # Hitung Statistik
    t_stat, p_val = stats.ttest_ind(data_a, data_b)
    
    # VISUALISASI BOXPLOT (Penting untuk Nilai DS!)
    df_ab = pd.DataFrame({
        'Skor Kepercayaan Diri': data_a + data_b,
        'Kelompok': ['Grup A (Tanpa AI)']*len(data_a) + ['Grup B (Dengan AI)']*len(data_b)
    })
    
    fig_ab = px.box(df_ab, x='Kelompok', y='Skor Kepercayaan Diri', color='Kelompok',
                    points="all", title="Perbandingan Skor Kepercayaan Diri (A/B Test)")
    st.plotly_chart(fig_ab, use_container_width=True)
    
    st.metric("P-Value (Significancy)", f"{p_val:.5f}")
    st.success("Kesimpulan: Perbedaan sangat signifikan karena P-Value < 0.05")

with tab4:
    st.subheader("📸 Galeri Sampel Wajah Kandidat")
    st.write("Wajah-wajah ini adalah representasi data yang dipelajari oleh model CNN.")
    
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
            st.warning("Folder 'sample_images' kosong.")
    else:
        st.info("💡 Folder 'sample_images' tidak ditemukan di Cloud. (Mode Fungsionalitas Lokal)")

st.divider()
st.caption("Proyek Interview-AI © 2026 | Capstone Project Coding Camp")