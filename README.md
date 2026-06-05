# 🤖 Interview-AI Assistant: Data Science Dashboard (InterSight)

[![Streamlit App](https://static.streamlit.io/badge_sticker/github.svg)](https://interview-ai-dashboard-su7byt4utngvkc3yqvtifr.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io)

Dashboard interaktif ini dikembangkan untuk mendokumentasikan, memvisualisasikan, dan menguji seluruh rangkaian proses **Data Science** dalam proyek Capstone **InterSight** (Team ID: `CC26-PSU188`). Fokus utama kami adalah menyelesaikan masalah ketidakseimbangan data (*imbalanced dataset*) emosi wajah dan memvalidasi efektivitas intervensi fitur berbasis AI secara statistik.

---

## 📌 1. Permasalahan & Solusi Bisnis

* **Akar Masalah:** Banyak pencari kerja (*fresh graduates*) mengalami kendala psikologis berupa kecemasan dan kurangnya kesadaran akan ekspresi non-verbal (*micro-expressions*) saat wawancara kerja, yang mengakibatkan tingginya angka kegagalan rekrutmen.
* **Solusi InterSight:** Sebuah platform *AI-Powered Smart Mirror* yang mendeteksi ekspresi wajah secara *real-time* untuk melatih ketenangan pengguna, dilengkapi dengan simulasi tanya-jawab mandiri dan laporan performa yang estetik.

---

## 📊 2. Alur Kerja Data Science (End-to-End Tasks)

Proyek analitik data ini mencakup implementasi komprehensif dari siklus hidup ilmu data:

### A. Data Wrangling & Kamus Data
* **Gathering Data:** Menggunakan dataset publik **FER-2013** dari Kaggle yang berisi lebih dari 35.000 data gambar ekspresi wajah berukuran $48 \times 48$ piksel.
* **Assessing Data:** Menemukan ketidakseimbangan kelas (*imbalanced data*) yang ekstrem, di mana kelas `happy` mendominasi (>7.000 gambar) sedangkan kelas `disgust` menjadi minoritas kritis (<500 gambar).
* **Cleaning & Hybrid Balancing:** Menerapkan strategi hibrida dengan melakukan *Undersampling* pada kelas mayoritas (`happy` dipangkas menjadi 4.000 sampel) dan *Oversampling* sintetik pada kelas minoritas (`disgust` direplikasi menjadi 4.000 sampel). Total dataset final menjadi **29.058 baris data** yang seimbang (*Flat-Balanced*).
* **Feature Engineering:** Mengonversi string piksel mentah menjadi matriks gambar selevel *grayscale* dan menormalisasi nilai intensitas piksel dengan membaginya dengan `255.0` (skala $[0, 1]$) untuk mempercepat konvergensi model CNN tim AI.

### B. Eksperimen Statistika (A/B Testing)
Kami melakukan eksperimen terkontrol terhadap 2 kelompok pengguna (Grup Kontrol tanpa AI vs Grup Eksperimen dengan dukungan intervensi motivasi AI). Menggunakan uji **Independent Two-Sample T-Test** via `scipy.stats`, diperoleh nilai **P-Value sebesar $2.14 \times 10^{-7}$**. Karena $P\text{-Value} < 0.05$, kesimpulan statistik membuktikan secara mutlak bahwa fitur asisten pintar InterSight signifikan meningkatkan kepercayaan diri pengguna.

---

## 🖥️ 3. Struktur Dokumen & Fitur Dashboard

Dashboard ini dibangun menggunakan komponen tab Streamlit yang terbagi menjadi 5 modul utama:
1.  **📊 Business Insight:** Menampilkan eksplorasi data tabular acak per kategori emosi beserta visualisasi grafik interaktif distribusi data akhir menggunakan Plotly (Bar & Pie Chart).
2.  **📑 Data Dictionary:** Menyediakan kamus data formal, transparansi alur pengolahan data, serta dokumentasi teknik rekayasa fitur (*feature engineering*).
3.  **🧪 A/B Testing:** Menyajikan pembuktian ilmiah efektivitas aplikasi melalui grafik *Box Plot* sebaran skor dan metrik *P-Value* real-time.
4.  **📸 Data Gallery:** Memuat galeri sampel visual citra wajah asli dari dataset sebagai bentuk audit kelayakan visual data.
5.  **🎨 UI/UX Mockup:** Menampilkan rancangan fitur masa depan (*Pre-Interview Check*) seperti pendeteksi kecerahan cahaya (*Light Checker code*) dan garis pandu posisi kepala kamera.

---
