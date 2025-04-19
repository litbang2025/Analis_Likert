import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import streamlit as st

# Styling untuk visualisasi
sns.set(style="whitegrid")
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Judul aplikasi Streamlit
st.title("📊 Analisis Skala Likert")

# ===============================
# 📥 Unggah Data CSV
# ===============================
# 📥 Unggah Data CSV (Laptop & Mobile)
# ===============================
st.sidebar.header("📥 Unggah Data / Masukkan Link")

# Unggah file (laptop)
uploaded_file = st.sidebar.file_uploader("Pilih file CSV dari perangkat", type=["csv"])

# Link alternatif (mobile)
csv_url = st.sidebar.text_input("Atau masukkan link file CSV (Google Drive, Dropbox, dll)")

df = None  # Inisialisasi

# Deteksi dari unggahan
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File berhasil diunggah dari perangkat.")
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")

# Deteksi dari URL
elif csv_url:
    try:
        df = pd.read_csv(csv_url)
        st.success("✅ File berhasil dimuat dari link.")
    except Exception as e:
        st.error(f"❌ Gagal memuat dari link: {e}")

# Notifikasi kalau belum ada data
if df is None:
    st.info("""
    📌 Kamu bisa unggah file CSV dari **laptop** atau tempel **link file CSV** dari Google Drive, Dropbox, atau lainnya.

    ⚠️ *Di HP, tombol unggah kadang hanya menampilkan gambar/video — gunakan link file sebagai alternatif.*
    """)

    
    st.write("📊 Data Analisis Awal:")
    st.write(df.head())

    # ===============================
    # 📝 Keterangan Data
    # ===============================
    total_responden = df.shape[0]  # Jumlah responden (baris)
    total_kolom = df.shape[1]      # Jumlah kolom dalam data
    kolom_likert = df.columns[2:]  # Kolom yang berisi pertanyaan Likert
    jumlah_pertanyaan = len(kolom_likert)  # Jumlah pertanyaan

    # Menampilkan informasi dasar tentang data
    st.subheader("ℹ️ Keterangan Awal:")
    st.write(f"- Jumlah Responden: {total_responden}")
    st.write(f"- Jumlah Total Kolom: {total_kolom}")
    st.write(f"- Jumlah Pertanyaan Skala Likert: {jumlah_pertanyaan} kolom")
    st.write(f"- Nama Kolom Pertanyaan: {list(kolom_likert)}")

    # ===============================
    # 📐 Uji Reliabilitas (Cronbach's Alpha)
    # ===============================
    def cronbach_alpha(data):
        item_vars = data.var(axis=0, ddof=1)
        total_var = data.sum(axis=1).var(ddof=1)
        n_items = data.shape[1]
        return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)

    likert_df = df.iloc[:, 2:]
    alpha = cronbach_alpha(likert_df)
    st.subheader(f"✅ Cronbach's Alpha: {alpha:.3f}")

    if alpha >= 0.9:
        interpretasi = "Sangat Baik (Excellent)"
    elif alpha >= 0.8:
        interpretasi = "Baik (Good)"
    elif alpha >= 0.7:
        interpretasi = "Cukup (Acceptable)"
    elif alpha >= 0.6:
        interpretasi = "Kurang (Questionable)"
    elif alpha >= 0.5:
        interpretasi = "Rendah (Poor)"
    else:
        interpretasi = "Tidak Dapat Diterima (Unacceptable)"
    
    st.write(f"📌 Interpretasi: {interpretasi}")

    # ===============================
    # 📊 Rata-Rata Skor per Pertanyaan
    # ===============================
    avg_scores = likert_df.mean().sort_values(ascending=False)

    # Kategori interpretasi skor
    def interpretasi_skor(skor):
        if skor >= 4.2:
            return "Sangat Baik"
        elif skor >= 3.6:
            return "Baik"
        elif skor >= 3.0:
            return "Cukup"
        else:
            return "Perlu Perhatian"

    avg_table = pd.DataFrame({
        "Pertanyaan": avg_scores.index,
        "Rata-Rata Skor": avg_scores.values,
        "Interpretasi": [interpretasi_skor(s) for s in avg_scores.values]
    })

    st.subheader("📋 Tabel Rata-Rata & Interpretasi:")
    st.write(avg_table)

    # ===============================
    # 🧐 Pertanyaan dengan Nilai Terendah
    # ===============================
    pertanyaan_terendah = avg_scores.idxmin()
    nilai_terendah = avg_scores.min()
    st.subheader(f"📉 Pertanyaan dengan Nilai Terendah:")
    st.write(f"{pertanyaan_terendah} dengan nilai {nilai_terendah:.2f}")

    # 🎨 Visualisasi Barplot dengan Nilai
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=avg_scores.values, y=avg_scores.index, palette='viridis', ax=ax)

    # Menampilkan nilai rata-rata di atas batang
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.title("📊 Rata-Rata Skor per Pertanyaan")
    plt.xlabel("Rata-Rata Skor (1–5)")
    plt.xlim(1, 5)
    st.pyplot(fig)

    # ===============================
    # 📉 Distribusi Skor Total per Responden
    # ===============================
    df['Total Skor'] = likert_df.sum(axis=1)

    # Visualisasi distribusi skor total
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Total Skor'], bins=10, kde=True, color='skyblue', ax=ax)
    ax.set_title("📉 Distribusi Skor Total per Responden")
    ax.set_xlabel("Total Skor")
    ax.set_ylabel("Jumlah Responden")
    st.pyplot(fig)

    # ===============================
    # 📉 Korelasi Antar Pertanyaan (Heatmap)
    # ===============================
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu', ax=ax)
    ax.set_title("📌 Korelasi Antar Pertanyaan")
    st.pyplot(fig)

    # ===============================
    # 📦 Kategori Skor Total
    # ===============================
    def kategorikan(skor):
        if skor >= 80:
            return "Sangat Positif"
        elif skor >= 60:
            return "Positif"
        elif skor >= 40:
            return "Netral"
        else:
            return "Negatif"

    df['Kategori'] = df['Total Skor'].apply(kategorikan)
    st.subheader("📊 Distribusi Kategori Skor Total:")
    st.write(df['Kategori'].value_counts())

    # ===============================
    # 📋 Ringkasan dan Rekomendasi
    # ===============================
    st.subheader("📑 Ringkasan Hasil Analisis:")
    st.write(f"Total Responden: {total_responden}")
    st.write(f"Total Kolom: {total_kolom}")
    st.write(f"Jumlah Pertanyaan Skala Likert: {jumlah_pertanyaan}")
    st.write(f"Pertanyaan dengan nilai terendah: {pertanyaan_terendah} (Nilai: {nilai_terendah:.2f})")

    if alpha < 0.7:
        st.warning("⚠️ Rekomendasi: Konsistensi instrumen survei perlu ditingkatkan. Pertimbangkan untuk merevisi pertanyaan atau meningkatkan pengukuran.")
    else:
        st.success("✅ Rekomendasi: Instrumen survei memiliki reliabilitas yang baik, dapat digunakan dengan percaya diri.")
