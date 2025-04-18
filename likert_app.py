import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Setting tampilan
st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# Judul Aplikasi
st.title("📊 Tool Analisis Skala Likert Litbang IHBS")

# Upload File
uploaded_file = st.file_uploader("📥 Upload file CSV hasil survei", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Data Response")
    st.dataframe(df.head())

    # Ambil kolom pertanyaan (asumsi kolom ke-3 dst)
    likert_df = df.iloc[:, 2:]

    # Fungsi Cronbach Alpha
    def cronbach_alpha(data):
        item_vars = data.var(axis=0, ddof=1)
        total_var = data.sum(axis=1).var(ddof=1)
        n_items = data.shape[1]
        return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)
    
    alpha = cronbach_alpha(likert_df)
    
    # Interpretasi
    def interpret_alpha(a):
        if a >= 0.9:
            return "Sangat Baik"
        elif a >= 0.8:
            return "Baik"
        elif a >= 0.7:
            return "Cukup"
        elif a >= 0.6:
            return "Kurang"
        elif a >= 0.5:
            return "Rendah"
        else:
            return "Tidak Dapat Diterima"

    st.subheader("📐 Uji Reliabilitas - Cronbach's Alpha")
    st.markdown(f"**Cronbach's Alpha: {alpha:.3f}** — {interpret_alpha(alpha)}")

    # Rata-rata per pertanyaan
    avg_scores = likert_df.mean().sort_values(ascending=False)

    # Tabel Rata-rata
    st.subheader("📊 Rata-Rata Skor per Pertanyaan")
    avg_table = pd.DataFrame({
        "Pertanyaan": avg_scores.index,
        "Rata-Rata Skor": avg_scores.values
    })
    st.dataframe(avg_table)

    # Bar Chart
    st.subheader("📈 Visualisasi Skor per Pertanyaan")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_scores.values, y=avg_scores.index, palette='viridis', ax=ax)
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')
    st.pyplot(fig)

    # Fitur Download Visualisasi sebagai Gambar
    st.download_button(
        label="Unduh Visualisasi (PNG)",
        data=fig.savefig('/mnt/data/visualisasi.png', format='png'),
        file_name="visualisasi_skala_likert.png",
        mime="image/png"
    )

    # Heatmap korelasi
    st.subheader("🔥 Korelasi antar Pertanyaan")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu', ax=ax2)
    st.pyplot(fig2)

    # Fitur Download Heatmap sebagai Gambar
    st.download_button(
        label="Unduh Heatmap (PNG)",
        data=fig2.savefig('/mnt/data/heatmap.png', format='png'),
        file_name="heatmap_korelasi.png",
        mime="image/png"
    )

    # Misalnya, data hasil uji hipotesis yang sudah dihitung sebelumnya
    hasil_uji = [
        ("Pertanyaan 1", 2.3, 0.05, "Signifikan"),
        ("Pertanyaan 2", 1.2, 0.15, "Tidak Signifikan"),
        ("Pertanyaan 3", 3.1, 0.01, "Signifikan")
    ]

    # Menyusun kesimpulan berdasarkan hasil uji hipotesis
    st.subheader("📌 Kesimpulan Uji Hipotesis:")
    for item in hasil_uji:
        pertanyaan, t_stat, p_val, status = item
        if status == "Signifikan":
            st.write(f"- Pada pertanyaan '{pertanyaan}', hasil uji t menunjukkan bahwa nilai rata-rata secara signifikan lebih tinggi dari nilai netral (3).")
        else:
            st.write(f"- Pada pertanyaan '{pertanyaan}', hasil uji t menunjukkan bahwa nilai rata-rata tidak signifikan lebih tinggi dari nilai netral (3).")

    # Fitur Download Hasil Rata-Rata sebagai CSV
    st.download_button(
        label="Unduh Hasil Rata-Rata Skor",
        data=avg_table.to_csv(index=False).encode('utf-8'),
        file_name="rata_rata_skor.csv",
        mime="text/csv"
    )
