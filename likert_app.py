import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import io

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

    # Keterangan Data
    num_respondents = len(df)
    num_columns = len(df.columns)
    num_questions = likert_df.shape[1]
    
    st.subheader("📝 Keterangan Data")
    st.write(f"Jumlah Responden: {num_respondents}")
    st.write(f"Jumlah Total Kolom (termasuk ID dan Nama): {num_columns}")
    st.write(f"Jumlah Pertanyaan Berbasis Skala Likert: {num_questions}")

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
            return "Perlu Perbaikan"
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
    # Kategorisasi Skor Rata-Rata
    avg_table['Interpretasi'] = avg_table['Rata-Rata Skor'].apply(lambda x: 'Sangat Baik' if x >= 4.2 else 
                                                                 ('Baik' if 3.6 <= x < 4.2 else 
                                                                  ('Cukup' if 3.0 <= x < 3.6 else 'Perlu Perhatian')))
    st.dataframe(avg_table)

    # Bar Chart untuk Rata-Rata
    st.subheader("📈 Visualisasi Skor per Pertanyaan")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_scores.values, y=avg_scores.index, palette='viridis', ax=ax)
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')
    st.pyplot(fig)

    # Distribusi Skor Total Responden
    st.subheader("📉 Distribusi Skor Total Responden")
    total_scores = likert_df.sum(axis=1)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(total_scores, kde=True, bins=15, ax=ax2)
    ax2.set_title("Distribusi Skor Total Responden")
    ax2.set_xlabel("Skor Total")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)

    # Identifikasi Pertanyaan dengan Nilai Terendah
    st.subheader("🔍 Identifikasi Pertanyaan dengan Nilai Terendah")
    lowest_questions = avg_table.nsmallest(3, 'Rata-Rata Skor')
    st.dataframe(lowest_questions)

    # Heatmap korelasi
    st.subheader("🌡️ Heatmap Korelasi Antar Pertanyaan")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu', ax=ax3)
    st.pyplot(fig3)

    # Kategorisasi Total Skor Responden
    st.subheader("📦 Kategorisasi Total Skor Responden")
    score_categories = pd.cut(total_scores, bins=[-np.inf, 40, 59, 79, np.inf], labels=["Negatif", "Netral", "Positif", "Sangat Positif"])
    score_category_table = pd.DataFrame({
        "Responden": df.iloc[:, 0],
        "Skor Total": total_scores,
        "Kategori": score_categories
    })
    st.dataframe(score_category_table)

    # Uji Hipotesis - Uji-t untuk setiap pertanyaan
    st.subheader("🧪 Uji Hipotesis (Opsional)")
    t_test_results = []
    for question in likert_df.columns:
        t_stat, p_value = stats.ttest_1samp(likert_df[question], 3)
        t_test_results.append({
            "Pertanyaan": question,
            "t-statistic": t_stat,
            "p-value": p_value,
            "Signifikan (p < 0.05)": "Ya" if p_value < 0.05 else "Tidak"
        })
    t_test_df = pd.DataFrame(t_test_results)
    st.dataframe(t_test_df)

    # Tabel Rangkuman
    st.subheader("📋 Tabel Rangkuman")
    st.dataframe(avg_table[['Pertanyaan', 'Rata-Rata Skor', 'Interpretasi']])
