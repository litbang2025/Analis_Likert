import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

# Styling visual
sns.set(style="whitegrid")
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ===============================
# 1. 📥 BACA DATA
# ===============================
st.title("📊 ANALISIS SKALA LIKERT (PROFESIONAL)")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📊 Data Analisis Awal:")
    st.dataframe(df.head())

    # ===============================
    # 📝 KETERANGAN DATA
    # ===============================
    total_responden = df.shape[0]
    total_kolom = df.shape[1]
    kolom_likert = df.columns[2:]
    jumlah_pertanyaan = len(kolom_likert)

    st.write("ℹ️ Keterangan Awal:")
    st.write(f"- Jumlah Responden: {total_responden}")
    st.write(f"- Total Kolom: {total_kolom}")
    st.write(f"- Kolom Pertanyaan (Likert): {jumlah_pertanyaan} kolom")
    st.write(f"- Nama Kolom Pertanyaan: {list(kolom_likert)}")

    # Ambil data Likert
    likert_df = df.iloc[:, 2:]
    # ===============================
    # 2. 📐 UJI RELIABILITAS (CRONBACH'S ALPHA)
    # ===============================
    def cronbach_alpha(data):
        item_vars = data.var(axis=0, ddof=1)
        total_var = data.sum(axis=1).var(ddof=1)
        n_items = data.shape[1]
        return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)

    alpha = cronbach_alpha(likert_df)
    st.write(f"✅ Cronbach's Alpha: {alpha:.3f}")

    # Penjelasan hasil
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
    # 3. 📊 RATA-RATA PER PERTANYAAN
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

    # Buat DataFrame ringkasan
    avg_table = pd.DataFrame({
        "Pertanyaan": avg_scores.index,
        "Rata-Rata Skor": avg_scores.values,
        "Interpretasi": [interpretasi_skor(s) for s in avg_scores.values]
    })

    st.write("\n📋 Tabel Rata-Rata & Interpretasi:")
    st.dataframe(avg_table)

    # 🎨 Visualisasi Barplot dengan Nilai
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=avg_scores.values, y=avg_scores.index, palette='viridis')

    # Tambahkan nilai rata-rata di ujung bar
    for i, v in enumerate(avg_scores.values):
        ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.title("📊 Rata-Rata Skor per Pertanyaan")
    plt.xlabel("Rata-Rata Skor (1–5)")
    plt.xlim(1, 5)
    plt.tight_layout()
    st.pyplot(plt)

    # ===============================
    # 4. 📉 DISTRIBUSI SKOR TOTAL PER RESPONDEN
    # ===============================
    df['Total Skor'] = likert_df.sum(axis=1)

    # Visualisasi distribusi skor total
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Total Skor'], bins=10, kde=True, color='skyblue')
    plt.title("📉 Distribusi Skor Total per Responden")
    plt.xlabel("Total Skor")
    plt.ylabel("Jumlah Responden")
    plt.tight_layout()
    st.pyplot(plt)

    # ===============================
    # 5. 📉 IDENTIFIKASI PERTANYAAN TERENDAH
    # ===============================
    lowest_scores = avg_scores.tail(3)  # Menampilkan 3 pertanyaan terendah
    st.write("\n📋 3 Pertanyaan dengan Skor Terendah:")
    st.dataframe(lowest_scores)

    # ===============================
    # 6. 📈 HEATMAP KORELASI
    # ===============================
    plt.figure(figsize=(8, 6))
    sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu')
    plt.title("📌 Korelasi Antar Pertanyaan")
    plt.tight_layout()
    st.pyplot(plt)

    # ===============================
    # 7. 📦 TABEL RANGKUMAN
    # ===============================
    summary_table = pd.DataFrame({
        "Pertanyaan": avg_scores.index,
        "Skor Rata-rata": avg_scores.values
    })

    st.write("\n📋 Tabel Rangkuman:")
    st.dataframe(summary_table)

    # ===============================
    # 8. 🧪 UJI HIPOTESIS (optional)
    # ===============================
    st.write("\n🧪 Uji T terhadap nilai netral (3):")
    hasil_uji = []

    for col in likert_df.columns:
        t_stat, p_val = ttest_1samp(likert_df[col], 3)
        status = "Signifikan" if p_val < 0.05 else "Tidak Signifikan"
        hasil_uji.append((col, t_stat, p_val, status))
        st.write(f"- {col}: t = {t_stat:.2f}, p = {p_val:.3f} → {status}")

    # ===============================
    # 9. 📤 (Opsional) SIMPAN HASIL
    # ===============================
    # summary_table.to_csv("hasil_analisis_survei.csv", index=False)
    st.write(f"Mayoritas pertanyaan memiliki skor rata-rata di atas {avg_scores.mean():.2f}, menunjukkan persepsi yang {interpretasi_skor(avg_scores.mean())}.") ```python
    plt.figure(figsize=(6,6))
    df['Kategori'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title("📊 Proporsi Kategori Responden")
    plt.ylabel('')
    plt.tight_layout()
    st.pyplot(plt)

    # Fungsi untuk memplot distribusi per pertanyaan
    def plot_distribusi_per_pertanyaan(df_likert):
        warna = ['#FF9999', '#FFCC99', '#99CCFF', '#66FF99', '#CCCCFF']

        for kolom in df_likert.columns:
            counts = df_likert[kolom].value_counts(normalize=True).sort_index()
            persen = (counts * 100).round(1)

            labels = [f'{int(skor)} ({p}%)' for skor, p in zip(counts.index, persen)]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(aspect="equal"))
            wedges, texts = ax.pie(
                counts,
                labels=labels,
                colors=warna[:len(counts)],
                startangle=90,
                wedgeprops=dict(width=0.4, edgecolor='white')
            )

            ax.set_title(kolom, fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

            skor_tertinggi = counts.idxmax()
            st.write(f"✅ Kesimpulan '{kolom}': Skor {skor_tertinggi} paling banyak dipilih ({persen[skor_tertinggi]}%)\n")

    # Memanggil fungsi untuk memplot distribusi per pertanyaan
    plot_distribusi_per_pertanyaan(likert_df)
