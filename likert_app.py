import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Setting tampilan
st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# Judul Aplikasi
st.title("📊 Analisis Skala Likert Profesional")

# Upload File
uploaded_file = st.file_uploader("📥 Upload file CSV hasil survei", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    likert_df = df.iloc[:, 2:]  # Ambil kolom dari kolom ke-3 ke kanan (index 2+)
    kolom_likert = likert_df.columns

    # ✅ Fungsi interpretasi
    def interpretasi_skor(skor):
        if skor > 4:
            return "Sangat Positif"
        elif skor > 3:
            return "Netral Positif"
        elif skor == 3:
            return "Netral"
        elif skor > 2:
            return "Netral Negatif"
        else:
            return "Sangat Negatif"

    # Sidebar
    st.sidebar.header("⚙️ Pilih Analisis")
    analisis_terpilih = st.sidebar.selectbox(
        "Jenis Analisis",
        ["Visualisasi", "Rata-Rata & Interpretasi", "Uji Reliabilitas", "Korelasi", "Export Excel"]
    )

    # --- Visualisasi ---
    if analisis_terpilih == "Visualisasi":
        st.subheader("📋 Data Response")
        st.dataframe(df.head())
        st.info(f"📌 Jumlah responden: **{df.shape[0]}**")

        st.subheader("📈 Visualisasi & Ringkasan Tiap Pertanyaan")
        for i, kolom in enumerate(kolom_likert, start=1):
            st.markdown(f"### ❓ Q{i}: {kolom}")
            plt.figure(figsize=(10, 3))
            order = sorted(likert_df[kolom].dropna().unique())
            sns.countplot(data=likert_df, y=kolom, order=order, palette="Blues_d")
            plt.xlabel("Jumlah Responden")
            plt.ylabel("Skala")
            st.pyplot(plt)
            plt.clf()

            frekuensi = likert_df[kolom].value_counts().sort_values(ascending=False)
            jawaban_terbanyak = frekuensi.index[0]
            jumlah_terbanyak = frekuensi.iloc[0]
            st.success(f"📝 Jawaban paling banyak: **{jawaban_terbanyak}** sebanyak **{jumlah_terbanyak}** responden.")

    # --- Rata-Rata & Interpretasi ---
    elif analisis_terpilih == "Rata-Rata & Interpretasi":
        st.subheader("📊 Rata-Rata Skor & Interpretasi")
        avg_scores = likert_df.mean()
        interpretasi = avg_scores.apply(interpretasi_skor)

        avg_table = pd.DataFrame({
            "Pertanyaan": avg_scores.index,
            "Rata-Rata Skor": avg_scores.values,
            "Interpretasi": interpretasi
        }).sort_values(by="Rata-Rata Skor", ascending=False)

        st.dataframe(avg_table)

        st.subheader("📈 Grafik Rata-Rata")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=avg_table["Rata-Rata Skor"], y=avg_table["Pertanyaan"], palette='viridis')
        for i, v in enumerate(avg_table["Rata-Rata Skor"]):
            ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')
        st.pyplot(fig)

        # 5. 📉 IDENTIFIKASI PERTANYAAN TERENDAH, PERTENGAHAN, & TERTINGGI
        lowest_scores = avg_scores.nsmallest(3)  # 3 pertanyaan terendah
        highest_scores = avg_scores.nlargest(3)  # 3 pertanyaan tertinggi
        median_scores = avg_scores.iloc[len(avg_scores)//3:2*len(avg_scores)//3]  # 3 pertanyaan pertengahan

        # Menampilkan hasil
        st.subheader("📉 3 Pertanyaan dengan Skor Terendah")
        st.dataframe(lowest_scores)

        st.subheader("📈 3 Pertanyaan dengan Skor Tertinggi")
        st.dataframe(highest_scores)

        st.subheader("📊 3 Pertanyaan dengan Skor Pertengahan")
        st.dataframe(median_scores)

        # Menambahkan interpretasi untuk setiap kategori
        lowest_scores_interpretation = pd.DataFrame({
            "Pertanyaan": lowest_scores.index,
            "Skor Rata-rata": lowest_scores.values,
            "Interpretasi": [interpretasi_skor(s) for s in lowest_scores.values]
        })

        highest_scores_interpretation = pd.DataFrame({
            "Pertanyaan": highest_scores.index,
            "Skor Rata-rata": highest_scores.values,
            "Interpretasi": [interpretasi_skor(s) for s in highest_scores.values]
        })

        median_scores_interpretation = pd.DataFrame({
            "Pertanyaan": median_scores.index,
            "Skor Rata-rata": median_scores.values,
            "Interpretasi": [interpretasi_skor(s) for s in median_scores.values]
        })

        # Menampilkan interpretasi
        st.subheader("📝 Interpretasi Pertanyaan dengan Skor Terendah")
        st.dataframe(lowest_scores_interpretation)

        st.subheader("📝 Interpretasi Pertanyaan dengan Skor Tertinggi")
        st.dataframe(highest_scores_interpretation)

        st.subheader("📝 Interpretasi Pertanyaan dengan Skor Pertengahan")
        st.dataframe(median_scores_interpretation)

        # Grafik perbandingan skor
        fig, ax = plt.subplots(figsize=(12, 6))

        # Gabungkan ketiganya
        combined_scores = pd.concat([lowest_scores, highest_scores, median_scores])
        labels = list(lowest_scores.index) + list(highest_scores.index) + list(median_scores.index)
        values = list(lowest_scores.values) + list(highest_scores.values) + list(median_scores.values)

        # Plotting
        sns.barplot(x=combined_scores.index, y=values, palette='coolwarm', ax=ax)

        # Menambah label
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel('Pertanyaan')
        ax.set_ylabel('Skor Rata-rata')
        ax.set_title('Perbandingan Skor Terendah, Pertengahan & Tertinggi')

        # Menampilkan grafik
        st.pyplot(fig)

    # --- Uji Reliabilitas ---
  # --- Uji Reliabilitas ---
elif analisis_terpilih == "Uji Reliabilitas":
    st.subheader("📐 Uji Reliabilitas - Cronbach's Alpha")

    def cronbach_alpha(data):
        item_vars = data.var(axis=0, ddof=1)
        total_var = data.sum(axis=1).var(ddof=1)
        n_items = data.shape[1]
        return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)

    alpha = cronbach_alpha(likert_df)

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

    st.markdown(f"**Cronbach's Alpha: {alpha:.3f}** — {interpret_alpha(alpha)}")

    # Penjelasan tentang Cronbach's Alpha
    if alpha >= 0.9:
        st.info("Nilai Cronbach's Alpha yang tinggi (≥ 0.9) menunjukkan bahwa instrumen survei Anda memiliki reliabilitas yang sangat baik. Artinya, pertanyaan-pertanyaan dalam survei ini sangat konsisten dan saling mendukung dalam mengukur konstruk yang sama.")
    elif alpha >= 0.8:
        st.info("Nilai Cronbach's Alpha yang baik (≥ 0.8) menunjukkan bahwa instrumen survei Anda memiliki reliabilitas yang cukup baik, meskipun masih ada sedikit variasi dalam konsistensi antar item.")
    elif alpha >= 0.7:
        st.info("Nilai Cronbach's Alpha yang cukup (≥ 0.7) menunjukkan bahwa instrumen survei Anda memiliki reliabilitas yang dapat diterima, namun masih ada ruang untuk meningkatkan konsistensi antar item.")
    elif alpha >= 0.6:
        st.info("Nilai Cronbach's Alpha yang kurang (≥ 0.6) menunjukkan bahwa instrumen survei Anda mungkin memiliki reliabilitas yang rendah, dan beberapa pertanyaan mungkin tidak konsisten dalam mengukur konstruk yang sama.")
    elif alpha >= 0.5:
        st.info("Nilai Cronbach's Alpha yang rendah (≥ 0.5) menunjukkan bahwa instrumen survei Anda mungkin tidak cukup reliabel dan beberapa pertanyaan perlu diperbaiki.")
    else:
        st.info("Nilai Cronbach's Alpha yang sangat rendah menunjukkan bahwa instrumen survei Anda tidak reliabel dan perlu dilakukan revisi serius pada pertanyaan-pertanyaan yang ada.")

# --- Korelasi ---
elif analisis_terpilih == "Korelasi":
    st.subheader("🔥 Korelasi antar Pertanyaan")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu', ax=ax2)
    st.pyplot(fig2)

    # Penjelasan tentang Korelasi
    st.markdown("Korelasi antar pertanyaan menunjukkan sejauh mana pertanyaan-pertanyaan dalam survei saling terkait. Nilai korelasi berkisar antara -1 hingga 1:")
    st.info("""
    - **Nilai korelasi mendekati 1**: Ada hubungan positif yang kuat antara dua pertanyaan, artinya jika responden memberikan jawaban tinggi pada satu pertanyaan, kemungkinan besar mereka akan memberikan jawaban tinggi pada pertanyaan lainnya.
    - **Nilai korelasi mendekati -1**: Ada hubungan negatif yang kuat antara dua pertanyaan, artinya jika responden memberikan jawaban tinggi pada satu pertanyaan, mereka cenderung memberikan jawaban rendah pada pertanyaan lainnya.
    - **Nilai korelasi mendekati 0**: Tidak ada hubungan yang signifikan antara dua pertanyaan, artinya perubahan pada jawaban salah satu pertanyaan tidak mempengaruhi jawaban pada pertanyaan lainnya.
    """)

    st.markdown("Anda dapat menggunakan informasi ini untuk mengevaluasi konsistensi antar pertanyaan atau untuk melihat apakah ada pertanyaan yang berlebihan (saling terkait kuat) atau kurang relevan (tidak terkait).")

    # --- Export Excel ---
    elif analisis_terpilih == "Export Excel":
        st.subheader("📤 Export Interpretasi ke Excel")

        avg_scores = likert_df.mean()
        interpretasi = avg_scores.apply(interpretasi_skor)

        df_export = pd.DataFrame({
            "Pernyataan": avg_scores.index,
            "Rata-rata Skor": avg_scores.values,
            "Interpretasi": interpretasi.values
        })

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Interpretasi')

        st.download_button(
            label="📥 Download Interpretasi (Excel)",
            data=output.getvalue(),
            file_name="interpretasi_likert.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
