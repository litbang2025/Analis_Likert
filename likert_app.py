import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import shapiro, kstest, norm, probplot

# Setting tampilan
st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# Judul Aplikasi
st.title("📊 Tool Analisis Skala Likert")

# Instruksi
st.markdown("""
**Petunjuk:**
- Pastikan file CSV memiliki format sebagai berikut:
  - Kolom A: Email
  - Kolom B: Nama
  - Kolom C dan seterusnya: Pertanyaan survei (Q1, Q2, Q3, dst.)
- Kolom pertanyaan **dimulai dari kolom C**, artinya aplikasi ini akan membaca kolom ke-3 dan seterusnya sebagai data skala Likert.
""")

# Upload File
uploaded_file = st.file_uploader("📥 Upload file CSV hasil survei", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    likert_df = df.iloc[:, 2:]  # Ambil kolom dari kolom ke-3 ke kanan (index 2+)
    kolom_likert = likert_df.columns

    # ✅ Fungsi interpretasi skor
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
        ["Visualisasi", "Rata-Rata & Interpretasi", "Uji Reliabilitas", "Korelasi", "Uji Normalitas", "Export Excel"]
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

        # Interpretasi
        def buat_interpretasi(scores):
            return pd.DataFrame({
                "Pertanyaan": scores.index,
                "Skor Rata-rata": scores.values,
                "Interpretasi": [interpretasi_skor(s) for s in scores.values]
            })

        st.subheader("📝 Interpretasi Pertanyaan dengan Skor Terendah")
        st.dataframe(buat_interpretasi(lowest_scores))

        st.subheader("📝 Interpretasi Pertanyaan dengan Skor Tertinggi")
        st.dataframe(buat_interpretasi(highest_scores))

        st.subheader("📝 Interpretasi Pertanyaan dengan Skor Pertengahan")
        st.dataframe(buat_interpretasi(median_scores))

        # Grafik perbandingan skor
        fig, ax = plt.subplots(figsize=(12, 6))
        combined_scores = pd.concat([lowest_scores, highest_scores, median_scores])
        labels = list(combined_scores.index)
        values = list(combined_scores.values)

        sns.barplot(x=labels, y=values, palette='coolwarm', ax=ax)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel('Pertanyaan')
        ax.set_ylabel('Skor Rata-rata')
        ax.set_title('Perbandingan Skor Terendah, Pertengahan & Tertinggi')

        st.pyplot(fig)

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

    # --- Korelasi ---
    elif analisis_terpilih == "Korelasi":
        st.subheader("🔥 Korelasi antar Pertanyaan")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu', ax=ax2)
        st.pyplot(fig2)

    # --- Uji Normalitas ---
    elif analisis_terpilih == "Uji Normalitas":
        st.subheader("🧪 Uji Normalitas Data")

        n = df.shape[0]
        st.info(f"📌 Jumlah responden: **{n}**")

        # Pilih metode otomatis
        skor_total = likert_df.mean(axis=1)

        if n <= 50:
            st.write("🔎 Metode: **Shapiro-Wilk Test** (n ≤ 50)")
            stat, p = shapiro(skor_total)
        else:
            st.write("🔎 Metode: **Kolmogorov-Smirnov Test** (n > 50)")
            stat, p = kstest(skor_total, 'norm', args=(skor_total.mean(), skor_total.std()))

        st.write(f"**Statistik Uji:** {stat:.4f}")
        st.write(f"**p-value:** {p:.4f}")

        if p > 0.05:
            st.success("✅ Data terdistribusi normal (p > 0.05)")
        else:
            st.error("❌ Data tidak terdistribusi normal (p ≤ 0.05)")

        # Tambahkan QQ-Plot
        st.subheader("📈 Visualisasi QQ-Plot")
        fig, ax = plt.subplots(figsize=(6, 6))
        probplot(skor_total, dist="norm", plot=ax)
        st.pyplot(fig)

    # --- Export Excel ---
    elif analisis_terpilih == "Export Excel":
        st.subheader("📤 Export Data ke Excel")

        @st.cache_data
        def convert_df(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            output.seek(0)
            return output

        excel_file = convert_df(df)
        st.download_button(
            label="Download Data Excel",
            data=excel_file,
            file_name="data_survei.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
