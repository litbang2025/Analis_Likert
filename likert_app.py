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
        ["Visualisasi", "Rata-Rata & Interpretasi", "Uji Reliabilitas", "Korelasi", "Uji Normalitas","Uji Lanjutan", "Export Excel"]
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

    # Validasi jika data likert kosong
    if likert_df.empty:
        st.warning("⚠️ Data likert tidak ditemukan atau kosong.")
    else:
        # Hitung skor total setiap responden
        skor_total = likert_df.mean(axis=1)

        # Statistik deskriptif singkat
        rata2 = skor_total.mean()
        median = skor_total.median()
        st.write(f"**Rata-rata Skor:** {rata2:.2f}")
        st.write(f"**Median Skor:** {median:.2f}")

        # Pilih metode uji normalitas
        if n <= 50:
            st.write("🔎 Metode: **Shapiro-Wilk Test** (n ≤ 50)")
            stat, p = shapiro(skor_total)
        else:
            st.write("🔎 Metode: **Kolmogorov-Smirnov Test** (n > 50)")
            stat, p = kstest(skor_total, 'norm', args=(skor_total.mean(), skor_total.std()))

        # Hasil uji normalitas
        st.write(f"**Statistik Uji:** {stat:.4f}")
        st.write(f"**p-value:** {p:.4f}")

        # Interpretasi
        if p > 0.05:
            st.success("✅ Data terdistribusi normal (p > 0.05)")
            st.info("✅ Cocok untuk uji parametrik seperti ANOVA atau regresi.")
        else:
            st.error("❌ Data tidak terdistribusi normal (p ≤ 0.05)")
            st.warning("👉 Disarankan melanjutkan dengan uji non-parametrik.")

        # Tambahan: QQ Plot
        st.subheader("📉 QQ Plot")
        fig_qq = plt.figure(figsize=(6, 6))
        probplot(skor_total, dist="norm", plot=plt)
        plt.title("QQ Plot - Skor Total")
        st.pyplot(fig_qq)

        # Tambahan: Histogram
        st.subheader("📊 Histogram Skor Total")
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(skor_total, bins=10, color="skyblue", edgecolor="black")
        ax_hist.set_title("Distribusi Skor Total")
        ax_hist.set_xlabel("Skor")
        ax_hist.set_ylabel("Frekuensi")
        st.pyplot(fig_hist)

        # Statistik deskriptif lanjutan
        st.subheader("📑 Statistik Deskriptif Lanjutan")
        deskriptif_df = pd.DataFrame({
            "Rata-rata": [rata2],
            "Median": [median],
            "Standar Deviasi": [skor_total.std()],
            "Skewness": [skor_total.skew()],
            "Kurtosis": [skor_total.kurt()],
            "IQR": [skor_total.quantile(0.75) - skor_total.quantile(0.25)]
        })
        st.dataframe(deskriptif_df)

        st.caption("""
        ℹ️ *Skewness* > 0 menunjukkan kemencengan ke kanan, < 0 ke kiri.
        *Kurtosis* tinggi menunjukkan ekor yang lebih berat dari distribusi normal.
        """)

        # Boxplot
        st.subheader("📦 Boxplot Skor Total")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=skor_total, color="lightblue", ax=ax_box)
        ax_box.set_title("Boxplot Skor Total")
        st.pyplot(fig_box)

        # Rekomendasi jika data tidak normal
        if p <= 0.05:
            st.subheader("📚 Rekomendasi Uji Non-parametrik")
            st.markdown("""
            Karena data tidak berdistribusi normal, berikut rekomendasi uji lanjutan:
            - **Uji Mann-Whitney U**: untuk dua kelompok. `scipy.stats.mannwhitneyu`
            - **Uji Kruskal-Wallis**: untuk tiga kelompok atau lebih. `scipy.stats.kruskal`
            - **Analisis Deskriptif**: Gunakan median, IQR, dan visualisasi seperti boxplot.
            """)

# --- Uji Lanjutan ---
elif analisis_terpilih == "Uji Lanjutan":
    st.subheader("🔬 Uji Lanjutan")
    st.markdown("Fitur ini akan menampilkan analisis tambahan seperti uji homogenitas, uji beda, atau regresi sederhana.")

    # Cek apakah skor_total sudah tersedia
    if 'skor_total' not in locals() and 'skor_total' not in globals():
        st.error("❌ Variabel 'skor_total' belum tersedia. Pastikan Anda sudah melakukan perhitungan skor sebelumnya.")
    else:
        kolom_kategori = st.selectbox("🔢 Pilih kolom kategori untuk Uji Kruskal-Wallis:", df.columns)

        if kolom_kategori:
            df_kruskal = df[[kolom_kategori]].copy()
            df_kruskal["Skor_Total"] = skor_total

            # Hapus data yang mengandung NaN
            df_kruskal.dropna(subset=[kolom_kategori, "Skor_Total"], inplace=True)

            # Validasi tipe data
            if not pd.api.types.is_categorical_dtype(df_kruskal[kolom_kategori]) and not pd.api.types.is_object_dtype(df_kruskal[kolom_kategori]):
                st.warning("⚠️ Kolom kategori sebaiknya bertipe kategorik atau string.")

            # Tampilkan jumlah data per kategori
            st.write("📊 Jumlah data per kategori:")
            st.dataframe(
                df_kruskal[kolom_kategori].value_counts()
                .reset_index()
                .rename(columns={"index": kolom_kategori, kolom_kategori: "Jumlah"})
            )

            # Visualisasi distribusi
            st.write("📈 Distribusi Skor per Kelompok:")
            fig = px.box(df_kruskal, x=kolom_kategori, y="Skor_Total", points="all", title="Boxplot Skor per Kelompok")
            st.plotly_chart(fig)

            # Siapkan data untuk uji Kruskal-Wallis
            grouped_data = [group["Skor_Total"].values for name, group in df_kruskal.groupby(kolom_kategori)]

            if len(grouped_data) >= 3:
                stat_kw, p_kw = kruskal(*grouped_data)
                st.write(f"**Statistik Kruskal-Wallis:** {stat_kw:.4f}")
                st.write(f"**p-value:** {p_kw:.4f}")

                if p_kw <= 0.05:
                    st.success("✅ Perbedaan antar kelompok signifikan (p ≤ 0.05)")

                    st.markdown("""
                    Karena hasil uji Kruskal-Wallis menunjukkan adanya perbedaan signifikan antar kelompok, 
                    maka dilakukan **analisis lanjutan (post-hoc)** menggunakan **Dunn's test** 
                    untuk mengetahui secara spesifik kelompok mana yang berbeda signifikan.
                    """)

                    try:
                        import scikit_posthocs as sp
                        dunn_result = sp.posthoc_dunn(
                            df_kruskal, val_col="Skor_Total", group_col=kolom_kategori, p_adjust='bonferroni'
                        )
                        st.subheader("🔬 Hasil Dunn’s Test (Post-hoc)")
                        st.write("p-value perbandingan antar kelompok (koreksi Bonferroni):")
                        st.dataframe(dunn_result.round(4))
                        st.markdown("""
                        **Interpretasi**:
                        - Nilai p ≤ 0.05 menunjukkan perbedaan signifikan antara dua kelompok.
                        - Perhatikan baris dan kolom yang bersesuaian untuk identifikasi pasangan kelompok yang berbeda.
                        """)
                    except ImportError:
                        st.error("❌ Paket `scikit-posthocs` belum terpasang. Jalankan `pip install scikit-posthocs`.")
                else:
                    st.info("ℹ️ Tidak ada perbedaan signifikan antar kelompok (p > 0.05)")
            else:
                st.warning("⚠️ Kolom kategori harus memiliki minimal tiga kelompok untuk Uji Kruskal-Wallis.")

           # Visualisasi Boxplot
            st.subheader(f"📦 Boxplot Skor Total per '{kolom_kategori}'")
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=kolom_kategori, y="Skor_Total", data=df_kruskal, palette="Set2", ax=ax6)
            ax6.set_title(f"Boxplot Skor Total berdasarkan {kolom_kategori}", fontsize=14)
            ax6.set_xlabel(kolom_kategori, fontsize=12)
            ax6.set_ylabel("Skor Total", fontsize=12)
            ax6.tick_params(axis='x', rotation=45)
            st.pyplot(fig6)
            
            # Visualisasi distribusi skor total
            st.subheader("📊 Distribusi Skor Total")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.histplot(skor_total, kde=True, bins=20, color="salmon", ax=ax3)
            ax3.set_title("Distribusi Skor Total Responden", fontsize=14)
            ax3.set_xlabel("Skor Total", fontsize=12)
            ax3.set_ylabel("Jumlah Responden", fontsize=12)
            st.pyplot(fig3)
            
            # QQ-Plot untuk semua kasus
            st.subheader("📈 Visualisasi QQ-Plot")
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            probplot(skor_total, dist="norm", plot=ax4)
            ax4.set_title("QQ-Plot Skor Total", fontsize=14)
            st.pyplot(fig4)

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
  
