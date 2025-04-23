# file: likert_app.py
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
