import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

# Styling untuk visualisasi
sns.set(style="whitegrid")
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ===============================
# 📥 Baca Data
# ===============================
df = pd.read_csv('data_analis_linkert.csv')

print("📊 Data Analisis Awal:")
display(df.head())

# ===============================
# 📝 Keterangan Data
# ===============================
total_responden = df.shape[0]
total_kolom = df.shape[1]
kolom_likert = df.columns[2:]
jumlah_pertanyaan = len(kolom_likert)

print("ℹ️ Keterangan Awal:")
print(f"- Jumlah Responden: {total_responden}")
print(f"- Total Kolom: {total_kolom}")
print(f"- Kolom Pertanyaan (Likert): {jumlah_pertanyaan} kolom")
print(f"- Nama Kolom Pertanyaan: {list(kolom_likert)}")

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
print(f"✅ Cronbach's Alpha: {alpha:.3f}")

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
    
print(f"📌 Interpretasi: {interpretasi}")

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

print("\n📋 Tabel Rata-Rata & Interpretasi:")
display(avg_table)

# 🎨 Visualisasi Barplot dengan Nilai
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=avg_scores.values, y=avg_scores.index, palette='viridis')

for i, v in enumerate(avg_scores.values):
    ax.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

plt.title("📊 Rata-Rata Skor per Pertanyaan")
plt.xlabel("Rata-Rata Skor (1–5)")
plt.xlim(1, 5)
plt.tight_layout()
plt.show()

# ===============================
# 📉 Distribusi Skor Total per Responden
# ===============================
df['Total Skor'] = likert_df.sum(axis=1)

# Visualisasi distribusi skor total
plt.figure(figsize=(8, 4))
sns.histplot(df['Total Skor'], bins=10, kde=True, color='skyblue')
plt.title("📉 Distribusi Skor Total per Responden")
plt.xlabel("Total Skor")
plt.ylabel("Jumlah Responden")
plt.tight_layout()
plt.show()

# ===============================
# 📉 Korelasi Antar Pertanyaan (Heatmap)
# ===============================
plt.figure(figsize=(8, 6))
sns.heatmap(likert_df.corr(), annot=True, cmap='YlGnBu')
plt.title("📌 Korelasi Antar Pertanyaan")
plt.tight_layout()
plt.show()

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
print("\n📊 Distribusi Kategori:")
print(df['Kategori'].value_counts())

# ===============================
# 7. 📦 Kesimpulan dan Rekomendasi
# ===============================
# Berdasarkan semua analisis yang telah dilakukan, buat kesimpulan dan rekomendasi untuk perbaikan.
