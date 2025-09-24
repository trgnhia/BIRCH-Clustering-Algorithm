# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 11

# =========================
# 0. CẤU HÌNH CHUNG
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_input_dataset.csv"
OUTPUT_CF_CSV = "Demographics_CF_output.csv"
OUTPUT_KMEANS_SUMMARY_CSV = "KMeans_cluster_summary.csv"
OUTPUT_KMEANS_ZSCORE_CSV = "KMeans_cluster_zscore.csv"

RANDOM_STATE = 42
N_KMEANS = 6            # số cụm macro sau KMeans (bạn có thể đổi)
BIRCH_THRESHOLD = 0.5   # ngưỡng BIRCH

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv(INPUT_CSV)

# =========================
# 2. Chọn các cột đặc trưng (đã là số/mã số)
# =========================
features = ["Age", "Children", "Income", "Total_Spent"]
#  "Marital_Status", "Education", bỏ các trường này vì có thể nó không được đánh giá đúng khi cleaned dataset
X = df[features].copy()

# Xử lý thiếu (nếu có)
X = X.replace([np.inf, -np.inf], np.nan)
for col in features:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

# =========================
# 3. Chuẩn hóa (StandardScaler)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. BIRCH: tạo micro-clusters (CF)
# =========================
birch_model = Birch(threshold=BIRCH_THRESHOLD, n_clusters=None)
birch_model.fit(X_scaled)
df["CF_Label"] = birch_model.labels_

# =========================
# 5. Tính toán chỉ số CF: N, LS, SS, Centroid, Radius (trên dữ liệu gốc)
# =========================
cf_list = []
for label, group in df.groupby("CF_Label"):
    points = group[features].values
    N = len(points)
    LS = np.sum(points, axis=0)
    SS = np.sum(points**2, axis=0)
    centroid = LS / max(N, 1)
    # Radius (L2) toàn cụm: sqrt( sum_j (E[x_j^2] - (E[x_j])^2) )
    radius = float(np.sqrt(np.sum(SS / max(N, 1) - centroid**2)))
    cf_list.append({
        "CF_Label": label,
        "N": int(N),
        "LS": LS.tolist(),
        "SS": SS.tolist(),
        "Centroid": centroid.tolist(),
        "Radius": radius
    })

cf_extra_df = pd.DataFrame(cf_list)
cf_summary = df.groupby("CF_Label")[features].mean().reset_index()
cf_summary = cf_summary.merge(cf_extra_df, on="CF_Label")
print("Tổng số CF được tạo ra:", len(cf_summary))
print(cf_summary.head(3))

# 10.1 PCA scatter plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["CF_Label"], cmap="tab20", s=30, alpha=0.7)
plt.colorbar(scatter, label="CF Label")
plt.title("BIRCH Micro-Clusters (PCA 2D, StandardScaler)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 1) Fit PCA (2 thành phần để vẽ 2D; cần nhiều hơn thì tăng n_components)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)  # scores: toạ độ của từng khách hàng trên PC1, PC2

# 2) Tỷ lệ phương sai giải thích
explained = (pca.explained_variance_ratio_ * 100).round(2)
print("Explained variance (%):", explained.tolist())  # [PC1%, PC2%]

# 3) LOADINGS (trọng số của feature lên PC). Mỗi cột là 1 PC.
loadings = pd.DataFrame(
    pca.components_.T, index=features, columns=["PC1_loading", "PC2_loading"]
).sort_values("PC1_loading", ascending=False)
print("\n=== LOADINGS (trọng số feature -> PC) ===")
print(loadings.round(3))

# 4) ĐÓNG GÓP % (chuẩn hoá bình phương loadings theo từng PC)
contrib = (loadings**2)
contrib_pc = contrib.div(contrib.sum(axis=0), axis=1) * 100
print("\n=== ĐÓNG GÓP % của từng feature vào PC (chuẩn hoá) ===")
print(contrib_pc.round(1))

# 5) TƯƠNG QUAN giữa feature (đã chuẩn hoá) và PC scores
scores = pd.DataFrame(X_pca, columns=["PC1_score", "PC2_score"])
Z = pd.DataFrame(X_scaled, columns=features)  # dữ liệu đã chuẩn hoá
corr_pc = Z.join(scores).corr().loc[features, ["PC1_score", "PC2_score"]]
print("\n=== TƯƠNG QUAN(feature, PC score) ===")
print(corr_pc.round(2))

# (Tuỳ chọn) vẽ barplot loadings để nhìn nhanh feature nào chi phối PC
ax = loadings.plot(kind="bar", figsize=(8,4))
ax.set_title("PCA Loadings")
ax.set_ylabel("Loading")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# (Tuỳ chọn) vẽ biplot giản lược: scores + vector loading
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X_pca[:,0], X_pca[:,1], s=10, alpha=0.4)
ax.set_xlabel(f"PC1 ({explained[0]}%)")
ax.set_ylabel(f"PC2 ({explained[1]}%)")
ax.set_title("Biplot (rút gọn)")

# scale mũi tên cho dễ thấy
arrow_scale = 2.5
for i, feat in enumerate(features):
    ax.arrow(0, 0,
             pca.components_[0, i]*arrow_scale,
             pca.components_[1, i]*arrow_scale,
             head_width=0.05, length_includes_head=True, alpha=0.8)
    ax.text(pca.components_[0, i]*arrow_scale*1.1,
            pca.components_[1, i]*arrow_scale*1.1,
            feat, fontsize=9)
plt.tight_layout(); plt.show()
# =========================
# Tính N cho mỗi CF
cf_counts = df.groupby("CF_Label").size().rename("N")  # Series: index=CF_Label, value=N

# Ma trận đặc trưng trung bình (như bạn đang dùng)
cf_no_extra = cf_summary.set_index("CF_Label")[features]

# Gắn nhãn ytick theo dạng "CF_Label (N=...)"
page_size = 20
n_pages = math.ceil(len(cf_no_extra) / max(page_size, 1))

for i in range(n_pages):
    start, end = i * page_size, min((i + 1) * page_size, len(cf_no_extra))
    slice_features = cf_no_extra.iloc[start:end].copy()

    # Tạo nhãn kèm N
    labels_with_n = [
        f"{idx} (N={int(cf_counts.loc[idx])})" if idx in cf_counts.index else f"{idx} (N=0)"
        for idx in slice_features.index
    ]

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(slice_features, annot=True, fmt=".1f",
                     cmap="YlGnBu", annot_kws={"size": 8})
    ax.set_title(f"Đặc trưng trung bình CF (Trang {i+1}/{n_pages})")
    ax.set_xlabel("Feature")
    ax.set_ylabel("CF Label")

    # Thay nhãn trục Y bằng "CF_Label (N=...)"
    ax.set_yticklabels(labels_with_n, rotation=0, fontsize=8)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.show()

# Lưu CF summary
cf_summary.to_csv(OUTPUT_CF_CSV, index=False)

# =========================
# 6. KMeans trên các centroid của CF
# =========================
# Lấy centroid (dạng gốc) của từng CF để gom thành macro-cluster
centroids = np.vstack(cf_summary["Centroid"].values)

# Nên chuẩn hóa lại centroid trước khi KMeans
centroids_scaled = StandardScaler().fit_transform(centroids)

kmeans = KMeans(n_clusters=N_KMEANS, random_state=RANDOM_STATE, n_init="auto")
cf_summary["KMeans_Label"] = kmeans.fit_predict(centroids_scaled)

# Map từ CF_Label -> KMeans_Label để gán cho từng khách hàng
cf_to_macro = dict(zip(cf_summary["CF_Label"], cf_summary["KMeans_Label"]))
df["KMeans_Label"] = df["CF_Label"].map(cf_to_macro)

# =========================
# 7. Thống kê chi tiết theo KMeans
# =========================
# (a) Thống kê mô tả
kmeans_mean = df.groupby("KMeans_Label")[features].mean().sort_index()
kmeans_median = df.groupby("KMeans_Label")[features].median().sort_index()
kmeans_std = df.groupby("KMeans_Label")[features].std().sort_index()
kmeans_count = df["KMeans_Label"].value_counts().sort_index().rename("Count")

# (b) Gộp thành một bảng lớn
summary_table = kmeans_mean.copy()
summary_table.columns = [f"{c}_mean" for c in summary_table.columns]

tmp_median = kmeans_median.add_suffix("_median")
tmp_std = kmeans_std.add_suffix("_std")

summary_table = summary_table.join(tmp_median, how="left").join(tmp_std, how="left")
summary_table = summary_table.join(kmeans_count, how="left")

# (c) Z-score theo cột (so với toàn bộ tập), để biết cụm nào cao/thấp bất thường
global_mean = df[features].mean()
global_std = df[features].std().replace(0, np.nan)

# z-score cho mean cụm: (mean_cum - mean_toan_bo) / std_toan_bo
z_table = (kmeans_mean - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

# Lưu bảng
summary_table.reset_index().to_csv(OUTPUT_KMEANS_SUMMARY_CSV, index=False)
z_table.reset_index().to_csv(OUTPUT_KMEANS_ZSCORE_CSV, index=False)

print("\n===== TÓM TẮT TRUNG BÌNH THEO CỤM (KMEANS) =====")
print(summary_table.round(2))
print("\n===== Z-SCORE (SO VỚI TOÀN BỘ) =====")
print(z_table.round(2))

# =========================
# 8. Diễn giải tự động đặc trưng nổi bật từng cụm
# =========================
def describe_cluster(k, zrow, top_k=3):
    """
    Tạo mô tả ngắn cho cụm k dựa trên z-score:
    - top_k feature cao nhất
    - top_k feature thấp nhất
    """
    vals = zrow.dropna()
    if vals.empty:
        return f"Cluster {k}: Không đủ dữ liệu để diễn giải."
    highs = vals.sort_values(ascending=False).head(top_k)
    lows = vals.sort_values(ascending=True).head(top_k)

    hi_txt = ", ".join([f"{f} (+{v:.2f}σ)" for f, v in highs.items()])
    lo_txt = ", ".join([f"{f} ({v:.2f}σ)" for f, v in lows.items()])

    return (f"Cluster {k} (n={int(kmeans_count.loc[k])}): "
            f"cao nổi bật ở [{hi_txt}] ; "
            f"thấp nổi bật ở [{lo_txt}].")

print("\n===== DIỄN GIẢI CỤM (TỰ ĐỘNG) =====")
for k in z_table.index:
    print(describe_cluster(k, z_table.loc[k], top_k=3))

# =========================
# 9. Trực quan hóa
# =========================

# 9.1 PCA toàn bộ khách hàng, tô màu theo KMeans
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=df["KMeans_Label"], cmap="tab10",
                      s=24, alpha=0.7, edgecolors="none")
plt.colorbar(scatter, label="KMeans Cluster")
plt.title("KMeans Clusters sau BIRCH (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

# 9.2 Heatmap trung bình (chuẩn hóa theo cột để dễ so sánh)
# Chuẩn hóa mean theo cột (min-max hoặc z-score) – ở đây dùng z-score theo cột cụm
mean_for_heat = kmeans_mean.copy()
mean_norm = (mean_for_heat - mean_for_heat.mean()) / (mean_for_heat.std().replace(0, np.nan))

plt.figure(figsize=(10, 6))
sns.heatmap(mean_norm.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("So sánh cụm theo Z-score (trên trung bình cụm)")
plt.xlabel("KMeans Cluster")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 9.3 Boxplot một vài feature then chốt theo cụm
for feat in ["Income", "Total_Spent", "Age"]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="KMeans_Label", y=feat, data=df, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=df, color="k", size=2, alpha=0.3)
    plt.title(f"Phân phối {feat} theo KMeans Cluster")
    plt.xlabel("KMeans Cluster")
    plt.ylabel(feat)
    plt.tight_layout()
    plt.show()

# 9.4 Bar chart số lượng theo cụm
plt.figure(figsize=(8, 5))
kmeans_count.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Số lượng khách hàng theo KMeans Cluster")
plt.xlabel("KMeans Cluster")
plt.ylabel("Số khách hàng")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 10. (Tuỳ chọn) Phân trang heatmap theo CF như bạn đã có


print("\n✅ Hoàn tất. Đã lưu:")
print(f"- CF summary: {OUTPUT_CF_CSV}")
print(f"- KMeans summary: {OUTPUT_KMEANS_SUMMARY_CSV}")
print(f"- KMeans z-score: {OUTPUT_KMEANS_ZSCORE_CSV}")
