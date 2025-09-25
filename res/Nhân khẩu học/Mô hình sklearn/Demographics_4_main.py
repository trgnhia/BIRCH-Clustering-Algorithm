# -*- coding: utf-8 -*-
import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_input_dataset.csv"

# Chọn K theo danh sách này (sẽ đánh giá và chọn K tốt nhất)
TRY_K_LIST = [4, 5, 6]

# BIRCH
BIRCH_THRESHOLD = 0.3       # nhỏ -> CF chặt hơn
MIN_CF_SIZE = 3              # lọc CF quá nhỏ trước KMeans (0 = không lọc)

# Biến số sử dụng
FEATURES = ["Age", "Children", "Income", "Total_Spent"]

# Tiền xử lý
APPLY_LOG_TRANSFORM = True   # log1p cho Income/Total_Spent
LOG_COLS = ["Income", "Total_Spent"]

RANDOM_STATE = 42

from pathlib import Path

# --- Thư mục output cùng cấp script ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Trường hợp chạy trong notebook/REPL
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_ver4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Xuất file
OUT_CF = str(OUT_DIR / "CF_summary.csv")
OUT_KMEANS_SUMMARY = str(OUT_DIR / "KMeans_summary.csv")
OUT_KMEANS_ZSCORE   = str(OUT_DIR / "KMeans_zscore.csv")

# =========================
# HÀM TIỆN ÍCH
# =========================
def load_and_prepare(df_path, features, log_cols=None, apply_log=False):
    df = pd.read_csv(df_path)
    X = df[features].copy()
    # xử lý thiếu
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in features:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    if apply_log and log_cols:
        for c in log_cols:
            if c in X.columns:
                X[c] = np.log1p(np.clip(X[c], a_min=0, a_max=None))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled, scaler

def build_cf(df, X_scaled, features, threshold=1.0):
    birch = Birch(threshold=threshold, n_clusters=None)
    birch.fit(X_scaled)
    df = df.copy()
    df["CF_Label"] = birch.labels_

    cf_list = []
    for label, g in df.groupby("CF_Label"):
        pts = g[features].values
        N = len(pts)
        LS = np.sum(pts, axis=0)
        SS = np.sum(pts**2, axis=0)
        centroid = LS / max(N, 1)
        radius = float(np.sqrt(np.sum(SS / max(N, 1) - centroid**2)))
        cf_list.append({
            "CF_Label": label,
            "N": int(N),
            "LS": LS.tolist(),
            "SS": SS.tolist(),
            "Centroid": centroid.tolist(),
            "Radius": radius
        })
    cf_extra = pd.DataFrame(cf_list)
    cf_mean = df.groupby("CF_Label")[features].mean().reset_index()
    cf_summary = cf_mean.merge(cf_extra, on="CF_Label")
    return df, cf_summary

def kmeans_weighted_on_cf(cf_summary, n_clusters, random_state=42):
    # centroid ở không gian gốc -> scale trước khi KMeans
    centroids = np.vstack(cf_summary["Centroid"].values)
    centroids_scaled = StandardScaler().fit_transform(centroids)
    weights = cf_summary["N"].values

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(centroids_scaled, sample_weight=weights)
    return km.labels_

def evaluate_macro_labels(df_with_cf, cf_summary, km_labels, X_scaled):
    # Map khách -> macro label
    cf_to_k = dict(zip(cf_summary["CF_Label"], km_labels))
    macro = df_with_cf["CF_Label"].map(cf_to_k).values

    # Nếu chỉ có 1 nhãn, không tính được
    if len(np.unique(macro)) < 2:
        return np.nan, np.nan, macro
    sil = silhouette_score(X_scaled, macro)
    db  = davies_bouldin_score(X_scaled, macro)
    return sil, db, macro

def pca_scatter(X_scaled, labels, title, cmap="tab10"):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, s=22, alpha=0.7, edgecolors="none")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]:.1f}%)")
    plt.tight_layout(); plt.show()

# =========================
# 1) Đọc & tiền xử lý
# =========================
df, X, X_scaled, scaler = load_and_prepare(
    INPUT_CSV, FEATURES, LOG_COLS, APPLY_LOG_TRANSFORM
)

# =========================
# 2) BIRCH -> CF
# =========================
df, cf_summary = build_cf(df, X_scaled, FEATURES, threshold=BIRCH_THRESHOLD)
print(f"Số CF tạo ra: {len(cf_summary)}")

# (tuỳ chọn) lọc CF nhỏ
if MIN_CF_SIZE > 0:
    kept = cf_summary["N"] >= MIN_CF_SIZE
    print(f"Lọc CF nhỏ: còn {kept.sum()}/{len(cf_summary)} CF (N >= {MIN_CF_SIZE})")
    cf_summary_big = cf_summary[kept].copy()
else:
    cf_summary_big = cf_summary.copy()

# Vẽ PCA theo CF (mỗi chấm = 1 KH, tô theo CF)
pca_scatter(X_scaled, df["CF_Label"], "BIRCH Micro-Clusters (PCA 2D)")



# 1) Fit PCA (2 thành phần để vẽ 2D; cần nhiều hơn thì tăng n_components)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)  # scores: toạ độ của từng khách hàng trên PC1, PC2

# 2) Tỷ lệ phương sai giải thích
explained = (pca.explained_variance_ratio_ * 100).round(2)
print("Explained variance (%):", explained.tolist())  # [PC1%, PC2%]

# 3) LOADINGS (trọng số của feature lên PC). Mỗi cột là 1 PC.
loadings = pd.DataFrame(
    pca.components_.T, index=FEATURES, columns=["PC1_loading", "PC2_loading"]
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
Z = pd.DataFrame(X_scaled, columns=FEATURES)  # dữ liệu đã chuẩn hoá
corr_pc = Z.join(scores).corr().loc[FEATURES, ["PC1_score", "PC2_score"]]
print("\n=== TƯƠNG QUAN(feature, PC score) ===")
print(corr_pc.round(2))

# (Tuỳ chọn) vẽ barplot loadings để nhìn nhanh feature nào chi phối PC
ax = loadings.plot(kind="bar", figsize=(8,4))
ax.set_title("PCA Loadings")
ax.set_ylabel("Loading")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# # (Tuỳ chọn) vẽ biplot giản lược: scores + vector loading
# fig, ax = plt.subplots(figsize=(8,6))
# ax.scatter(X_pca[:,0], X_pca[:,1], s=10, alpha=0.4)
# ax.set_xlabel(f"PC1 ({explained[0]}%)")
# ax.set_ylabel(f"PC2 ({explained[1]}%)")
# ax.set_title("Biplot (rút gọn)")

# # scale mũi tên cho dễ thấy
# arrow_scale = 2.5
# for i, feat in enumerate(FEATURES):
#     ax.arrow(0, 0,
#              pca.components_[0, i]*arrow_scale,
#              pca.components_[1, i]*arrow_scale,
#              head_width=0.05, length_includes_head=True, alpha=0.8)
#     ax.text(pca.components_[0, i]*arrow_scale*1.1,
#             pca.components_[1, i]*arrow_scale*1.1,
#             feat, fontsize=9)
# plt.tight_layout(); plt.show()

# Heatmap CF (kèm N trong nhãn)
cf_no_extra = cf_summary.set_index("CF_Label")[FEATURES]
cf_counts = cf_summary.set_index("CF_Label")["N"]
page_size = 20
n_pages = math.ceil(len(cf_no_extra) / max(page_size, 1))
for i in range(n_pages):
    start, end = i * page_size, min((i + 1) * page_size, len(cf_no_extra))
    sl = cf_no_extra.iloc[start:end].copy()
    ylabels = [f"{idx} (N={int(cf_counts.loc[idx])})" for idx in sl.index]

    plt.figure(figsize=(12,6))
    ax = sns.heatmap(sl, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"size":8})
    ax.set_title(f"Đặc trưng trung bình CF (Trang {i+1}/{n_pages})")
    ax.set_xlabel("Feature"); ax.set_ylabel("CF Label")
    ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout(); plt.show()
# =========================
# 3) Chọn K tốt nhất (KMeans có trọng số)
# =========================
results = []
best = None
for k in TRY_K_LIST:
    labels_km = kmeans_weighted_on_cf(cf_summary_big, k, random_state=RANDOM_STATE)
    cf_to_k = dict(zip(cf_summary_big["CF_Label"], labels_km))

    # Ánh xạ: CF nhỏ (nếu bị lọc) -> gán tạm cụm gần nhất bằng centroid (sử dụng KMeans predict)
    if len(cf_summary_big) < len(cf_summary):
        # Train KMeans lại trên all CF để predict cho CF nhỏ
        centroids_big = np.vstack(cf_summary_big["Centroid"].values)
        scaler_c = StandardScaler().fit(centroids_big)
        km_all = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km_all.fit(scaler_c.transform(centroids_big), sample_weight=cf_summary_big["N"].values)

        # predict cho CF nhỏ
        small = cf_summary[~cf_summary["CF_Label"].isin(cf_summary_big["CF_Label"])]
        if len(small) > 0:
            preds = km_all.predict(scaler_c.transform(np.vstack(small["Centroid"].values)))
            for lab, p in zip(small["CF_Label"].values, preds):
                cf_to_k[lab] = int(p)

    macro = df["CF_Label"].map(cf_to_k).values
    if len(np.unique(macro)) < 2:
        sil, db = np.nan, np.nan
    else:
        sil = silhouette_score(X_scaled, macro)
        db = davies_bouldin_score(X_scaled, macro)

    results.append((k, sil, db))
    if best is None or (not np.isnan(sil) and sil > best[1]):
        best = (k, sil, db, cf_to_k)

print("\n== Đánh giá K ==")
print(pd.DataFrame(results, columns=["K","Silhouette","DaviesBouldin"]).round(3))

# Dùng K tốt nhất
K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
df["KMeans_Label"] = df["CF_Label"].map(CF_TO_K)
print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")

# =========================
# 4) Báo cáo KMeans (trên khách hàng)
# =========================
kmeans_mean   = df.groupby("KMeans_Label")[FEATURES].mean().sort_index()
kmeans_median = df.groupby("KMeans_Label")[FEATURES].median().sort_index()
kmeans_std    = df.groupby("KMeans_Label")[FEATURES].std().sort_index()
kmeans_count  = df["KMeans_Label"].value_counts().sort_index().rename("Count")

summary_table = (
    kmeans_mean.add_suffix("_mean")
    .join(kmeans_median.add_suffix("_median"))
    .join(kmeans_std.add_suffix("_std"))
    .join(kmeans_count)
)

global_mean = df[FEATURES].mean()
global_std  = df[FEATURES].std().replace(0, np.nan)
z_table = (kmeans_mean - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

summary_table.reset_index().to_csv(OUT_KMEANS_SUMMARY, index=False)
z_table.reset_index().to_csv(OUT_KMEANS_ZSCORE, index=False)

print("\n===== TÓM TẮT CỤM (KMeans) =====")
print(summary_table.round(2))
print("\n===== Z-SCORE so với toàn bộ =====")
print(z_table.round(2))

# =========================
# 5) Lưu CF summary & trực quan
# =========================
cf_summary.to_csv(OUT_CF, index=False)

# PCA theo KMeans
pca_scatter(X_scaled, df["KMeans_Label"], f"KMeans Clusters (PCA 2D) — K={K_BEST}")

# Heatmap z-score của cụm (dễ so sánh)
plt.figure(figsize=(10, 6))
sns.heatmap(((kmeans_mean - kmeans_mean.mean())/kmeans_mean.std()).T,
            annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Z-score theo trung bình cụm (chuẩn hoá theo cột)")
plt.xlabel("KMeans Cluster"); plt.ylabel("Feature")
plt.tight_layout(); plt.show()


# Boxplot (Income/Total_Spent/Age)
for feat in ["Income", "Total_Spent", "Age"]:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="KMeans_Label", y=feat, data=df, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=df, color="k", size=2, alpha=0.25)
    plt.title(f"Phân phối {feat} theo KMeans (K={K_BEST})")
    plt.tight_layout(); plt.show()

# 9.4 Bar chart số lượng theo cụm
plt.figure(figsize=(8, 5))
kmeans_count.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Số lượng khách hàng theo KMeans Cluster")
plt.xlabel("KMeans Cluster")
plt.ylabel("Số khách hàng")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\n✅ Hoàn tất.")
print(f"- CF summary: {OUT_CF}")
print(f"- KMeans summary: {OUT_KMEANS_SUMMARY}")
print(f"- KMeans z-score: {OUT_KMEANS_ZSCORE}")
