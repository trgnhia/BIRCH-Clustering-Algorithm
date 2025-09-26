# -*- coding: utf-8 -*-
import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.colors import ListedColormap

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_input_dataset.csv"

# Nhóm chi tiêu theo sản phẩm
FEATURES_PRODUCTS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]

# Thêm Income & Total_Spent như bạn yêu cầu
FEATURES_MONETARY = ["Income", "Total_Spent"]

# Tập feature dùng để phân cụm
FEATURES = FEATURES_PRODUCTS + FEATURES_MONETARY

# Thử K và chọn K tốt nhất theo Silhouette (tham khảo thêm DB)
TRY_K_LIST = [4, 5, 6]

# BIRCH (micro-clusters)
BIRCH_THRESHOLD = 0.3     # bạn đã xác nhận 0.2 là "điểm ngọt"
MIN_CF_SIZE = 3           # loại CF quá nhỏ trước khi gom KMeans

# Tiền xử lý
APPLY_LOG_TRANSFORM = True
LOG_COLS = FEATURES        # log cho toàn bộ cột tiền

RANDOM_STATE = 42

# --- Thư mục output cùng cấp script ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_product_mix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CF               = str(OUT_DIR / "CF_summary.csv")
OUT_KMEANS_SUMMARY   = str(OUT_DIR / "KMeans_summary.csv")
OUT_KMEANS_ZSCORE    = str(OUT_DIR / "KMeans_zscore.csv")
OUT_PRODUCT_MIX_CSV  = str(OUT_DIR / "ProductMix_byCluster.csv")
OUT_EVAL_K           = str(OUT_DIR / "EvalK.csv")

# =========================
# HÀM TIỆN ÍCH
# =========================
def load_and_prepare(df_path, features, log_cols=None, apply_log=False):
    df = pd.read_csv(df_path)
    X_raw = df[features].copy()   # bản gốc để báo cáo
    X = df[features].copy()

    # xử lý thiếu / vô cực
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
    return df, X_raw, X, X_scaled, scaler

def build_cf_consistent(df, X_raw, X_log, X_scaled, features, threshold=1.0):
    birch = Birch(threshold=threshold, n_clusters=None)
    birch.fit(X_scaled)
    labels = birch.labels_
    df = df.copy()
    df["CF_Label"] = labels

    rows = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        N = int(len(idx))
        centroid_scaled = X_scaled[idx].mean(axis=0)
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))
        mean_raw = pd.Series(X_raw.iloc[idx].mean(axis=0), index=features)

        rows.append({
            "CF_Label": lab,
            "N": N,
            "Centroid_scaled": centroid_scaled,
            "Radius_scaled": radius_scaled,
            **{f"{c}_mean_raw": mean_raw[c] for c in features},
        })

    cf_summary = pd.DataFrame(rows)
    # Thêm bảng mean raw gọn
    cf_mean_raw = df.groupby("CF_Label")[features].mean().reset_index()
    cf_summary = cf_mean_raw.merge(
        cf_summary.drop(columns=[f"{c}_mean_raw" for c in features]),
        on="CF_Label", how="left"
    )
    return df, cf_summary

def kmeans_weighted_on_cf_scaled(cf_summary, n_clusters, random_state=42):
    cents = np.vstack(cf_summary["Centroid_scaled"].values)
    weights = cf_summary["N"].values
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(cents, sample_weight=weights)
    return km

def map_all_cf_to_klabels(df_with_cf, cf_summary, km_big, cf_summary_big):
    cf_to_k = {}
    labs_big = cf_summary_big["CF_Label"].values
    preds_big = km_big.labels_
    for lab, p in zip(labs_big, preds_big):
        cf_to_k[int(lab)] = int(p)
    if len(cf_summary_big) < len(cf_summary):
        small = cf_summary[~cf_summary["CF_Label"].isin(labs_big)]
        if len(small) > 0:
            cent_small = np.vstack(small["Centroid_scaled"].values)
            preds_small = km_big.predict(cent_small)
            for lab, p in zip(small["CF_Label"].values, preds_small):
                cf_to_k[int(lab)] = int(p)
    macro = df_with_cf["CF_Label"].map(cf_to_k).values
    return cf_to_k, macro

def pca_fit_and_print(X_scaled, feature_names):
    """Trả về (X_pca, var%, pca_model, loadings_df, contrib_df, corr_df)"""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained = (pca.explained_variance_ratio_ * 100).round(2)
    print("Explained variance (%):", explained.tolist())

    loadings = pd.DataFrame(
        pca.components_.T, index=feature_names, columns=["PC1_loading","PC2_loading"]
    ).sort_values("PC1_loading", ascending=False)
    print("\n=== LOADINGS (trọng số feature -> PC) ===")
    print(loadings.round(3))

    contrib = (loadings**2)
    contrib_pc = contrib.div(contrib.sum(axis=0), axis=1) * 100
    print("\n=== ĐÓNG GÓP % của từng feature vào PC (chuẩn hoá) ===")
    print(contrib_pc.round(1))

    scores = pd.DataFrame(X_pca, columns=["PC1_score","PC2_score"])
    Z = pd.DataFrame(X_scaled, columns=feature_names)
    corr_pc = Z.join(scores).corr().loc[feature_names, ["PC1_score","PC2_score"]]
    print("\n=== TƯƠNG QUAN(feature, PC score) ===")
    print(corr_pc.round(2))

    return X_pca, explained, pca, loadings, contrib_pc, corr_pc

def pca_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    n_clusters = len(np.unique(labels))
    cmap_custom = ListedColormap(plt.cm.get_cmap("tab10").colors[:n_clusters])

    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap_custom,
                     s=22, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

# =========================
# 1) Đọc & tiền xử lý
# =========================
df, X_raw, X_log, X_scaled, scaler = load_and_prepare(
    INPUT_CSV, FEATURES, LOG_COLS, APPLY_LOG_TRANSFORM
)

# =========================
# 2) BIRCH → CF
# =========================
df, cf_summary = build_cf_consistent(df, X_raw, X_log, X_scaled, FEATURES, threshold=BIRCH_THRESHOLD)
print(f"Số CF tạo ra: {len(cf_summary)}")

# Lọc CF nhỏ
if MIN_CF_SIZE > 0:
    kept = cf_summary["N"] >= MIN_CF_SIZE
    print(f"Lọc CF nhỏ: còn {kept.sum()}/{len(cf_summary)} CF (N >= {MIN_CF_SIZE})")
    cf_summary_big = cf_summary[kept].copy()
else:
    cf_summary_big = cf_summary.copy()

# =========================
# [ADD PLOT] PCA scatter theo CF (mỗi chấm = 1 KH, màu = CF)
# =========================
pca_scatter(X_scaled, df["CF_Label"], "BIRCH Micro-Clusters (PCA 2D)")

# =========================
# [ADD PLOT] Phân tích PCA (loadings / đóng góp % / tương quan) trên X_scaled
# =========================
X_pca, explained, pca_model, loadings_df, contrib_pc_df, corr_df = pca_fit_and_print(X_scaled, FEATURES)

ax = loadings_df.plot(kind="bar", figsize=(9,4))
ax.set_title("PCA Loadings"); ax.set_ylabel("Loading")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# =========================
# [ADD PLOT] Heatmap CF (mean ở thang gốc, kèm N, phân trang)
# =========================
# cf_no_extra = cf_summary.set_index("CF_Label")[FEATURES]  # mean raw theo CF
# cf_counts = cf_summary.set_index("CF_Label")["N"]
# page_size = 20
# n_pages = math.ceil(len(cf_no_extra)/max(page_size,1))
# for i in range(n_pages):
#     sl = cf_no_extra.iloc[i*page_size : min((i+1)*page_size, len(cf_no_extra))].copy()
#     ylabels = [f"{idx} (N={int(cf_counts.loc[idx])})" for idx in sl.index]
#     plt.figure(figsize=(12,6))
#     ax = sns.heatmap(sl, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"size":8})
#     ax.set_title(f"Đặc trưng trung bình CF (Trang {i+1}/{n_pages})")
#     ax.set_xlabel("Feature"); ax.set_ylabel("CF Label")
#     ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
#     plt.xticks(rotation=45, ha="right", fontsize=9)
#     plt.tight_layout(); plt.show()

# =========================
# 3) Chọn K
# =========================
rows = []
best = None
for k in TRY_K_LIST:
    km_big = kmeans_weighted_on_cf_scaled(cf_summary_big, k, random_state=RANDOM_STATE)
    cf_to_k, macro = map_all_cf_to_klabels(df, cf_summary, km_big, cf_summary_big)
    if len(np.unique(macro)) < 2:
        s, db = np.nan, np.nan
    else:
        s  = silhouette_score(X_scaled, macro)
        db = davies_bouldin_score(X_scaled, macro)
    rows.append((k, s, db))
    if best is None or (not np.isnan(s) and s > best[1]):
        best = (k, s, db, cf_to_k)

eval_df = pd.DataFrame(rows, columns=["K","Silhouette","DaviesBouldin"]).round(3)
eval_df.to_csv(OUT_EVAL_K, index=False)
print("\n== Đánh giá K ==")
print(eval_df)

K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
df["KMeans_Label"] = df["CF_Label"].map(CF_TO_K)
print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")

# =========================
# 4) Báo cáo theo cụm (thang gốc)
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

# Z-score so với toàn bộ
global_mean = df[FEATURES].mean()
global_std  = df[FEATURES].std().replace(0, np.nan)
z_table = (kmeans_mean - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

summary_table.reset_index().to_csv(OUT_KMEANS_SUMMARY, index=False)
z_table.reset_index().to_csv(OUT_KMEANS_ZSCORE, index=False)

print("\n===== TÓM TẮT CỤM (mean/median/std + count) =====")
print(summary_table.round(2))
print("\n===== Z-SCORE so với toàn bộ =====")
print(z_table.round(2))

# =========================
# 5) Product-mix % theo cụm
# =========================
prod_sum = df[FEATURES_PRODUCTS].sum(axis=1).replace(0, np.nan)  # tránh chia 0
product_mix = df[FEATURES_PRODUCTS].div(prod_sum, axis=0)

product_mix_by_cluster = product_mix.groupby(df["KMeans_Label"]).mean()
product_mix_by_cluster.to_csv(OUT_PRODUCT_MIX_CSV)

print("\n===== PRODUCT-MIX % theo cụm (hàng=cluster, cột=nhóm) =====")
print((product_mix_by_cluster*100).round(1))

# [ADD PLOT] stacked bar product-mix %
(product_mix_by_cluster*100).plot(kind="bar", stacked=True, figsize=(10,6))
plt.ylabel("Tỉ trọng (%)")
plt.title("Product-mix trung bình theo cụm")
plt.legend(title="Nhóm sản phẩm", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.show()

# =========================
# 6) PCA scatter theo cụm (trên X_scaled) — KMeans
# =========================
pca_scatter(X_scaled, df["KMeans_Label"], f"KMeans Clusters (K={K_BEST})")

# =========================
# 7) Heatmap z-score giữa các cụm (chuẩn hoá theo cột)
# =========================
plt.figure(figsize=(12,6))
sns.heatmap(((kmeans_mean - kmeans_mean.mean())/kmeans_mean.std()).T,
            annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Z-score theo trung bình cụm (chuẩn hoá theo cột)")
plt.xlabel("KMeans Cluster"); plt.ylabel("Feature")
plt.tight_layout(); plt.show()

# =========================
# 8) Một vài biểu đồ phân phối hữu ích
# =========================
for feat in ["Income", "Total_Spent"]:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="KMeans_Label", y=feat, data=df, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=df, color="k", size=2, alpha=0.25)
    plt.title(f"Phân phối {feat} theo cụm (K={K_BEST})")
    plt.tight_layout(); plt.show()

# [ADD PLOT] CF size & radius (chuẩn hoá)
plt.figure(figsize=(8,4))
cf_summary["N"].hist(bins=20)
plt.title("Phân phối kích thước CF (N)"); plt.xlabel("N"); plt.ylabel("Số CF")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
cf_summary["Radius_scaled"].hist(bins=20)
plt.title("Phân phối bán kính CF (trong không gian chuẩn hoá)")
plt.xlabel("Radius_scaled"); plt.ylabel("Số CF")
plt.tight_layout(); plt.show()

# =========================
# 9) Lưu CF summary gọn (bỏ mảng numpy)
# =========================
cf_to_save = cf_summary.drop(columns=["Centroid_scaled"], errors="ignore")
cf_to_save.to_csv(OUT_CF, index=False)

print("\n✅ Hoàn tất.")
print(f"- EvalK: {OUT_EVAL_K}")
print(f"- CF summary: {OUT_CF}")
print(f"- KMeans summary: {OUT_KMEANS_SUMMARY}")
print(f"- KMeans z-score: {OUT_KMEANS_ZSCORE}")
print(f"- Product-mix: {OUT_PRODUCT_MIX_CSV}")
