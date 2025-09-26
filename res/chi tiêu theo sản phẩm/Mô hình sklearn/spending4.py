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

# 6 cột chi tiêu theo sản phẩm (chỉ dùng các cột này)
PRODUCTS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]

# BIRCH & KMeans
BIRCH_THRESHOLD = 0.30       # thử thêm 0.28 / 0.32 để fine-tune
MIN_CF_SIZE     = 3          # lọc CF nhỏ (2–3 thường ổn)
WEIGHT_EXP      = 1.6        # trọng số luỹ thừa N**α (1.3–1.8), giảm nhiễu CF nhỏ
TRY_K_LIST      = [3, 4, 5, 6, 7]
RANDOM_STATE    = 42

# Xuất file
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_products_clr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVAL_K          = str(OUT_DIR / "EvalK.csv")
OUT_CLUSTER_SUMMARY = str(OUT_DIR / "Cluster_Summary.csv")
OUT_MIX_BY_CLUSTER  = str(OUT_DIR / "ProductMix_byCluster.csv")
OUT_CF              = str(OUT_DIR / "CF_summary.csv")

# =========================
# HÀM TIỆN ÍCH
# =========================
def assert_columns_exist(df, cols, label="dataset"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in {label}: {missing}")

def build_mix_clr_and_logtotal(df, product_cols, eps=1e-6):
    """Tạo product-mix %, CLR(mix) và log_total từ 6 cột Mnt*."""
    total = df[product_cols].sum(axis=1)
    total_safe = total.replace(0, np.nan)
    mix = df[product_cols].div(total_safe, axis=0).fillna(0.0)

    # CLR transform (Aitchison) cho dữ liệu compositional
    log_mix = np.log(mix + eps)
    clr = log_mix.sub(log_mix.mean(axis=1), axis=0)  # trừ log-mean theo từng dòng

    # log_total (giữ 1 trục quy mô nhẹ nhàng)
    log_total = np.log1p(total.clip(lower=0))

    # Ma trận feature đầu vào
    X = pd.concat([clr.add_prefix("clr_"), pd.Series(log_total, name="log_total")], axis=1)
    return X, mix, total

def birch_cf(X_scaled, threshold):
    birch = Birch(threshold=threshold, n_clusters=None)
    birch.fit(X_scaled)
    return birch.labels_

def summarize_cf(df, cf_labels, X_scaled, product_cols):
    df = df.copy()
    df["CF_Label"] = cf_labels
    rows = []
    for lab, idx in df.groupby("CF_Label").indices.items():
        idx = np.array(list(idx))
        N = int(len(idx))
        centroid_scaled = X_scaled[idx].mean(axis=0)
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))
        mean_raw = df.iloc[idx][product_cols].mean()
        rows.append({
            "CF_Label": lab,
            "N": N,
            "Centroid_scaled": centroid_scaled,
            "Radius_scaled": radius_scaled,
            **{f"{c}_mean_raw": mean_raw[c] for c in product_cols}
        })
    cf_summary = pd.DataFrame(rows)
    mean_raw_tbl = df.groupby("CF_Label")[product_cols].mean().reset_index()
    cf_summary = mean_raw_tbl.merge(
        cf_summary.drop(columns=[f"{c}_mean_raw" for c in product_cols]),
        on="CF_Label", how="left"
    )
    return df, cf_summary

def kmeans_on_cf(cf_summary, n_clusters, weight_exp=1.0, random_state=42):
    cents = np.vstack(cf_summary["Centroid_scaled"].values)
    weights = (cf_summary["N"].astype(float).values) ** float(weight_exp)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    km.fit(cents, sample_weight=weights)
    return km

def map_cf_to_macro(df_with_cf, cf_summary, km, cf_summary_big):
    cf_to_k = {}
    for lab, k in zip(cf_summary_big["CF_Label"].values, km.labels_):
        cf_to_k[int(lab)] = int(k)
    if len(cf_summary_big) < len(cf_summary):
        small = cf_summary[~cf_summary["CF_Label"].isin(cf_summary_big["CF_Label"])]
        if len(small) > 0:
            cents_small = np.vstack(small["Centroid_scaled"].values)
            preds = km.predict(cents_small)
            for lab, k in zip(small["CF_Label"].values, preds):
                cf_to_k[int(lab)] = int(k)
    macro = df_with_cf["CF_Label"].map(cf_to_k).values
    return cf_to_k, macro

def pca_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    n_clusters = len(np.unique(labels))
    cmap_custom = ListedColormap(plt.cm.get_cmap("tab10").colors[:max(n_clusters,1)])
    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap_custom,
                     s=22, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(sc); cbar.set_label("Label")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

def suggest_clusters_for_product(cluster_means_spend, cluster_mix, product_col, top_n=2, metric="mix"):
    """metric='mix' (sở thích) hoặc 'spend' (doanh thu tuyệt đối)"""
    if metric == "mix":
        s = cluster_mix[product_col]
    else:
        s = cluster_means_spend[product_col]
    ranking = s.sort_values(ascending=False)
    return ranking.index.tolist()[:top_n], ranking

# =========================
# 1) LOAD & FEATURE ENGINEERING (CLR + log_total)
# =========================
df = pd.read_csv(INPUT_CSV)
assert_columns_exist(df, PRODUCTS, "PRODUCTS")

X, product_mix_each, product_total = build_mix_clr_and_logtotal(df, PRODUCTS)
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 2) BIRCH → CF
# =========================
cf_labels = birch_cf(X_scaled, threshold=BIRCH_THRESHOLD)
df_cf, cf_summary = summarize_cf(df[PRODUCTS], cf_labels, X_scaled, PRODUCTS)
print(f"Số CF tạo ra: {len(cf_summary)}")

# Lọc CF nhỏ
if MIN_CF_SIZE > 0:
    kept = cf_summary["N"] >= MIN_CF_SIZE
    print(f"Lọc CF nhỏ: còn {kept.sum()}/{len(cf_summary)} CF (N >= {MIN_CF_SIZE})")
    cf_summary_big = cf_summary[kept].copy()
else:
    cf_summary_big = cf_summary.copy()

# PCA theo CF (để quan sát)
pca_scatter(X_scaled, df_cf["CF_Label"], "BIRCH Micro-Clusters (CLR(mix) + log_total)")

# =========================
# 3) CHỌN K
# =========================
rows, best = [], None
for k in TRY_K_LIST:
    km = kmeans_on_cf(cf_summary_big, k, weight_exp=WEIGHT_EXP, random_state=RANDOM_STATE)
    cf_to_k, macro = map_cf_to_macro(df_cf, cf_summary, km, cf_summary_big)
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
print("\n== Đánh giá K =="); print(eval_df)

K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
df_cf["KMeans_Label"] = df_cf["CF_Label"].map(CF_TO_K)
print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")

# =========================
# =========================
# 4) BÁO CÁO CỤM (chi tiêu tuyệt đối & product-mix %)
# =========================

# CHÚ Ý: df_cf đã có 6 cột PRODUCTS (vì summarize_cf nhận df[PRODUCTS]).
# => KHÔNG join với df nữa để tránh trùng tên cột.

# Chi tiêu tuyệt đối theo cụm (thang gốc)
cluster_means_spend  = df_cf.groupby("KMeans_Label")[PRODUCTS].mean().sort_index()
cluster_median_spend = df_cf.groupby("KMeans_Label")[PRODUCTS].median().sort_index()
cluster_std_spend    = df_cf.groupby("KMeans_Label")[PRODUCTS].std().sort_index()
cluster_count        = df_cf["KMeans_Label"].value_counts().sort_index().rename("Count")

# Product-mix % theo cụm
prod_sum = df_cf[PRODUCTS].sum(axis=1).replace(0, np.nan)
product_mix = df_cf[PRODUCTS].div(prod_sum, axis=0).fillna(0.0)
cluster_mix = product_mix.groupby(df_cf["KMeans_Label"]).mean().sort_index()

# Z-score theo cụm (trên thang tuyệt đối) để đọc hiểu
global_mean = df_cf[PRODUCTS].mean()
global_std  = df_cf[PRODUCTS].std().replace(0, np.nan)
z_table = (cluster_means_spend - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

# Xuất CSV
summary_table = (
    cluster_means_spend.add_suffix("_mean")
    .join(cluster_median_spend.add_suffix("_median"))
    .join(cluster_std_spend.add_suffix("_std"))
    .join(cluster_count)
)
summary_table.reset_index().to_csv(OUT_CLUSTER_SUMMARY, index=False)
cluster_mix.reset_index().to_csv(OUT_MIX_BY_CLUSTER, index=False)


print("\n===== TÓM TẮT CỤM (mean/median/std + count) =====")
print(summary_table.round(2))
print("\n===== PRODUCT-MIX % theo cụm =====")
print((cluster_mix*100).round(1))
print("\n===== Z-SCORE so với toàn bộ (chi tiêu tuyệt đối) =====")
print(z_table.round(2))

# =========================
# 5) GỢI Ý CỤM MỤC TIÊU CHO 1 SẢN PHẨM A
# =========================
product_A = "MntWines"  # đổi sang sản phẩm bạn cần
top_mix,  rank_mix  = suggest_clusters_for_product(cluster_means_spend, cluster_mix, product_A, top_n=2, metric="mix")
top_spend, rank_spd = suggest_clusters_for_product(cluster_means_spend, cluster_mix, product_A, top_n=2, metric="spend")

print(f"\n>>> Gợi ý target cho {product_A}:")
print("- Theo TỈ TRỌNG (mix): ưu tiên cụm", top_mix, "\n", (rank_mix*100).round(1).to_string())
print("- Theo CHI TIÊU tuyệt đối (spend): ưu tiên cụm", top_spend, "\n", cluster_means_spend[product_A].loc[rank_spd.index].round(1).to_string())

# =========================
# 6) TRỰC QUAN
# =========================
# PCA theo KMeans
pca_scatter(X_scaled, df_cf["KMeans_Label"], f"KMeans Clusters (K={K_BEST}) — CLR(mix) + log_total")

# Heatmap z-score (chi tiêu tuyệt đối)
plt.figure(figsize=(12,6))
sns.heatmap(z_table.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Z-score chi tiêu tuyệt đối theo cụm (chuẩn hoá theo cột)")
plt.xlabel("Cluster"); plt.ylabel("Product")
plt.tight_layout(); plt.show()

# Stacked bar product-mix %
(cluster_mix*100).plot(kind="bar", stacked=True, figsize=(10,6))
plt.ylabel("Tỉ trọng (%)")
plt.title("Product-mix trung bình theo cụm")
plt.legend(title="Product", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.show()

# Boxplot minh hoạ cho 2 sản phẩm
for feat in ["MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"]:
    tmp = df_cf   # df_cf đã có các cột Mnt*
    plt.figure(figsize=(8,5))
    sns.boxplot(x="KMeans_Label", y=feat, data=tmp, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=tmp, color="k", size=2, alpha=0.25)
    plt.title(f"Phân phối {feat} theo cụm (K={K_BEST})")
    plt.tight_layout(); plt.show()

# =========================
# 7) LƯU CF SUMMARY (bỏ mảng numpy cho gọn)
# =========================
cf_to_save = cf_summary.drop(columns=["Centroid_scaled"], errors="ignore")
cf_to_save.to_csv(OUT_CF, index=False)

print("\n✅ Hoàn tất.")
print(f"- EvalK: {OUT_EVAL_K}")
print(f"- Cluster Summary: {OUT_CLUSTER_SUMMARY}")
print(f"- Product-mix by cluster: {OUT_MIX_BY_CLUSTER}")
print(f"- CF summary: {OUT_CF}")
