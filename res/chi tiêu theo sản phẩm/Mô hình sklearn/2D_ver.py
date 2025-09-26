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
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"

# 6 cột chi tiêu theo sản phẩm (chỉ dùng các cột này)
PRODUCTS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]

# BIRCH & KMeans
BIRCH_THRESHOLD   = 0.20           # level-1 (leaf-like)
ROLLUP_THRESHOLD  = 0.50           # [NEW] level-2 (non-leaf / roll-up từ CF level-1)
MIN_CF_SIZE       = 3              # lọc CF nhỏ
MAX_CF_RADIUS_Q   = 0.90           # [NEW] lọc CF bán kính > percentile 90% (khử CF quá loãng)
WEIGHT_EXP        = 1.6            # trọng số N**α khi KMeans trên CF
TRY_K_LIST        = [7]
RANDOM_STATE      = 42

# Refinement
DO_REFIT_ASSIGN   = True           # [NEW] gán lại nhãn theo macro-centroid (nearest-centroid)

# Xuất file
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_products_clr_2dhier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVAL_K            = str(OUT_DIR / "EvalK.csv")
OUT_CLUSTER_SUMMARY   = str(OUT_DIR / "Cluster_Summary.csv")
OUT_MIX_BY_CLUSTER    = str(OUT_DIR / "ProductMix_byCluster.csv")
OUT_CF_L1             = str(OUT_DIR / "CF_L1_summary.csv")
OUT_CF_L2             = str(OUT_DIR / "CF_L2_summary.csv")

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

    log_mix = np.log(mix + eps)
    clr = log_mix.sub(log_mix.mean(axis=1), axis=0)

    log_total = np.log1p(total.clip(lower=0))
    X = pd.concat([clr.add_prefix("clr_"), pd.Series(log_total, name="log_total")], axis=1)
    return X, mix, total

def birch_labels(X_scaled, threshold):
    birch = Birch(threshold=threshold, n_clusters=None)
    birch.fit(X_scaled)
    return birch.labels_

def summarize_cf(df_products, cf_labels, X_scaled, product_cols):
    """Tạo bảng CF summary (centroid ở X_scaled + mean thang gốc)."""
    tmp = df_products.copy()
    tmp["CF_Label"] = cf_labels
    rows = []
    for lab, idx in tmp.groupby("CF_Label").indices.items():
        idx = np.asarray(list(idx))
        N = int(len(idx))
        centroid_scaled = X_scaled[idx].mean(axis=0)
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))
        mean_raw = tmp.iloc[idx][product_cols].mean()
        rows.append({
            "CF_Label": lab,
            "N": N,
            "Centroid_scaled": centroid_scaled,
            "Radius_scaled": radius_scaled,
            **{f"{c}_mean_raw": mean_raw[c] for c in product_cols}
        })
    cf_summary = pd.DataFrame(rows)
    mean_raw_tbl = tmp.groupby("CF_Label")[product_cols].mean().reset_index()
    cf_summary = mean_raw_tbl.merge(
        cf_summary.drop(columns=[f"{c}_mean_raw" for c in product_cols]),
        on="CF_Label", how="left"
    )
    return tmp, cf_summary

def filter_cf(cf_summary, min_size=0, max_radius_q=None):
    """[NEW] Lọc CF theo N và theo ngưỡng radius (percentile)."""
    mask = pd.Series(True, index=cf_summary.index)
    if min_size > 0:
        mask &= cf_summary["N"] >= min_size
    if max_radius_q is not None:
        thr = cf_summary["Radius_scaled"].quantile(max_radius_q)
        mask &= cf_summary["Radius_scaled"] <= thr
    return cf_summary[mask].copy()

def rollup_cf(cf_summary_l1, rollup_threshold):
    """
    [NEW] Phân cấp: gom các CF level-1 (centroid_scaled) thành super-CF level-2
    bằng cách chạy Birch trên centroid L1 rồi gộp theo nhãn tạo ra.
    """
    cents = np.vstack(cf_summary_l1["Centroid_scaled"].values)
    birch2 = Birch(threshold=rollup_threshold, n_clusters=None)
    birch2.fit(cents)
    labels2 = birch2.labels_

    df_l1 = cf_summary_l1.copy()
    df_l1["L2_Label"] = labels2

    rows = []
    for lab, g in df_l1.groupby("L2_Label"):
        # gộp centroid_scaled có trọng số N
        w = g["N"].values.astype(float)
        cents = np.vstack(g["Centroid_scaled"].values)
        w_sum = w.sum()
        centroid_l2 = (cents * w[:, None]).sum(axis=0) / max(w_sum, 1.0)

        # radius của super-CF (ước lượng) = RMS của var + chênh centroid
        # đơn giản hoá: dùng phương sai nội bộ + phương sai giữa tâm
        # (đủ tốt cho mục đích nén/ổn định)
        rows.append({
            "L2_Label": int(lab),
            "N": int(w_sum),
            "Centroid_scaled": centroid_l2
        })
    cf_l2 = pd.DataFrame(rows)
    return df_l1, cf_l2

def kmeans_on_cf(cf_summary, n_clusters, weight_exp=1.0, random_state=42):
    cents = np.vstack(cf_summary["Centroid_scaled"].values)
    weights = (cf_summary["N"].astype(float).values) ** float(weight_exp)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    km.fit(cents, sample_weight=weights)
    return km

def map_cf_to_macro(df_with_cf, cf_summary_all, km, cf_summary_used):
    """
    Ánh xạ CF -> nhãn KMeans. Nếu chỉ train trên tập 'cf_summary_used' (đã lọc),
    thì predict cho phần CF còn lại bằng cùng mô hình.
    """
    cf_to_k = {}
    for lab, k in zip(cf_summary_used["CF_Label"].values, km.labels_):
        cf_to_k[int(lab)] = int(k)

    if len(cf_summary_used) < len(cf_summary_all):
        small = cf_summary_all[~cf_summary_all["CF_Label"].isin(cf_summary_used["CF_Label"])]
        if len(small) > 0:
            cents_small = np.vstack(small["Centroid_scaled"].values)
            preds = km.predict(cents_small)
            for lab, k in zip(small["CF_Label"].values, preds):
                cf_to_k[int(lab)] = int(k)

    macro = df_with_cf["CF_Label"].map(cf_to_k).values
    return cf_to_k, macro
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_colormap(n_clusters):
    """Tạo colormap với đúng số cluster."""
    base_cmap = cm.get_cmap("tab20")   # hoặc tab10, tab20c, Set3...
    colors = base_cmap.colors[:n_clusters] if hasattr(base_cmap, "colors") else base_cmap(np.linspace(0, 1, n_clusters))
    return mcolors.ListedColormap(colors)
def pca_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    n_clusters = len(np.unique(labels))
    cmap_custom = get_colormap(n_clusters)
    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap_custom,
                     s=22, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(sc); cbar.set_label("Label")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

def suggest_clusters_for_product(cluster_means_spend, cluster_mix, product_col, top_n=2, metric="mix"):
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
# 2) BIRCH — LEVEL 1 (leaf-like CF) + LỌC THEO RADIUS
# =========================
cf_labels_l1 = birch_labels(X_scaled, threshold=BIRCH_THRESHOLD)
df_cf_l1, cf_l1 = summarize_cf(df[PRODUCTS], cf_labels_l1, X_scaled, PRODUCTS)
print(f"Số CF L1 tạo ra: {len(cf_l1)}")

# lọc theo N và radius percentile
cf_l1_filtered = filter_cf(cf_l1, min_size=MIN_CF_SIZE, max_radius_q=MAX_CF_RADIUS_Q)  # [NEW]
print(f"Lọc CF L1: còn {len(cf_l1_filtered)}/{len(cf_l1)} "
      f"(N >= {MIN_CF_SIZE}, radius <= p{int(MAX_CF_RADIUS_Q*100)})")

# =========================
# 2b) BIRCH — LEVEL 2 (roll-up hierarchy)
# =========================
df_l1_with_l2, cf_l2 = rollup_cf(cf_l1_filtered, rollup_threshold=ROLLUP_THRESHOLD)  # [NEW]
print(f"Số super-CF L2 sau roll-up: {len(cf_l2)}")

# PCA theo CF L1 để quan sát
pca_scatter(X_scaled, df_cf_l1["CF_Label"], "BIRCH Micro-Clusters (CLR(mix) + log_total)")


# =========================
# PCA cho CF L2 (super-CF)
# =========================
# Gắn nhãn L2 cho từng khách hàng gốc
df_cf_l1 = df_cf_l1.merge(
    df_l1_with_l2[["CF_Label", "L2_Label"]].drop_duplicates(),
    on="CF_Label", how="left"
)

# Vẽ PCA scatter cho CF L2
pca_scatter(X_scaled, df_cf_l1["L2_Label"], "BIRCH Super-Clusters (CF L2, CLR(mix) + log_total)")

# =========================
# PHÂN TÍCH PCA (2 thành phần cho 2D)
# =========================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
explained = (pca.explained_variance_ratio_ * 100).round(2)

print("\n=== PCA Variance explained (%) ===")
print(f"PC1: {explained[0]}%")
print(f"PC2: {explained[1]}%")

# Loadings (ảnh hưởng của feature → PC)
loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=["PC1_loading", "PC2_loading"]
)
print("\n=== PCA LOADINGS (feature → PC) ===")
print(loadings.round(3))

# % đóng góp của từng feature vào PC
contrib = (loadings ** 2)
contrib_pc = contrib.div(contrib.sum(axis=0), axis=1) * 100
print("\n=== ĐÓNG GÓP % của từng feature vào PC ===")
print(contrib_pc.round(1))

# Tương quan giữa feature và PC scores
scores = pd.DataFrame(X_pca, columns=["PC1_score", "PC2_score"])
Z = pd.DataFrame(X_scaled, columns=X.columns)
corr_pc = Z.join(scores).corr().loc[X.columns, ["PC1_score", "PC2_score"]]
print("\n=== TƯƠNG QUAN (feature, PC score) ===")
print(corr_pc.round(2))

# Vẽ biểu đồ loadings
ax = loadings.plot(kind="bar", figsize=(8,5))
ax.set_title("PCA Loadings (feature → PC1, PC2)")
ax.set_ylabel("Loading")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# =========================
# 3) CHỌN K (KMeans trên LEVEL-2 để ổn định hơn)
# =========================
# Lấy bảng CF L2 theo format giống L1 để tái sử dụng kmeans_on_cf
cf_l2_for_km = cf_l2.rename(columns={"L2_Label":"CF_Label"})  # (CF_Label, N, Centroid_scaled)

rows, best = [], None
for k in TRY_K_LIST:
    km = kmeans_on_cf(cf_l2_for_km, k, weight_exp=WEIGHT_EXP, random_state=RANDOM_STATE)
    # ánh xạ: CF_L1_filtered -> L2 label -> macro k
    # trước hết map L2 -> macro k
    l2_to_k = dict(zip(cf_l2_for_km["CF_Label"].values, km.labels_))
    # CF L1 filtered có cột L2_Label
    cf_to_k_used = dict(zip(df_l1_with_l2["CF_Label"].values,
                            df_l1_with_l2["L2_Label"].map(l2_to_k).values))
    # CF L1 còn lại (nếu có) predict trực tiếp trên model KMeans dùng centroid L1
    cf_to_k_all = {}
    cf_to_k_all.update(cf_to_k_used)

    # bất kỳ CF L1 nào chưa có nhãn macro -> predict
    missing = cf_l1[~cf_l1["CF_Label"].isin(cf_to_k_used.keys())]
    if len(missing) > 0:
        preds = km.predict(np.vstack(missing["Centroid_scaled"].values))
        for lab, kpred in zip(missing["CF_Label"].values, preds):
            cf_to_k_all[int(lab)] = int(kpred)

    macro = df_cf_l1["CF_Label"].map(cf_to_k_all).values
    if len(np.unique(macro)) < 2:
        s, db = np.nan, np.nan
    else:
        s  = silhouette_score(X_scaled, macro)
        db = davies_bouldin_score(X_scaled, macro)
    rows.append((k, s, db))
    if best is None or (not np.isnan(s) and s > best[1]):
        best = (k, s, db, cf_to_k_all)

eval_df = pd.DataFrame(rows, columns=["K","Silhouette","DaviesBouldin"]).round(3)
eval_df.to_csv(OUT_EVAL_K, index=False)
print("\n== Đánh giá K =="); print(eval_df)

K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
df_cf_l1["KMeans_Label"] = df_cf_l1["CF_Label"].map(CF_TO_K)
print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")

# =========================
# 3b) REFINEMENT — gán lại theo macro-centroid (nearest centroid) [NEW]
# =========================
if DO_REFIT_ASSIGN:
    # macro-centroid trong X_scaled (trung bình tất cả điểm theo nhãn hiện có)
    macro_centroids = []
    macro_labels_sorted = np.array(sorted(np.unique(df_cf_l1["KMeans_Label"])))  # ép về np.array

    for k in macro_labels_sorted:
        idx = np.where(df_cf_l1["KMeans_Label"].values == k)[0]
        macro_centroids.append(X_scaled[idx].mean(axis=0))
    macro_centroids = np.vstack(macro_centroids)

    # gán lại nhãn
    d2 = ((X_scaled[:, None, :] - macro_centroids[None, :, :])**2).sum(axis=2)
    reassigned = macro_labels_sorted[np.argmin(d2, axis=1)]  # giờ hoạt động OK
    df_cf_l1["KMeans_Label"] = reassigned

    # đánh giá lại
    if len(np.unique(reassigned)) > 1:
        s2 = silhouette_score(X_scaled, reassigned)
        db2 = davies_bouldin_score(X_scaled, reassigned)
    else:
        s2, db2 = np.nan, np.nan

    print(f"[Refinement] Silhouette={s2:.3f} | DB={db2:.3f} (so với trước {SIL_BEST:.3f}/{DB_BEST:.3f})")
    SIL_BEST, DB_BEST = s2, db2


# =========================
# 4) BÁO CÁO CỤM (chi tiêu tuyệt đối & product-mix %)
# =========================
df_cf = df_cf_l1  # dùng dataframe có nhãn cuối

cluster_means_spend  = df_cf.groupby("KMeans_Label")[PRODUCTS].mean().sort_index()
cluster_median_spend = df_cf.groupby("KMeans_Label")[PRODUCTS].median().sort_index()
cluster_std_spend    = df_cf.groupby("KMeans_Label")[PRODUCTS].std().sort_index()
cluster_count        = df_cf["KMeans_Label"].value_counts().sort_index().rename("Count")

prod_sum = df_cf[PRODUCTS].sum(axis=1).replace(0, np.nan)
product_mix = df_cf[PRODUCTS].div(prod_sum, axis=0).fillna(0.0)
cluster_mix = product_mix.groupby(df_cf["KMeans_Label"]).mean().sort_index()

global_mean = df_cf[PRODUCTS].mean()
global_std  = df_cf[PRODUCTS].std().replace(0, np.nan)
z_table = (cluster_means_spend - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

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
# 5) GỢI Ý CỤM MỤC TIÊU CHO 1 SẢN PHẨM 
# =========================
product_A = "MntWines"
top_mix,  rank_mix  = suggest_clusters_for_product(cluster_means_spend, cluster_mix, product_A, top_n=2, metric="mix")
top_spend, rank_spd = suggest_clusters_for_product(cluster_means_spend, cluster_mix, product_A, top_n=2, metric="spend")

print(f"\n>>> Gợi ý target cho {product_A}:")
print("- Theo TỈ TRỌNG (mix): ưu tiên cụm", top_mix, "\n", (rank_mix*100).round(1).to_string())
print("- Theo CHI TIÊU tuyệt đối (spend): ưu tiên cụm", top_spend, "\n",
      cluster_means_spend[product_A].loc[rank_spd.index].round(1).to_string())

# =========================
# 6) TRỰC QUAN
# =========================
pca_scatter(X_scaled, df_cf["KMeans_Label"], f"KMeans Clusters (K={len(cluster_count)}) — CLR(mix) + log_total")

plt.figure(figsize=(12,6))
sns.heatmap(z_table.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Z-score chi tiêu tuyệt đối theo cụm (chuẩn hoá theo cột)")
plt.xlabel("Cluster"); plt.ylabel("Product")
plt.tight_layout(); plt.show()

(cluster_mix*100).plot(kind="bar", stacked=True, figsize=(10,6))
plt.ylabel("Tỉ trọng (%)"); plt.title("Product-mix trung bình theo cụm")
plt.legend(title="Product", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.show()

for feat in PRODUCTS:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="KMeans_Label", y=feat, data=df_cf, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=df_cf, color="k", size=2, alpha=0.25)
    plt.title(f"Phân phối {feat} theo cụm (K={len(cluster_count)})")
    plt.tight_layout(); plt.show()

# =========================
# 7) LƯU CF SUMMARY (L1 & L2)
# =========================
cf_l1.drop(columns=["Centroid_scaled"], errors="ignore").to_csv(OUT_CF_L1, index=False)
cf_l2.to_csv(OUT_CF_L2, index=False)

print("\n✅ Hoàn tất.")
print(f"- EvalK: {OUT_EVAL_K}")
print(f"- Cluster Summary: {OUT_CLUSTER_SUMMARY}")
print(f"- Product-mix by cluster: {OUT_MIX_BY_CLUSTER}")
print(f"- CF L1 summary: {OUT_CF_L1}")
print(f"- CF L2 summary: {OUT_CF_L2}")

# =========================
# 6) DỰ ĐOÁN KHÁCH HÀNG MỚI
# =========================
new_customer = {
    "MntWines": 350,
    "MntFruits": 20,
    "MntMeatProducts": 150,
    "MntFishProducts": 40,
    "MntSweetProducts": 30,
    "MntGoldProds": 50
}

new_df = pd.DataFrame([new_customer])

# Pipeline: mix %, CLR, log_total
new_total = new_df[PRODUCTS].sum(axis=1)
new_total_safe = new_total.replace(0, np.nan)
new_mix = new_df[PRODUCTS].div(new_total_safe, axis=0).fillna(0.0)

eps = 1e-6
log_mix = np.log(new_mix + eps)
new_clr = log_mix.sub(log_mix.mean(axis=1), axis=0)
new_log_total = np.log1p(new_total.clip(lower=0))

new_X = pd.concat([new_clr.add_prefix("clr_"),
                   pd.Series(new_log_total, name="log_total")], axis=1)

# Scale
new_scaled = scaler.transform(new_X)

# Nearest centroid trong không gian đã scale
macro_labels_sorted = np.array(sorted(np.unique(df_cf["KMeans_Label"])))
macro_centroids = np.vstack([
    X_scaled[df_cf["KMeans_Label"].values == k].mean(axis=0)
    for k in macro_labels_sorted
])
dists = ((macro_centroids - new_scaled)**2).sum(axis=1)
pred_idx = int(np.argmin(dists))
pred_label = int(macro_labels_sorted[pred_idx])

print("\n===== NEW CUSTOMER =====")
print(new_customer)
print("→ Khách hàng mới thuộc cụm:", pred_label)

# Lấy thông tin cụm đó
print("\n=== ĐẶC TRƯNG CỤM", pred_label, "===")
print(summary_table.loc[pred_label].round(2))
print("\n=== PRODUCT-MIX % của cụm", pred_label, "===")
print((cluster_mix*100).loc[pred_label].round(1))
