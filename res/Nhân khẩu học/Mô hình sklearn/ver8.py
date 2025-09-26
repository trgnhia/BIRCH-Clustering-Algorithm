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

FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]

# Birch
BIRCH_THRESHOLD   = 0.2     # CF Level-1
ROLLUP_THRESHOLD  = 0.4     # CF Level-2 (gom lại từ CF L1)
MIN_CF_SIZE       = 3       # lọc CF nhỏ
MAX_CF_RADIUS_Q   = 0.90    # lọc CF quá loãng (bán kính > p90)
WEIGHT_EXP        = 1.2     # trọng số N**α khi train KMeans
TRY_K_LIST        = [3, 4, 5, 6]
RANDOM_STATE      = 42
DO_REFIT_ASSIGN   = True

# Thư mục output
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "output_ver5_birch_hier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# HÀM TIỆN ÍCH
# =========================
def load_and_prepare(df_path, features, log_cols=None, apply_log=False):
    df = pd.read_csv(df_path)
    X = df[features].copy()
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

def summarize_cf(df, cf_labels, X_scaled, features):
    """Tóm tắt CF: centroid_scaled, bán kính, mean gốc."""
    df_tmp = df.copy()
    df_tmp["CF_Label"] = cf_labels
    rows = []
    for lab, idx in df_tmp.groupby("CF_Label").indices.items():
        idx = np.asarray(list(idx))
        N = int(len(idx))
        centroid_scaled = X_scaled[idx].mean(axis=0)
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))
        mean_raw = df_tmp.iloc[idx][features].mean()
        rows.append({
            "CF_Label": lab,
            "N": N,
            "Centroid_scaled": centroid_scaled,
            "Radius_scaled": radius_scaled,
            **{f"{c}_mean_raw": mean_raw[c] for c in features}
        })
    cf_summary = pd.DataFrame(rows)
    return df_tmp, cf_summary

def filter_cf(cf_summary, min_size=0, max_radius_q=None):
    mask = pd.Series(True, index=cf_summary.index)
    if min_size > 0:
        mask &= cf_summary["N"] >= min_size
    if max_radius_q is not None:
        thr = cf_summary["Radius_scaled"].quantile(max_radius_q)
        mask &= cf_summary["Radius_scaled"] <= thr
    return cf_summary[mask].copy()

def rollup_cf(cf_summary_l1, rollup_threshold):
    cents = np.vstack(cf_summary_l1["Centroid_scaled"].values)
    birch2 = Birch(threshold=rollup_threshold, n_clusters=None)
    birch2.fit(cents)
    labels2 = birch2.labels_
    df_l1 = cf_summary_l1.copy()
    df_l1["L2_Label"] = labels2
    rows = []
    for lab, g in df_l1.groupby("L2_Label"):
        w = g["N"].values.astype(float)
        cents = np.vstack(g["Centroid_scaled"].values)
        w_sum = w.sum()
        centroid_l2 = (cents * w[:, None]).sum(axis=0) / max(w_sum, 1.0)
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

def pca_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    n_clusters = len(np.unique(labels))
    cmap_custom = ListedColormap(plt.cm.get_cmap("tab10").colors[:max(n_clusters,1)])
    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap_custom,
                     s=22, alpha=0.7, edgecolors="none")
    plt.colorbar(sc).set_label("Label")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

# =========================
# 1) LOAD & SCALE
# =========================
df, X, X_scaled, scaler = load_and_prepare(
    INPUT_CSV, FEATURES, log_cols=["Income", "Total_Spending"], apply_log=True
)

# =========================
# 2) BIRCH L1 + lọc CF
# =========================
cf_labels_l1 = Birch(threshold=BIRCH_THRESHOLD, n_clusters=None).fit_predict(X_scaled)
df_cf_l1, cf_l1 = summarize_cf(df, cf_labels_l1, X_scaled, FEATURES)
print(f"Số CF L1 tạo ra: {len(cf_l1)}")

cf_l1_filtered = filter_cf(cf_l1, min_size=MIN_CF_SIZE, max_radius_q=MAX_CF_RADIUS_Q)
print(f"Lọc CF L1: còn {len(cf_l1_filtered)}/{len(cf_l1)} (N >= {MIN_CF_SIZE}, radius <= p{int(MAX_CF_RADIUS_Q*100)})")

# =========================
# 3) ROLL-UP CF L2
# =========================
df_l1_with_l2, cf_l2 = rollup_cf(cf_l1_filtered, rollup_threshold=ROLLUP_THRESHOLD)
print(f"Số super-CF L2 sau roll-up: {len(cf_l2)}")

pca_scatter(X_scaled, df_cf_l1["CF_Label"], "BIRCH Micro-Clusters (L1)")

# =========================
# 4) CHỌN K (KMeans trên CF L2)
# =========================
rows, best = [], None
cf_l2_for_km = cf_l2.rename(columns={"L2_Label":"CF_Label"})

for k in TRY_K_LIST:
    km = kmeans_on_cf(cf_l2_for_km, k, weight_exp=WEIGHT_EXP, random_state=RANDOM_STATE)
    l2_to_k = dict(zip(cf_l2_for_km["CF_Label"].values, km.labels_))
    cf_to_k_all = {}
    for cf, l2 in zip(df_l1_with_l2["CF_Label"], df_l1_with_l2["L2_Label"]):
        cf_to_k_all[int(cf)] = l2_to_k[int(l2)]
    macro = df_cf_l1["CF_Label"].map(cf_to_k_all)

    # Bỏ các dòng chưa được ánh xạ (NaN)
    mask = macro.notna()
    macro_valid = macro[mask].astype(int).values
    X_valid = X_scaled[mask]

    if len(np.unique(macro_valid)) < 2:
        s, db = np.nan, np.nan
    else:
        s = silhouette_score(X_valid, macro_valid)
        db = davies_bouldin_score(X_valid, macro_valid)

    rows.append((k, s, db))
    if best is None or (not np.isnan(s) and s > best[1]):
        best = (k, s, db, cf_to_k_all)

eval_df = pd.DataFrame(rows, columns=["K","Silhouette","DaviesBouldin"]).round(3)
print("\n== Đánh giá K =="); print(eval_df)

K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
df_cf_l1["KMeans_Label"] = df_cf_l1["CF_Label"].map(CF_TO_K)
print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")

# =========================
# 5) REFINEMENT (nearest centroid)
# =========================
if DO_REFIT_ASSIGN:
    macro_centroids = []
    macro_labels_sorted = np.array(sorted(np.unique(df_cf_l1["KMeans_Label"])))
    for k in macro_labels_sorted:
        idx = np.where(df_cf_l1["KMeans_Label"].values == k)[0]
        macro_centroids.append(X_scaled[idx].mean(axis=0))
    macro_centroids = np.vstack(macro_centroids)
    d2 = ((X_scaled[:, None, :] - macro_centroids[None, :, :])**2).sum(axis=2)
    reassigned = macro_labels_sorted[np.argmin(d2, axis=1)]
    df_cf_l1["KMeans_Label"] = reassigned
    if len(np.unique(reassigned)) > 1:
        s2 = silhouette_score(X_scaled, reassigned)
        db2 = davies_bouldin_score(X_scaled, reassigned)
    else:
        s2, db2 = np.nan, np.nan
    print(f"[Refinement] Silhouette={s2:.3f} | DB={db2:.3f}")

# =========================
# 6) BÁO CÁO CỤM
# =========================
cluster_means = df_cf_l1.groupby("KMeans_Label")[FEATURES].mean().sort_index()
cluster_count = df_cf_l1["KMeans_Label"].value_counts().sort_index()
z_table = (cluster_means - df[FEATURES].mean()) / df[FEATURES].std().replace(0,np.nan)

print("\n===== TÓM TẮT CỤM (mean + count) =====")
print(cluster_means.round(2).join(cluster_count.rename("Count")))
print("\n===== Z-SCORE so với toàn bộ =====")
print(z_table.round(2))

# =========================
# 7) DỰ ĐOÁN KHÁCH HÀNG MỚI
# =========================
new_customer = {
    "Customer_Age": 35,
    "Children": 2,
    "Income": 45000,
    "Total_Spending": 1200
}
new_df = pd.DataFrame([new_customer])
new_df["Income"] = np.log1p(new_df["Income"])
new_df["Total_Spending"] = np.log1p(new_df["Total_Spending"])
new_scaled = scaler.transform(new_df[FEATURES])

macro_labels_sorted = np.array(sorted(np.unique(df_cf_l1["KMeans_Label"])))
macro_centroids = np.vstack([
    X_scaled[df_cf_l1["KMeans_Label"].values == k].mean(axis=0)
    for k in macro_labels_sorted
])
dists = ((macro_centroids - new_scaled)**2).sum(axis=1)
pred_label = int(macro_labels_sorted[np.argmin(dists)])

print("\n===== NEW CUSTOMER =====")
print(new_customer)
print("→ Thuộc cụm:", pred_label)
print("\n=== ĐẶC TRƯNG CỤM ===")
print(cluster_means.loc[pred_label].round(2))
