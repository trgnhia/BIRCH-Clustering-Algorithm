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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score as ARI

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_input_dataset.csv"
FEATURES = ["Age", "Children", "Income", "Total_Spent"]

# BIRCH
BIRCH_THRESHOLD = 0.3

# K cố định theo yêu cầu
K_FIXED = 6

# Tiền xử lý
APPLY_LOG_TRANSFORM = True          # log1p cho Income/Total_Spent để giảm outlier
LOG_COLS = ["Income", "Total_Spent"]

RANDOM_STATE = 42

from pathlib import Path

# --- Thư mục output cùng cấp script ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Trường hợp chạy trong notebook/REPL
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_ver_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# Xuất file
OUT_CF = str(OUT_DIR / "CF_summary_birch_none.csv")
OUT_SUMMARY_A =  str(OUT_DIR / "KMeans_summary_A_birchNone_k5.csv")
OUT_Z_A = str(OUT_DIR /"KMeans_zscore_A_birchNone_k5.csv")
OUT_SUMMARY_B = str(OUT_DIR / "CF_summary_birch_none.csv")
OUT_SUMMARY_C = str(OUT_DIR / "KMeans_summary_C_direct_k5.csv")
OUT_Z_C = str(OUT_DIR / "KMeans_zscore_C_direct_k5.csv")

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
    # log1p (tuỳ chọn)
    if apply_log and log_cols:
        for c in log_cols:
            if c in X.columns:
                X[c] = np.log1p(np.clip(X[c], a_min=0, a_max=None))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled, scaler

def build_cf_labels(df, X_scaled, features, threshold):
    """Birch n_clusters=None để tạo CF (micro-clusters) và summary CF."""
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

def kmeans_on_cf_weighted(cf_summary, k, random_state=42):
    """KMeans trên centroid CF với sample_weight = N."""
    centroids = np.vstack(cf_summary["Centroid"].values)
    centroids_scaled = StandardScaler().fit_transform(centroids)
    weights = cf_summary["N"].values
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(centroids_scaled, sample_weight=weights)
    return km

def eval_labels(name, labels, X_scaled):
    if len(np.unique(labels)) < 2:
        return dict(Method=name, Silhouette=np.nan, DB=np.nan, CH=np.nan)
    return dict(
        Method=name,
        Silhouette=silhouette_score(X_scaled, labels),
        DB=davies_bouldin_score(X_scaled, labels),
        CH=calinski_harabasz_score(X_scaled, labels),
    )

def pca_scatter(X_scaled, labels, title, cmap="tab10"):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap,
                     s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]:.1f}%)"); plt.ylabel(f"PC2 ({var[1]:.1f}%)")
    plt.tight_layout(); plt.show()

def cluster_summary(df, label_col, features, out_summary=None, out_z=None):
    mean = df.groupby(label_col)[features].mean().sort_index()
    med  = df.groupby(label_col)[features].median().sort_index()
    std  = df.groupby(label_col)[features].std().sort_index()
    cnt  = df[label_col].value_counts().sort_index().rename("Count")
    summary = (
        mean.add_suffix("_mean")
        .join(med.add_suffix("_median"))
        .join(std.add_suffix("_std"))
        .join(cnt)
    )
    global_mean = df[features].mean()
    global_std  = df[features].std().replace(0, np.nan)
    z_table = (mean - global_mean) / global_std
    z_table = z_table.replace([np.inf, -np.inf], np.nan)

    if out_summary: summary.reset_index().to_csv(out_summary, index=False)
    if out_z:       z_table.reset_index().to_csv(out_z, index=False)

    print(f"\n===== TÓM TẮT CỤM ({label_col}) =====")
    print(summary.round(2))
    print("\n===== Z-SCORE so với toàn bộ =====")
    print(z_table.round(2))
    return summary, z_table

# =========================
# 1) Đọc & tiền xử lý
# =========================
df, X, X_scaled, scaler = load_and_prepare(
    INPUT_CSV, FEATURES, LOG_COLS, APPLY_LOG_TRANSFORM
)

# =========================
# PHƯƠNG ÁN A: Birch(n_clusters=None) -> KMeans(k=5) weighted
# =========================
df_A, cf_summary_A = build_cf_labels(df, X_scaled, FEATURES, threshold=BIRCH_THRESHOLD)
print(f"Số CF (A - Birch None): {len(cf_summary_A)}")
cf_summary_A.to_csv(OUT_CF, index=False)

# KMeans trên centroid CF có trọng số
km_A = kmeans_on_cf_weighted(cf_summary_A, K_FIXED, random_state=RANDOM_STATE)
cf_to_k_A = dict(zip(cf_summary_A["CF_Label"], km_A.labels_))
df_A["KMeans_Label_A"] = df_A["CF_Label"].map(cf_to_k_A)

# Đánh giá & tổng hợp
metrics_A = eval_labels(f"A) BirchNone -> KMeans(k={K_FIXED}) weighted", df_A["KMeans_Label_A"].values, X_scaled)
summary_A, z_A = cluster_summary(df_A, "KMeans_Label_A", FEATURES, OUT_SUMMARY_A, OUT_Z_A)
pca_scatter(X_scaled, df_A["KMeans_Label_A"], f"A) KMeans after Birch CF (k={K_FIXED})")

# =========================
# PHƯƠNG ÁN B: Birch(n_clusters=5) (gộp cuối trong Birch)
# =========================
birch_B = Birch(threshold=BIRCH_THRESHOLD, n_clusters=K_FIXED)  # Agglomerative cuối
birch_B.fit(X_scaled)
df_B = df.copy()
df_B["Birch_Label_B"] = birch_B.labels_

metrics_B = eval_labels(f"B) Birch(n_clusters={K_FIXED})", df_B["Birch_Label_B"].values, X_scaled)
# (Tuỳ chọn) tóm tắt theo nhãn Birch
summary_B, _ = cluster_summary(df_B, "Birch_Label_B", FEATURES, out_summary=OUT_SUMMARY_B, out_z=None)
pca_scatter(X_scaled, df_B["Birch_Label_B"], f"B) Birch final labels (k={K_FIXED})")

# =========================
# PHƯƠNG ÁN C: KMeans trực tiếp k=5 (không qua Birch)
# =========================
km_C = KMeans(n_clusters=K_FIXED, random_state=RANDOM_STATE, n_init="auto")
df_C = df.copy()
df_C["KMeans_Label_C"] = km_C.fit_predict(X_scaled)

metrics_C = eval_labels(f"C) KMeans trực tiếp (k={K_FIXED})", df_C["KMeans_Label_C"].values, X_scaled)
summary_C, z_C = cluster_summary(df_C, "KMeans_Label_C", FEATURES, OUT_SUMMARY_C, OUT_Z_C)
pca_scatter(X_scaled, df_C["KMeans_Label_C"], f"C) KMeans direct (k={K_FIXED})")

# =========================
# SO SÁNH CHẤT LƯỢNG + NHẤT QUÁN NHÃN
# =========================
cmp = pd.DataFrame([metrics_A, metrics_B, metrics_C]).round(3)
print("\n=== So sánh chất lượng (Silhouette↑, CH↑ tốt; DB↓ tốt) ===")
print(cmp)

print("\n=== ARI giữa các phương án (1.0 = giống hệt, ~0 = ngẫu nhiên) ===")
print("A vs B:", f"{ARI(df_A['KMeans_Label_A'], df_B['Birch_Label_B']):.3f}")
print("A vs C:", f"{ARI(df_A['KMeans_Label_A'], df_C['KMeans_Label_C']):.3f}")
print("B vs C:", f"{ARI(df_B['Birch_Label_B'], df_C['KMeans_Label_C']):.3f}")

# =========================
# (Tuỳ chọn) hiển thị phân bố kích thước cụm
# =========================
plt.figure(figsize=(8,4))
df_A["KMeans_Label_A"].value_counts().sort_index().plot(kind="bar")
plt.title("A) Count per cluster"); plt.xlabel("Cluster"); plt.ylabel("Count"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
df_B["Birch_Label_B"].value_counts().sort_index().plot(kind="bar")
plt.title("B) Count per cluster"); plt.xlabel("Cluster"); plt.ylabel("Count"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
df_C["KMeans_Label_C"].value_counts().sort_index().plot(kind="bar")
plt.title("C) Count per cluster"); plt.xlabel("Cluster"); plt.ylabel("Count"); plt.tight_layout(); plt.show()
