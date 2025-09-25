# -*- coding: utf-8 -*-
import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from time import perf_counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.metrics.cluster import adjusted_rand_score as ARI

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_input_dataset.csv"
FEATURES = ["Age", "Children", "Income", "Total_Spent"]

# BIRCH
BIRCH_THRESHOLD = 0.3

# K cố định cho 3 phương án
K_FIXED = 6

# Tiền xử lý
APPLY_LOG_TRANSFORM = True                # log1p cho Income/Total_Spent để giảm outlier
LOG_COLS = ["Income", "Total_Spent"]

RANDOM_STATE = 42
plt.rcParams["font.size"] = 11

# --- Thư mục output cùng cấp script ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_ver_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tên file xuất
OUT_CF_A          = OUT_DIR / "CF_summary_A_birchNone.csv"
OUT_SUMMARY_A     = OUT_DIR / f"KMeans_summary_A_birchNone_k{K_FIXED}.csv"
OUT_Z_A           = OUT_DIR / f"KMeans_zscore_A_birchNone_k{K_FIXED}.csv"

OUT_SUMMARY_B     = OUT_DIR / f"Birch_summary_B_nclusters{k_FIXED if (k_FIXED:=K_FIXED) else K_FIXED}.csv"  # safe trick
# (để tránh cảnh báo, vẫn cứ dùng K_FIXED)
OUT_SUMMARY_B     = OUT_DIR / f"Birch_summary_B_nclusters{K_FIXED}.csv"

OUT_SUMMARY_C     = OUT_DIR / f"KMeans_summary_C_direct_k{K_FIXED}.csv"
OUT_Z_C           = OUT_DIR / f"KMeans_zscore_C_direct_k{K_FIXED}.csv"

OUT_TIME_CSV      = OUT_DIR / "runtime_comparison.csv"
OUT_METRICS_CSV   = OUT_DIR / "quality_comparison.csv"

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

def pca_scatter(X_scaled, labels, title, cmap="tab10", savepath=None):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap,
                     s=18, alpha=0.7, edgecolors="none")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]:.1f}%)"); plt.ylabel(f"PC2 ({var[1]:.1f}%)")
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()

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

def zscore_heatmap(z_table, title, savepath=None):
    plt.figure(figsize=(9, 6))
    sns.heatmap(z_table.T, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title); plt.xlabel("Cluster"); plt.ylabel("Feature")
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()

# =========================
# 1) Đọc & tiền xử lý
# =========================
df, X, X_scaled, scaler = load_and_prepare(
    INPUT_CSV, FEATURES, LOG_COLS, APPLY_LOG_TRANSFORM
)

# =========================
# PHƯƠNG ÁN A: Birch(n_clusters=None) -> KMeans weighted (k = K_FIXED)
# =========================
tA0 = perf_counter()
tA_cf0 = perf_counter()
df_A, cf_summary_A = build_cf_labels(df, X_scaled, FEATURES, threshold=BIRCH_THRESHOLD)
tA_cf = perf_counter() - tA_cf0
cf_summary_A.to_csv(OUT_CF_A, index=False)
print(f"Số CF (A - Birch None): {len(cf_summary_A)}")

tA_km0 = perf_counter()
km_A = kmeans_on_cf_weighted(cf_summary_A, K_FIXED, random_state=RANDOM_STATE)
cf_to_k_A = dict(zip(cf_summary_A["CF_Label"], km_A.labels_))
df_A["KMeans_Label_A"] = df_A["CF_Label"].map(cf_to_k_A)
tA_km = perf_counter() - tA_km0

tA_total = perf_counter() - tA0
metrics_A = eval_labels(f"A) BirchNone -> KMeans(k={K_FIXED}) weighted",
                        df_A["KMeans_Label_A"].values, X_scaled)
summary_A, z_A = cluster_summary(df_A, "KMeans_Label_A", FEATURES,
                                 out_summary=OUT_SUMMARY_A, out_z=OUT_Z_A)
pca_scatter(X_scaled, df_A["KMeans_Label_A"],
            f"A) KMeans after Birch CF (k={K_FIXED})",
            savepath=OUT_DIR / f"pca_A_k{K_FIXED}.png")
zscore_heatmap(z_A, f"A) Z-score theo trung bình cụm (k={K_FIXED})",
               savepath=OUT_DIR / f"zheat_A_k{K_FIXED}.png")

plt.figure(figsize=(8,4))
df_A["KMeans_Label_A"].value_counts().sort_index().plot(kind="bar", color="#69b3a2", edgecolor="black")
plt.title("A) Count per cluster"); plt.xlabel("Cluster"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(OUT_DIR / "counts_A.png", dpi=140, bbox_inches="tight"); plt.show()

# =========================
# PHƯƠNG ÁN B: Birch(n_clusters=K_FIXED) (gộp cuối trong Birch)
# =========================
tB0 = perf_counter()
birch_B = Birch(threshold=BIRCH_THRESHOLD, n_clusters=K_FIXED)
birch_B.fit(X_scaled)
df_B = df.copy()
df_B["Birch_Label_B"] = birch_B.labels_
tB_total = perf_counter() - tB0

metrics_B = eval_labels(f"B) Birch(n_clusters={K_FIXED})",
                        df_B["Birch_Label_B"].values, X_scaled)
summary_B, z_B = cluster_summary(df_B, "Birch_Label_B", FEATURES,
                                 out_summary=OUT_SUMMARY_B, out_z=None)
pca_scatter(X_scaled, df_B["Birch_Label_B"],
            f"B) Birch final labels (k={K_FIXED})",
            savepath=OUT_DIR / f"pca_B_k{K_FIXED}.png")
zscore_heatmap((summary_B[[f"{c}_mean" for c in FEATURES]]
                .pipe(lambda m: (m - m.mean())/m.std())).T,
               f"B) Z-score (chuẩn hoá theo cột cụm, k={K_FIXED})",
               savepath=OUT_DIR / f"zheat_B_k{K_FIXED}.png")

plt.figure(figsize=(8,4))
df_B["Birch_Label_B"].value_counts().sort_index().plot(kind="bar", color="#8ecae6", edgecolor="black")
plt.title("B) Count per cluster"); plt.xlabel("Cluster"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(OUT_DIR / "counts_B.png", dpi=140, bbox_inches="tight"); plt.show()

# =========================
# PHƯƠNG ÁN C: KMeans trực tiếp k=K_FIXED (không qua Birch)
# =========================
tC0 = perf_counter()
km_C = KMeans(n_clusters=K_FIXED, random_state=RANDOM_STATE, n_init="auto")
df_C = df.copy()
df_C["KMeans_Label_C"] = km_C.fit_predict(X_scaled)
tC_total = perf_counter() - tC0

metrics_C = eval_labels(f"C) KMeans trực tiếp (k={K_FIXED})",
                        df_C["KMeans_Label_C"].values, X_scaled)
summary_C, z_C = cluster_summary(df_C, "KMeans_Label_C", FEATURES,
                                 out_summary=OUT_SUMMARY_C, out_z=OUT_Z_C)
pca_scatter(X_scaled, df_C["KMeans_Label_C"],
            f"C) KMeans direct (k={K_FIXED})",
            savepath=OUT_DIR / f"pca_C_k{K_FIXED}.png")
zscore_heatmap(z_C, f"C) Z-score theo trung bình cụm (k={K_FIXED})",
               savepath=OUT_DIR / f"zheat_C_k{K_FIXED}.png")

plt.figure(figsize=(8,4))
df_C["KMeans_Label_C"].value_counts().sort_index().plot(kind="bar", color="#ffb703", edgecolor="black")
plt.title("C) Count per cluster"); plt.xlabel("Cluster"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(OUT_DIR / "counts_C.png", dpi=140, bbox_inches="tight"); plt.show()

# =========================
# SO SÁNH CHẤT LƯỢNG + NHẤT QUÁN NHÃN
# =========================
cmp = pd.DataFrame([metrics_A, metrics_B, metrics_C]).round(3)
print("\n=== So sánh chất lượng (Silhouette↑, CH↑ tốt; DB↓ tốt) ===")
print(cmp)
cmp.to_csv(OUT_METRICS_CSV, index=False)

print("\n=== ARI giữa các phương án (1.0 = giống hệt, ~0 = ngẫu nhiên) ===")
print("A vs B:", f"{ARI(df_A['KMeans_Label_A'], df_B['Birch_Label_B']):.3f}")
print("A vs C:", f"{ARI(df_A['KMeans_Label_A'], df_C['KMeans_Label_C']):.3f}")
print("B vs C:", f"{ARI(df_B['Birch_Label_B'], df_C['KMeans_Label_C']):.3f}")

# =========================
# THỐNG KÊ THỜI GIAN CHẠY
# =========================
time_rows = [
    {"Method": f"A) BirchNone + KMeans(k={K_FIXED})",
     "Total_s": tA_total, "CF_build_s": tA_cf, "KMeans_fit_s": tA_km},

    {"Method": f"B) Birch(n_clusters={K_FIXED})",
     "Total_s": tB_total, "CF_build_s": np.nan, "KMeans_fit_s": np.nan},

    {"Method": f"C) KMeans trực tiếp (k={K_FIXED})",
     "Total_s": tC_total, "CF_build_s": np.nan, "KMeans_fit_s": tC_total},
]
time_df = pd.DataFrame(time_rows)
time_df.to_csv(OUT_TIME_CSV, index=False)

print("\n=== Thời gian chạy (giây) ===")
print(time_df.round(4))

plt.figure(figsize=(9,4))
sns.barplot(x="Method", y="Total_s", data=time_df)
plt.title("Thời gian tổng mỗi phương án (giây)")
plt.ylabel("Seconds"); plt.xlabel("")
plt.xticks(rotation=10, ha="right"); plt.tight_layout()
plt.savefig(OUT_DIR / "runtime_total.png", dpi=140, bbox_inches="tight")
plt.show()

print("\n✅ Hoàn tất.")
print(f"- Thư mục kết quả: {OUT_DIR.resolve()}")
