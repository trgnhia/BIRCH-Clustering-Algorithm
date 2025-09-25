# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, KMeans

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_input_dataset.csv"
FEATURES = ["Age", "Children", "Income", "Total_Spent"]

BIRCH_THRESHOLD = 0.3
K_FIXED = 6

APPLY_LOG_TRANSFORM = True
LOG_COLS = ["Income", "Total_Spent"]

RANDOM_STATE = 42

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "output_runtime_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TIME_CSV = OUT_DIR / "runtime_comparison.csv"

# =========================
# TIỀN XỬ LÝ
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
    X_scaled = StandardScaler().fit_transform(X)
    return df, X_scaled

df, X_scaled = load_and_prepare(INPUT_CSV, FEATURES, LOG_COLS, APPLY_LOG_TRANSFORM)

# =========================
# PHƯƠNG ÁN A: Birch(n_clusters=None) -> KMeans weighted
# =========================
tA0 = perf_counter()

# Thời gian tạo CF
tA_cf0 = perf_counter()
birch_A = Birch(threshold=BIRCH_THRESHOLD, n_clusters=None)
birch_A.fit(X_scaled)
cf_build_s = perf_counter() - tA_cf0

# Lấy centroids + counts
cf_labels = birch_A.labels_
cf_centroids = []
cf_counts = []
for lab in np.unique(cf_labels):
    pts = X_scaled[cf_labels == lab]
    cf_centroids.append(pts.mean(axis=0))
    cf_counts.append(len(pts))
cf_centroids = np.vstack(cf_centroids)

# Thời gian KMeans
tA_km0 = perf_counter()
km_A = KMeans(n_clusters=K_FIXED, random_state=RANDOM_STATE, n_init="auto")
km_A.fit(cf_centroids, sample_weight=cf_counts)
kmeans_fit_s = perf_counter() - tA_km0

tA_total = perf_counter() - tA0

# =========================
# PHƯƠNG ÁN B: Birch(n_clusters=K_FIXED)
# =========================
tB0 = perf_counter()
birch_B = Birch(threshold=BIRCH_THRESHOLD, n_clusters=K_FIXED)
birch_B.fit(X_scaled)
tB_total = perf_counter() - tB0

# =========================
# PHƯƠNG ÁN C: KMeans trực tiếp
# =========================
tC0 = perf_counter()
km_C = KMeans(n_clusters=K_FIXED, random_state=RANDOM_STATE, n_init="auto")
km_C.fit(X_scaled)
tC_total = perf_counter() - tC0

# =========================
# KẾT QUẢ
# =========================
time_rows = [
    {"Method": f"A) BirchNone + KMeans(k={K_FIXED})", 
     "Total_s": tA_total, "CF_build_s": cf_build_s, "KMeans_fit_s": kmeans_fit_s},
    {"Method": f"B) Birch(n_clusters={K_FIXED})", 
     "Total_s": tB_total, "CF_build_s": np.nan, "KMeans_fit_s": np.nan},
    {"Method": f"C) KMeans trực tiếp (k={K_FIXED})", 
     "Total_s": tC_total, "CF_build_s": np.nan, "KMeans_fit_s": tC_total},
]
time_df = pd.DataFrame(time_rows)
time_df.to_csv(OUT_TIME_CSV, index=False)

print("\n=== Thời gian chạy (giây) ===")
print(time_df.round(4))
print("\n✅ Hoàn tất.")
print(f"- File kết quả: {OUT_TIME_CSV.resolve()}")
