# -*- coding: utf-8 -*-
import warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"
FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]

BIRCH_THRESHOLD = 0.4
K_FINAL = 6
RANDOM_STATE = 42

# =========================
# TIỆN ÍCH
# =========================
def evaluate_clustering(X, labels):
    """Trả về (silhouette, davies-bouldin)."""
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    s = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return round(s, 3), round(db, 3)

# =========================
# 1) LOAD DATA & SCALER
# =========================
df = pd.read_csv(INPUT_CSV)
X = df[FEATURES].copy()

# xử lý log transform cho Income & Total_Spending
X["Income"] = np.log1p(X["Income"].clip(lower=0))
X["Total_Spending"] = np.log1p(X["Total_Spending"].clip(lower=0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

# =========================
# 2) Birch + KMeans
# =========================
start = time.perf_counter()
birch_model = Birch(threshold=BIRCH_THRESHOLD, n_clusters=None)
cf_labels = birch_model.fit_predict(X_scaled)

# dùng centroid Birch cho KMeans
cf_centroids = []
cf_weights = []
for lab in np.unique(cf_labels):
    idx = np.where(cf_labels == lab)[0]
    cf_centroids.append(X_scaled[idx].mean(axis=0))
    cf_weights.append(len(idx))
cf_centroids = np.vstack(cf_centroids)
cf_weights = np.array(cf_weights)

kmeans_birch = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=20)
kmeans_birch.fit(cf_centroids, sample_weight=cf_weights)
final_labels = kmeans_birch.predict(X_scaled)

t = time.perf_counter() - start
sil, db = evaluate_clustering(X_scaled, final_labels)
results.append(["Birch + KMeans", t, sil, db])

# =========================
# 3) Birch trực tiếp (n_clusters = K_FINAL)
# =========================
start = time.perf_counter()
birch_direct = Birch(threshold=BIRCH_THRESHOLD, n_clusters=K_FINAL)
labels_birch = birch_direct.fit_predict(X_scaled)
t = time.perf_counter() - start
sil, db = evaluate_clustering(X_scaled, labels_birch)
results.append(["Birch (direct)", t, sil, db])

# =========================
# 4) KMeans trực tiếp
# =========================
start = time.perf_counter()
kmeans_direct = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=20)
labels_kmeans = kmeans_direct.fit_predict(X_scaled)
t = time.perf_counter() - start
sil, db = evaluate_clustering(X_scaled, labels_kmeans)
results.append(["KMeans (direct)", t, sil, db])

# =========================
# 5) SO SÁNH
# =========================
res_df = pd.DataFrame(results, columns=["Phương pháp", "Thời gian (s)", "Silhouette", "DaviesBouldin"])
print("\n=== KẾT QUẢ SO SÁNH ===")
print(res_df)
