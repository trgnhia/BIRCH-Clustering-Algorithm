# kmeans_purchase_behavior.py
# -*- coding: utf-8 -*-
# KMeans ONLY cho Purchase Behavior
# Columns: NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,
#          NumStorePurchases, NumWebVisitsMonth, Recency

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ========================
# Config
# ========================
INPUT_CSV = "marketing_campaign.csv"   # đổi nếu cần ("cleaned_dataset.csv")
K_RANGE = range(2, 7)                  # thử k=2..6
USE_AUTO_SEP = True

REQUIRED = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]

# Alias nếu file đổi tên cột
ALIAS = {
    # "Num_Web_Purchases": "NumWebPurchases",
    # "Num_Catalog_Purchases": "NumCatalogPurchases",
}

# ========================
# Helpers
# ========================
def read_csv_safely(path, auto_sep=True):
    if auto_sep:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)
        if df.shape[1] == 1:  # thường do delimiter ';'
            for sep in [";", ",", "\t", "|"]:
                try:
                    df_try = pd.read_csv(path, sep=sep)
                    if df_try.shape[1] > 1:
                        df = df_try
                        break
                except Exception:
                    continue
    else:
        df = pd.read_csv(path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def apply_alias(df, alias_map):
    rename_map = {old:new for old,new in alias_map.items() if old in df.columns}
    return df.rename(columns=rename_map) if rename_map else df

def ensure_columns(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print("❌ Thiếu cột bắt buộc:", miss)
        print("Các cột hiện có:", list(df.columns))
        sys.exit(1)

# ========================
# 1) Load & check
# ========================
df = read_csv_safely(INPUT_CSV, auto_sep=USE_AUTO_SEP)
df = apply_alias(df, ALIAS)
ensure_columns(df, REQUIRED)

# ========================
# 2) Build X (only 6 columns) + clean
# ========================
X = df[REQUIRED].copy()
for c in REQUIRED:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X[c] = X[c].fillna(X[c].median())

# ========================
# 3) Scale
# ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 4) KMeans ONLY (auto k theo silhouette)
# ========================
best_k, best_score, best_km = None, -1.0, None
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    if len(np.unique(labels)) < 2:
        continue
    try:
        sc = silhouette_score(X_scaled, labels, metric="euclidean")
    except Exception:
        sc = -1.0
    if sc > best_score:
        best_k, best_score, best_km = k, sc, km

final_labels = best_km.labels_
cluster_col = f"Cluster_Behavior_kmeans_k{best_k}"
df[cluster_col] = final_labels

# ========================
# 5) Summaries
# ========================
print("== Best k (KMeans only) ==", best_k, "| silhouette =", round(best_score, 4))
print("\n== Cluster counts ==")
print(df[cluster_col].value_counts().sort_index().to_string())

print("\n== Cluster means on 6 features (original scale) ==")
means = df.groupby(cluster_col)[REQUIRED].mean().round(2)
print(means.to_string())

# ========================
# 6) PCA(2D) visualization (legend khớp màu)
# ========================
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_scaled)
expl = pca2.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={expl[0]:.3f}, PC2={expl[1]:.3f}, total={expl[:2].sum():.3f}")

plt.figure(figsize=(7,5))

unique_clusters = np.sort(np.unique(final_labels))
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=unique_clusters.min(), vmax=unique_clusters.max())

scatter = plt.scatter(X_2d[:,0], X_2d[:,1],
                      s=18, alpha=0.9, c=final_labels,
                      cmap=cmap, norm=norm)

plt.title(f"Purchase Behavior Clusters (KMeans only, k={best_k}) – PCA(2D)")
plt.xlabel("PC1"); plt.ylabel("PC2")

import matplotlib.lines as mlines
handles, labels = [], []
for cid in unique_clusters:
    color = cmap(norm(cid))
    handles.append(mlines.Line2D([0],[0], marker='o', linestyle='',
                                 markerfacecolor=color, markeredgecolor='none',
                                 markersize=8))
    labels.append(f"Cluster {cid}")
plt.legend(handles, labels, title="Clusters", loc="best", fontsize=9)

plt.tight_layout()
plt.show()
