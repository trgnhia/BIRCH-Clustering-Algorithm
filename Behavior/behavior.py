# birch_kmeans_purchase_behavior.py
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score

# ========================
# Config
# ========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"      # dùng file đã tiền xử lý
BIRCH_THRESHOLD = 0.40                 # 0.35 ~ 0.45 tùy dữ liệu
K_RANGE = range(3, 7)                  # thử k = 3..6
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
        if df.shape[1] == 1:
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
    renamed = {}
    for old, new in alias_map.items():
        if old in df.columns:
            renamed[old] = new
    if renamed:
        df = df.rename(columns=renamed)
    return df

def ensure_columns(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print("❌ Thiếu cột bắt buộc:", miss)
        print("Các cột hiện có:", list(df.columns))
        sys.exit(1)

def auto_segment_names(cluster_means):
    """
    Nhận vào DataFrame mean theo cụm trên 6 cột REQUIRED.
    Trả về dict: {cluster_id: segment_name}.
    """
    means = cluster_means.copy()
    z = (means - means.mean()) / (means.std(ddof=0) + 1e-9)

    names = {}
    for cid, row in z.iterrows():
        web_p = row["NumWebPurchases"]
        web_v = row["NumWebVisitsMonth"]
        store_p = row["NumStorePurchases"]
        catalog_p = row["NumCatalogPurchases"]
        deals_p = row["NumDealsPurchases"]
        recency = row["Recency"]  # z>0 => ít mua gần đây

        if (web_p > 0.5 and web_v > 0.5) and (store_p < 0.2) and (catalog_p <= 0.2):
            names[cid] = "Digital Shoppers"
        elif (store_p > 0.6) and (web_v < 0.0) and (web_p < 0.0):
            names[cid] = "In-store Loyal"
        elif deals_p > 0.6:
            names[cid] = "Deal Seekers"
        elif (web_v < -0.2) and (web_p < -0.2) and (store_p < -0.2) and (recency > 0.4):
            names[cid] = "Dormant Customers"
        else:
            dominant = {
                "Web": web_p + web_v,
                "Store": store_p,
                "Catalog": catalog_p,
                "Deals": deals_p
            }
            top = max(dominant, key=dominant.get)
            fallback = {
                "Web": "Digital-leaning",
                "Store": "In-store-leaning",
                "Catalog": "Catalog-leaning",
                "Deals": "Deal-leaning"
            }[top]
            names[cid] = fallback
    return names

# ---- Heatmap helpers (Age groups) ----
def make_age_group(df, age_col='Age', year_birth_col='Year_Birth', ref_year=2014):
    """Tạo cột Age_Group từ Age hoặc Year_Birth."""
    if age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors='coerce')
    elif year_birth_col in df.columns:
        age = ref_year - pd.to_numeric(df[year_birth_col], errors='coerce')
    else:
        raise ValueError("Không tìm thấy 'Age' hay 'Year_Birth' để tạo nhóm tuổi.")

    bins = [-np.inf, 30, 40, 50, 60, np.inf]
    labels = ["<30 years old", "30-39 years old", "40-49 years old",
              "50-59 years old", "60+ years old"]
    return pd.cut(age, bins=bins, labels=labels)

def plot_age_spending_heatmap(df):
    """Vẽ heatmap chi tiêu trung bình theo nhóm tuổi (Mnt*)."""
    prod_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                 "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    prod_cols = [c for c in prod_cols if c in df.columns]
    if not prod_cols:
        print("⚠️ Không tìm thấy các cột chi tiêu Mnt* để vẽ heatmap.")
        return

    df2 = df.copy()
    df2["Age_Group"] = make_age_group(df2)

    pivot = df2.groupby("Age_Group")[prod_cols].mean().round(2)
    data = pivot.values

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(data, aspect='auto')

    ax.set_xticks(range(len(prod_cols)))
    ax.set_xticklabels(prod_cols, rotation=20, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")

    ax.set_title("AVERAGE PRODUCT SPENDING BY AGE GROUP")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

# ========================
# 1) Load & check
# ========================
df = read_csv_safely(INPUT_CSV, auto_sep=USE_AUTO_SEP)
df = apply_alias(df, ALIAS)
ensure_columns(df, REQUIRED)

# ========================
# 2) Build X with only 6 columns
# ========================
X = df[REQUIRED].copy()

# ép numeric + fill missing
for c in REQUIRED:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    med = X[c].median()
    X[c] = X[c].fillna(med)

# ========================
# 3) Scale
# ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 4) BIRCH -> KMeans (auto k)
# ========================
birch = Birch(threshold=BIRCH_THRESHOLD, branching_factor=50, n_clusters=None)
birch.fit(X_scaled)

leaf_centers = birch.subcluster_centers_
leaf_labels  = birch.labels_

best_k, best_score, best_km = None, -1.0, None
if leaf_centers is not None and leaf_centers.shape[0] >= 3:
    for k in K_RANGE:
        if k > leaf_centers.shape[0]:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(leaf_centers)
        sample_labels = km.labels_[leaf_labels]
        if len(np.unique(sample_labels)) < 2:
            continue
        try:
            sc = silhouette_score(X_scaled, sample_labels, metric="euclidean")
        except Exception:
            sc = -1.0
        if sc > best_score:
            best_k, best_score, best_km = k, sc, km

# Fallback
if best_km is None:
    best_k = 3
    best_km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_scaled)
    final_labels = best_km.labels_
else:
    final_labels = best_km.labels_[leaf_labels]

cluster_col = f"Cluster_Behavior_k{best_k}"
df[cluster_col] = final_labels

# ========================
# 5) Summaries + Segment names
# ========================
print("== Chọn k tốt nhất ==", best_k, "| silhouette =", round(best_score, 4))

print("\n== Cluster counts ==")
print(df[cluster_col].value_counts().sort_index().to_string())

print("\n== Cluster means on 6 features (original scale) ==")
means = df.groupby(cluster_col)[REQUIRED].mean().round(2)
print(means.to_string())

segment_names = auto_segment_names(means)
print("\n== Segment names (auto) ==")
for cid in sorted(segment_names):
    print(f"Cluster {cid}: {segment_names[cid]}")

df["SegmentName"] = df[cluster_col].map(segment_names)

print("\n== A few labeled rows ==")
print(df[[cluster_col, "SegmentName"] + REQUIRED].head(10).to_string(index=False))

# ========================
# 6) PCA(2D) + legend chuẩn màu
# ========================
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))

# Dùng cmap + norm để legend khớp màu tuyệt đối
unique_clusters = np.sort(np.unique(final_labels))
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=unique_clusters.min(), vmax=unique_clusters.max())

scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      s=18, alpha=0.9, c=final_labels,
                      cmap=cmap, norm=norm)

plt.title(f"Purchase Behavior Clusters (BIRCH → KMeans, k={best_k}) – PCA(2D)")
plt.xlabel("PC1"); plt.ylabel("PC2")

import matplotlib.lines as mlines
handles = []
labels = []
for cid in unique_clusters:
    color = cmap(norm(cid))
    handles.append(mlines.Line2D([0], [0], marker='o', linestyle='',
                                 markerfacecolor=color, markeredgecolor='none',
                                 markersize=8))
    nice = segment_names.get(cid, "")
    labels.append(f"Cluster {cid}" + (f" – {nice}" if nice else ""))

plt.legend(handles, labels, title="Clusters", loc="best", fontsize=8)
plt.tight_layout()
plt.show()

# ========================
# 7) Heatmap chi tiêu theo nhóm tuổi
# ========================
plot_age_spending_heatmap(df)

# (Tuỳ chọn) Ghi đè clustered_data.csv:
# df.to_csv("clustered_data.csv", index=False, encoding="utf-8-sig")
