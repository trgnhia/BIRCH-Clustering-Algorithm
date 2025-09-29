# -*- coding: utf-8 -*-


import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score

# ========= Config =========
DEFAULT_PRIMARY = "cleaned_dataset.csv"
DEFAULT_FALLBACK = "marketing_campaign.csv"

BIRCH_THRESHOLD = 0.35                 # try 0.30–0.50 if you want
K_RANGE = range(2, 7)                  # try k=2..6
JITTER_STD = 0.03                      # jitter for PCA scatter to reduce overplotting for binary features
RANDOM_STATE = 42

REQUIRED_CANONICAL = [
    "AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
    "AcceptedCmp4","AcceptedCmp5","Response","Complain"
]

# -------- helpers ----------
def detect_input_path():
    # 1) CLI arg, 2) DEFAULT_PRIMARY if exists, 3) DEFAULT_FALLBACK if exists
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    if os.path.exists(DEFAULT_PRIMARY):
        return DEFAULT_PRIMARY
    if os.path.exists(DEFAULT_FALLBACK):
        return DEFAULT_FALLBACK
    # last resort: whatever user passed (even if doesn't exist) for clearer error
    return sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_PRIMARY

def read_csv_safely(path):
    """Try to auto-detect delimiter; if single column, force sep=';'."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=';')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_column_map(cols):
    """
    Create a mapping from various possible header variants to canonical column names.
    Variants tolerated: case-insensitive, extra spaces, underscores, minor spacing in 'AcceptedCmp 1', etc.
    """
    variants = {}
    # produce normalized key for each provided col
    def norm(s):
        return (s.lower()
                  .replace(" ", "")
                  .replace("_", "")
                  .replace("-", ""))
    inv = {norm(c): c for c in cols}

    # generate candidate normalized keys for each canonical
    for i in range(1, 6):
        canon = f"AcceptedCmp{i}"
        keys = {norm(canon), norm(f"Accepted Cmp {i}"), norm(f"AcceptedCmp {i}"), norm(f"acceptedcmp{i}")}
        for k in keys:
            if k in inv and canon not in variants:
                variants[canon] = inv[k]

    for canon in ["Response", "Complain"]:
        keys = {norm(canon), norm(canon + "s"), norm(canon + "_flag")}
        for k in keys:
            if k in inv and canon not in variants:
                variants[canon] = inv[k]
    return variants

def ensure_columns(df):
    colmap = build_column_map(df.columns)
    missing = [c for c in REQUIRED_CANONICAL if c not in colmap]
    if missing:
        raise ValueError(f"❌ Missing required columns (after normalization): {missing}\n"
                         f"Available columns: {list(df.columns)}")
    return colmap

def to_binary01(s):
    # coerce to numeric, NaN->0, clip to 0/1, cast to int
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    # if values look like booleans/Yes/No, map them
    if x.isna().sum() == 0 and x.dropna().isin([0,1]).all():
        return x.astype(int)
    # try typical 'Yes/No', 'True/False', 'Y/N' strings
    if s.dtype == object:
        m = s.astype(str).str.strip().str.lower()
        mapped = (m.isin(["1","true","yes","y"])).astype(int)
        return mapped
    return x.clip(0,1).astype(int)

def add_jitter(a, std=0.03, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    return a + rng.normal(0.0, std, size=a.shape)

def main():
    input_csv = detect_input_path()
    print(f"== Reading: {input_csv}")
    df = read_csv_safely(input_csv)

    # Ensure required columns (tolerant mapping)
    colmap = ensure_columns(df)

    # Build X with canonical order
    X = pd.DataFrame({
        c: to_binary01(df[colmap[c]])
        for c in REQUIRED_CANONICAL
    })

    # ========= 3) Scale =========
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # ========= 4) BIRCH -> KMeans (auto-k) =========
    birch = Birch(threshold=BIRCH_THRESHOLD, branching_factor=50, n_clusters=None)
    birch.fit(X_scaled)

    leaf_centers = getattr(birch, "subcluster_centers_", None)
    leaf_labels  = getattr(birch, "labels_", None)

    best_k, best_score, best_km = None, -1.0, None
    if leaf_centers is not None and leaf_labels is not None and leaf_centers.shape[0] >= 2:
        for k in K_RANGE:
            if k > leaf_centers.shape[0]:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(leaf_centers)
            sample_labels = km.labels_[leaf_labels]
            if len(np.unique(sample_labels)) < 2:
                continue
            try:
                sc = silhouette_score(X_scaled, sample_labels, metric="euclidean")
            except Exception:
                sc = -1.0
            if sc > best_score:
                best_k, best_score, best_km = k, sc, km

    # Fallback: if too few leaves or something odd
    if best_km is None:
        best_k = 2
        best_km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        final_labels = best_km.labels_
    else:
        best_k = best_k or best_km.n_clusters
        final_labels = best_km.labels_[leaf_labels]

    cluster_col = f"Cluster_Campaign_k{best_k}"

    # ========= 5) Summaries & simple naming =========
    out = X.copy()
    out[cluster_col] = final_labels

    accept_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                   "AcceptedCmp4","AcceptedCmp5","Response"]
    summary = (out.groupby(cluster_col)[accept_cols + ["Complain"]]
               .mean().round(3))

    summary["AcceptanceRate"] = summary[accept_cols].mean(axis=1).round(3)

    # simple rule-based names
    names = {}
    for cid, row in summary.iterrows():
        if row["Complain"] >= 0.2 and row["AcceptanceRate"] < 0.15:
            names[cid] = "Negative (complainers)"
        elif row["AcceptanceRate"] >= 0.4:
            names[cid] = "Responsive"
        elif row["AcceptanceRate"] <= 0.15:
            names[cid] = "Indifferent"
        else:
            names[cid] = "Mixed"

    summary["Cluster_Name"] = summary.index.map(names)

    print("== Best k ==", best_k, "| silhouette =", round(float(best_score), 4))
    print("\n== Cluster sizes ==")
    print(out[cluster_col].value_counts().sort_index().to_string())
    print("\n== Campaign response by cluster (proportions) ==")
    print(summary.to_string())

    # Save summary
    summary_path = "campaign_clusters_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path)
    print(f"\n=> Saved summary to: {summary_path}")

    # ========= 6) PCA(2D) for visualization =========
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_scaled)

    # Jitter (helps when data are 0/1 and PCA produces discretized clouds)
    X_2d_j = X_2d.copy()
    if JITTER_STD and JITTER_STD > 0:
        X_2d_j = add_jitter(X_2d_j, std=JITTER_STD, seed=RANDOM_STATE)

    plt.figure(figsize=(7.5, 5.5))
    scatter = plt.scatter(X_2d_j[:,0], X_2d_j[:,1], s=18, alpha=0.85, c=final_labels)
    plt.title(f"Campaign Response Clusters (BIRCH \u2192 KMeans, k={best_k}) – PCA(2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    # Legend
    handles, _ = scatter.legend_elements(num=best_k)
    legend_labels = [f"{cid}: {names.get(cid, 'Cluster')}" for cid in sorted(set(final_labels))]
    plt.legend(handles, legend_labels, title="Clusters", loc="best", frameon=True)

    plt.tight_layout()
    plot_path = "campaign_clusters_plot.png"
    plt.savefig(plot_path, dpi=160)
    print(f"=> Saved PCA plot to: {plot_path}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
