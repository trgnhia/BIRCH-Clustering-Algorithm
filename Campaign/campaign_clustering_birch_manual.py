# -*- coding: utf-8 -*-

"""
Manual BIRCH (no external Birch library) + KMeans on subcluster centers.
(See comments inside for explanations.)
"""
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ========= Config =========
DEFAULT_PRIMARY = "cleaned_dataset.csv"
DEFAULT_FALLBACK = "marketing_campaign.csv"

B_THRESHOLD = 0.35          # BIRCH radius threshold T
BRANCHING_FACTOR = 50       # max number of entries per node
K_RANGE = range(2, 7)       # try k=2..6 for KMeans
JITTER_STD = 0.03
RANDOM_STATE = 42

REQUIRED_CANONICAL = [
    "AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
    "AcceptedCmp4","AcceptedCmp5","Response","Complain"
]

# ---------------- CF-Tree (Manual BIRCH) ----------------
class CFEntry:
    """Clustering Feature entry: (N, LS, SS)."""
    def __init__(self, dim):
        self.n = 0
        self.ls = np.zeros(dim, dtype=float)  # linear sum
        self.ss = np.zeros(dim, dtype=float)  # squared sum per-dimension

    @classmethod
    def from_point(cls, x):
        e = cls(len(x))
        e.add_point(x)
        return e

    def copy(self):
        e = CFEntry(len(self.ls))
        e.n = self.n
        e.ls = self.ls.copy()
        e.ss = self.ss.copy()
        return e

    def add_point(self, x):
        self.n += 1
        self.ls += x
        self.ss += x * x

    def add_entry(self, other):
        self.n += other.n
        self.ls += other.ls
        self.ss += other.ss

    @property
    def centroid(self):
        if self.n == 0:
            return self.ls  # zeros
        return self.ls / self.n

    @property
    def radius(self):
        """RMS radius: sqrt(sum(var_i)) where var_i = (SS_i/N) - (LS_i/N)^2"""
        if self.n <= 1:
            return 0.0
        mean_sq = self.ss / self.n
        mean = self.ls / self.n
        var = np.maximum(mean_sq - mean * mean, 0.0)
        return math.sqrt(float(np.sum(var)))

    def merged_radius_with_point(self, x):
        e = self.copy()
        e.add_point(x)
        return e.radius

class CFNode:
    """Node of CF-Tree; can be leaf or internal."""
    def __init__(self, dim, is_leaf=True, parent=None):
        self.dim = dim
        self.is_leaf = is_leaf
        self.parent = parent
        self.entries = []      # list[CFEntry]
        self.children = []     # list[CFNode] for internal nodes
        self.next_leaf = None  # linked list for leaves (optional)

    @property
    def n_entries(self):
        return len(self.entries)

class CFTree:
    def __init__(self, threshold, branching_factor, dim):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.dim = dim
        self.root = CFNode(dim, is_leaf=True, parent=None)
        self.leaf_nodes = [self.root]

    @staticmethod
    def _euclid2(a, b):
        d = a - b
        return float(np.dot(d, d))

    def _choose_leaf(self, node, x):
        if node.is_leaf:
            return node
        centroids = [e.centroid for e in node.entries]
        j = int(np.argmin([self._euclid2(c, x) for c in centroids]))
        return self._choose_leaf(node.children[j], x)

    def insert(self, x):
        node = self._choose_leaf(self.root, x)
        if node.n_entries == 0:
            node.entries.append(CFEntry.from_point(x))
        else:
            centroids = [e.centroid for e in node.entries]
            dists = [self._euclid2(c, x) for c in centroids]
            j = int(np.argmin(dists))
            e = node.entries[j]
            if e.merged_radius_with_point(x) <= self.threshold:
                e.add_point(x)
            else:
                node.entries.append(CFEntry.from_point(x))
        if node.n_entries > self.branching_factor:
            self._split_node(node)

    def _aggregate_entries(self, entries):
        agg = CFEntry(self.dim)
        for e in entries:
            agg.add_entry(e)
        return agg

    def _split_node(self, node):
        centroids = [e.centroid for e in node.entries]
        max_d = -1.0; seed_i = 0; seed_j = 1
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                d = self._euclid2(centroids[i], centroids[j])
                if d > max_d:
                    max_d = d; seed_i = i; seed_j = j
        groupA = [node.entries[seed_i]]
        groupB = [node.entries[seed_j]]
        for idx, e in enumerate(node.entries):
            if idx in (seed_i, seed_j): 
                continue
            dA = self._euclid2(groupA[0].centroid, e.centroid)
            dB = self._euclid2(groupB[0].centroid, e.centroid)
            if dA <= dB:
                groupA.append(e)
            else:
                groupB.append(e)

        left = CFNode(self.dim, is_leaf=node.is_leaf, parent=node.parent)
        right = CFNode(self.dim, is_leaf=node.is_leaf, parent=node.parent)
        for e in groupA: left.entries.append(e)
        for e in groupB: right.entries.append(e)

        # maintain leaf list
        if node.is_leaf:
            if node in self.leaf_nodes:
                i = self.leaf_nodes.index(node)
                self.leaf_nodes[i:i+1] = [left, right]
            else:
                self.leaf_nodes.extend([left, right])

        if node.parent is None:
            new_root = CFNode(self.dim, is_leaf=False, parent=None)
            new_root.children = [left, right]
            new_root.entries = [self._aggregate_entries(left.entries), self._aggregate_entries(right.entries)]
            left.parent = new_root; right.parent = new_root
            self.root = new_root
        else:
            parent = node.parent
            idx_old = parent.children.index(node)
            parent.children.pop(idx_old)
            parent.entries.pop(idx_old)
            parent.children.extend([left, right])
            parent.entries.extend([self._aggregate_entries(left.entries), self._aggregate_entries(right.entries)])
            left.parent = parent; right.parent = parent
            if parent.n_entries > self.branching_factor:
                self._split_node(parent)

    def extract_leaf_centers(self):
        centers = []
        for leaf in self.leaf_nodes:
            for e in leaf.entries:
                centers.append(e.centroid)
        return np.array(centers) if centers else np.empty((0, self.dim))

    def assign_leaf_labels(self, X):
        centers = self.extract_leaf_centers()
        if centers.shape[0] == 0:
            return np.zeros(len(X), dtype=int), centers
        labels = []
        for x in X:
            dists = np.sum((centers - x)**2, axis=1)
            labels.append(int(np.argmin(dists)))
        return np.array(labels, dtype=int), centers

# -------- data utils --------
def detect_input_path():
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    if os.path.exists(DEFAULT_PRIMARY):
        return DEFAULT_PRIMARY
    if os.path.exists(DEFAULT_FALLBACK):
        return DEFAULT_FALLBACK
    return sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_PRIMARY

def read_csv_safely(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=';')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_column_map(cols):
    def norm(s):
        return (s.lower().replace(" ", "").replace("_", "").replace("-", ""))
    inv = {norm(c): c for c in cols}
    variants = {}
    for i in range(1, 6):
        canon = f"AcceptedCmp{i}"
        for k in {norm(canon), norm(f"Accepted Cmp {i}"), norm(f"AcceptedCmp {i}"), norm(f"acceptedcmp{i}")}:
            if k in inv and canon not in variants:
                variants[canon] = inv[k]
    for canon in ["Response","Complain"]:
        for k in {norm(canon), norm(canon+'s'), norm(canon+'_flag')}:
            if k in inv and canon not in variants:
                variants[canon] = inv[k]
    return variants

def ensure_columns(df):
    colmap = build_column_map(df.columns)
    missing = [c for c in REQUIRED_CANONICAL if c not in colmap]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}\nAvailable: {list(df.columns)}")
    return colmap

def to_binary01(s):
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    if x.isna().sum() == 0 and x.dropna().isin([0,1]).all():
        return x.astype(int)
    if s.dtype == object:
        m = s.astype(str).str.strip().str.lower()
        return (m.isin(["1","true","yes","y"])).astype(int)
    return x.clip(0,1).astype(int)

def add_jitter(a, std=0.03, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    return a + rng.normal(0.0, std, size=a.shape)

def main():
    input_csv = detect_input_path()
    print(f"== Reading: {input_csv}")
    df = read_csv_safely(input_csv)
    colmap = ensure_columns(df)
    X = pd.DataFrame({c: to_binary01(df[colmap[c]]) for c in REQUIRED_CANONICAL})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # ---- Build manual BIRCH tree ----
    dim = X_scaled.shape[1]
    tree = CFTree(threshold=B_THRESHOLD, branching_factor=BRANCHING_FACTOR, dim=dim)
    for i in range(X_scaled.shape[0]):
        tree.insert(X_scaled[i, :])

    leaf_labels, leaf_centers = tree.assign_leaf_labels(X_scaled)

    # ---- KMeans on leaf centers (auto-k) ----
    best_k, best_score, best_km = None, -1.0, None
    if leaf_centers is not None and leaf_centers.shape[0] >= 2:
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

    if best_km is None:
        best_k = 2
        best_km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        final_labels = best_km.labels_
    else:
        final_labels = best_km.labels_[leaf_labels]

    cluster_col = f"Cluster_Campaign_k{best_k}"

    accept_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]
    out = X.copy()
    out[cluster_col] = final_labels

    summary = (out.groupby(cluster_col)[accept_cols + ["Complain"]].mean().round(3))
    summary["AcceptanceRate"] = summary[accept_cols].mean(axis=1).round(3)

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

    summary_path = "campaign_clusters_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path)
    print(f"\n=> Saved summary to: {summary_path}")

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_scaled)
    X_2d_j = add_jitter(X_2d, std=JITTER_STD, seed=RANDOM_STATE) if JITTER_STD>0 else X_2d

    plt.figure(figsize=(7.5, 5.5))
    scatter = plt.scatter(X_2d_j[:,0], X_2d_j[:,1], s=18, alpha=0.85, c=final_labels)
    plt.title(f"Campaign Response Clusters (Manual BIRCH → KMeans, k={best_k}) – PCA(2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
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
