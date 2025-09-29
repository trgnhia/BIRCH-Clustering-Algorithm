# birch_manual_kmeans.py
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------
# Configs
# ----------------------
INPUT_CSV = "cleaned_dataset.csv"  # your preprocessed file
USE_AUTO_SEP = True
REQUIRED = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]

BRANCHING_FACTOR = 50     
THRESHOLD = 0.35           
K_RANGE = range(3, 7)      
RANDOM_STATE = 42

# ----------------------
# IO Helpers
# ----------------------
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

def ensure_columns(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print("❌ Missing required columns:", miss)
        print("Available columns:", list(df.columns))
        sys.exit(1)

# ----------------------
# BIRCH: Core data structures
# ----------------------
class CFEntry:
    """Clustering Feature entry: (N, LS, SS)"""
    def __init__(self, dim, init_point=None):
        self.N = 0
        self.LS = np.zeros(dim, dtype=float)
        self.SS = 0.0
        if init_point is not None:
            self.add_point(init_point)

    def add_point(self, x):
        self.N += 1
        self.LS += x
        self.SS += float(np.dot(x, x))

    def merge(self, other: "CFEntry"):
        self.N += other.N
        self.LS += other.LS
        self.SS += other.SS

    @property
    def centroid(self):
        if self.N == 0: 
            return None
        return self.LS / self.N

    @property
    def radius(self):
        if self.N <= 1:
            return 0.0
        mu = self.centroid
        return math.sqrt(max(self.SS / self.N - float(np.dot(mu, mu)), 0.0))

    def distance_to_point(self, x):
        mu = self.centroid
        return float(np.linalg.norm(x - mu))

    def distance_to_entry(self, other: "CFEntry"):
        return float(np.linalg.norm(self.centroid - other.centroid))

class CFNode:
    """A CF-Tree node: internal or leaf."""
    def __init__(self, is_leaf, B, threshold, parent=None):
        self.is_leaf = is_leaf
        self.B = B
        self.threshold = threshold
        self.parent = parent

        self.entries = []   
        self.children = [] 

        self.prev_leaf = None
        self.next_leaf = None

    def add_entry(self, entry: CFEntry, child=None):
        self.entries.append(entry)
        if not self.is_leaf:
            assert child is not None, "Internal node requires a child node."
            self.children.append(child)
            child.parent = self

    def update_parent_entry_reference(self):
        """Ensure parent's summary entry equals sum of this node's entries."""
        if self.parent is None:
            return
        idx = None
        for i, ch in enumerate(self.parent.children):
            if ch is self:
                idx = i; break
        if idx is None:
            return
        summary = CFEntry(dim=self.entries[0].LS.size)
        for e in self.entries:
            summary.merge(e)
        self.parent.entries[idx] = summary

    def nearest_entry_index(self, x):
        """Return index of nearest entry centroid to point x."""
        dists = [e.distance_to_point(x) for e in self.entries]
        return int(np.argmin(dists)) if dists else None

    def is_overflow(self):
        return len(self.entries) > self.B

class CFTree:
    """Minimal CF-Tree with threshold-based absorption and farthest-pair splits."""
    def __init__(self, dim, B=50, threshold=0.5):
        self.dim = dim
        self.B = B
        self.threshold = threshold
        self.root = CFNode(is_leaf=True, B=B, threshold=threshold, parent=None)
        self.first_leaf = self.root  # keep head of leaf chain

    # ---- Split helpers ----
    @staticmethod
    def _farthest_pair(entries):
        """Find two farthest entries (indices) among entries based on centroid distance."""
        m = len(entries)
        max_d = -1.0
        pair = (0, 1) if m >= 2 else (0, 0)
        for i in range(m):
            for j in range(i+1, m):
                d = entries[i].distance_to_entry(entries[j])
                if d > max_d:
                    max_d = d
                    pair = (i, j)
        return pair

    def _split_node(self, node: CFNode):
        """Split a node into two using farthest pair as seeds; reassign entries."""
        entries = node.entries
        if len(entries) <= node.B:
            return  # no need

        i, j = self._farthest_pair(entries)
        seed_a = entries[i]
        seed_b = entries[j]

        # Prepare new nodes of the same type as "node"
        new_node_a = CFNode(node.is_leaf, node.B, node.threshold, parent=node.parent)
        new_node_b = CFNode(node.is_leaf, node.B, node.threshold, parent=node.parent)

        # For leaf chain handling
        if node.is_leaf:
            # Insert new_node_a <-> new_node_b where node was
            new_node_a.prev_leaf = node.prev_leaf
            new_node_b.next_leaf = node.next_leaf
            new_node_a.next_leaf = new_node_b
            new_node_b.prev_leaf = new_node_a
            if node.prev_leaf:
                node.prev_leaf.next_leaf = new_node_a
            if node.next_leaf:
                node.next_leaf.prev_leaf = new_node_b
            if self.first_leaf is node:
                self.first_leaf = new_node_a

        # Reassign each entry to the closer seed
        def closer_to_seed(e):
            da = seed_a.distance_to_entry(e)
            db = seed_b.distance_to_entry(e)
            return 0 if da <= db else 1

        groups = {0: [], 1: []}
        if node.is_leaf:
            for e in entries:
                g = closer_to_seed(e)
                if g == 0: new_node_a.entries.append(e)
                else:      new_node_b.entries.append(e)
        else:
            # internal: keep (entry, child) pairing
            for e, child in zip(node.entries, node.children):
                g = closer_to_seed(e)
                if g == 0:
                    new_node_a.entries.append(e); new_node_a.children.append(child); child.parent = new_node_a
                else:
                    new_node_b.entries.append(e); new_node_b.children.append(child); child.parent = new_node_b

        parent = node.parent
        if parent is None:
            # create new root (internal) with two children
            new_root = CFNode(is_leaf=False, B=node.B, threshold=node.threshold, parent=None)
            # summary entries for children
            sum_a = CFEntry(self.dim)
            for e in new_node_a.entries: sum_a.merge(e)
            sum_b = CFEntry(self.dim)
            for e in new_node_b.entries: sum_b.merge(e)
            new_root.add_entry(sum_a, child=new_node_a)
            new_root.add_entry(sum_b, child=new_node_b)
            self.root = new_root
        else:
            # Replace 'node' with two new nodes in parent
            # Find node idx in parent
            idx = None
            for i, ch in enumerate(parent.children):
                if ch is node:
                    idx = i; break
            assert idx is not None

            # Remove old slot
            parent.entries.pop(idx)
            parent.children.pop(idx)

            # Insert two new children + summary entries
            sum_a = CFEntry(self.dim); [sum_a.merge(e) for e in new_node_a.entries]
            sum_b = CFEntry(self.dim); [sum_b.merge(e) for e in new_node_b.entries]
            parent.entries.append(sum_a); parent.children.append(new_node_a); new_node_a.parent = parent
            parent.entries.append(sum_b); parent.children.append(new_node_b); new_node_b.parent = parent

            # If parent overflows, split it recursively
            if parent.is_overflow():
                self._split_node(parent)

    # ---- Insert ----
    def insert(self, x):
        """Insert a single sample x (1D ndarray)."""
        node = self.root
        # 1) Traverse to a leaf
        while not node.is_leaf:
            # choose nearest child by nearest entry centroid
            idx = node.nearest_entry_index(x)
            node = node.children[idx]

        # 2) At leaf: find nearest entry
        if not node.entries:
            node.entries.append(CFEntry(self.dim, x))
        else:
            idx = node.nearest_entry_index(x)
            nearest = node.entries[idx]
            # Try to absorb
            tmp = CFEntry(self.dim)
            tmp.merge(nearest)    # copy stats
            tmp.add_point(x)
            if tmp.radius <= self.threshold:
                nearest.add_point(x)
            else:
                # Create a new micro-cluster
                node.entries.append(CFEntry(self.dim, x))

        # 3) If leaf overflows => split
        if node.is_overflow():
            self._split_node(node)

        # 4) After changes, update summaries up to root
        self._update_to_root(node)

    def _update_to_root(self, node):
        """Recompute summary CFs from 'node' up to the root."""
        cur = node
        while cur is not None:
            if cur.parent is not None:
                # replace parent's entry corresponding to 'cur' by sum of cur.entries
                summary = CFEntry(self.dim)
                for e in cur.entries:
                    summary.merge(e)
                # locate child index
                for i, ch in enumerate(cur.parent.children):
                    if ch is cur:
                        cur.parent.entries[i] = summary
                        break
                # if parent overflowed due to a previous step, split
                if cur.parent.is_overflow():
                    self._split_node(cur.parent)
            cur = cur.parent

    # ---- Collect leaf subclusters ----
    def get_leaf_entries(self):
        """Return a list of CFEntry from all leaves (micro-clusters)."""
        out = []
        # iterate over leaf chain for efficiency (if available)
        leaf = self.first_leaf
        visited = set()
        while leaf is not None and (id(leaf) not in visited):
            visited.add(id(leaf))
            out.extend(leaf.entries)
            leaf = leaf.next_leaf
        if not out:
            # fallback: DFS from root
            stack = [self.root]
            while stack:
                node = stack.pop()
                if node.is_leaf:
                    out.extend(node.entries)
                else:
                    stack.extend(node.children)
        return out

# ----------------------
# Pipeline
# ----------------------
def main():
    # 1) Load
    df = read_csv_safely(INPUT_CSV, auto_sep=USE_AUTO_SEP)
    ensure_columns(df, REQUIRED)

    X = df[REQUIRED].copy()
    # numeric + fillna
    for c in REQUIRED:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    # 2) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    dim = X_scaled.shape[1]

    # 3) Build CF-Tree (manual BIRCH)
    tree = CFTree(dim=dim, B=BRANCHING_FACTOR, threshold=THRESHOLD)
    for i, x in enumerate(X_scaled):
        tree.insert(x)

    # 4) Collect leaf subclusters
    leaf_entries = tree.get_leaf_entries()
    leaf_centers = np.stack([e.centroid for e in leaf_entries]) if leaf_entries else None
    leaf_labels = None
    if leaf_centers is not None and len(leaf_entries) > 0:
        # Assign each sample to nearest leaf centroid
        from scipy.spatial.distance import cdist
        D = cdist(X_scaled, leaf_centers, metric="euclidean")
        leaf_labels = np.argmin(D, axis=1)

    # 5) KMeans on leaf centroids
    best_k, best_score, best_km = None, -1.0, None
    if leaf_centers is not None and leaf_centers.shape[0] >= 3:
        for k in K_RANGE:
            if k > leaf_centers.shape[0]:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(leaf_centers)
            sample_labels = km.labels_[leaf_labels]  # map leaf -> final
            if len(np.unique(sample_labels)) < 2:
                continue
            try:
                sc = silhouette_score(X_scaled, sample_labels, metric="euclidean")
            except Exception:
                sc = -1.0
            if sc > best_score:
                best_k, best_score, best_km = k, sc, km

    if best_km is None:
        # fallback: run KMeans directly on samples
        best_k = 3
        best_km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        final_labels = best_km.labels_
    else:
        final_labels = best_km.labels_[leaf_labels]

    cluster_col = f"Cluster_Behavior_k{best_k}"
    df[cluster_col] = final_labels

    # 6) Summaries
    print("== Best k ==", best_k, "| silhouette =", round(best_score, 4))
    print("\n== Cluster counts ==")
    print(df[cluster_col].value_counts().sort_index().to_string())

    print("\n== Cluster means on 6 features (original scale) ==")
    means = df.groupby(cluster_col)[REQUIRED].mean().round(2)
    print(means.to_string())

    # 7) PCA Plot
    try:
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d = pca2.fit_transform(X_scaled)
        plt.figure(figsize=(7, 5))
        unique = np.unique(final_labels)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=unique.min(), vmax=unique.max())
        plt.scatter(X_2d[:,0], X_2d[:,1], s=12, alpha=0.85, c=final_labels, cmap=cmap, norm=norm)
        plt.title(f"BIRCH (manual) → KMeans, k={best_k} – PCA(2D)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot error:", e)

    # 8) Save labeled data (optional)
    # df.to_csv("clustered_data_birch_manual.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
