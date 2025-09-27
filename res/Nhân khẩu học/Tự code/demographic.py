# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# CẤU HÌNH HẰNG SỐ
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"

BIRCH_THRESHOLD   = 0.40   # Ngưỡng radius để tạo CF leaf (L1)
ROLLUP_THRESHOLD  = 0.60   # Ngưỡng roll-up cho L2
MIN_CF_SIZE       = 3      # Loại bỏ CF có < 3 điểm
MAX_CF_RADIUS_Q   = 0.90   # Giữ CF có radius <= p90
RANDOM_STATE      = 42

# =========================
# 1. Load & Chuẩn hóa dữ liệu
# =========================
df = pd.read_csv(INPUT_CSV)
FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]
X = df[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================= 
# 2. CF Subcluster Class
# =========================
class CFSubcluster:
    def __init__(self, x_scaled=None, raw_features=None):
        if x_scaled is not None:
            self.N = 1
            self.LS = np.array(x_scaled, dtype=float)
            self.SS = np.array(x_scaled, dtype=float)**2
            self.sum_income   = raw_features["Income"]
            self.sum_spending = raw_features["Total_Spending"]
            self.sum_children = raw_features["Children"]
            self.sum_age      = raw_features["Customer_Age"]
        else:
            self.N = 0
            self.LS = None
            self.SS = None
            self.sum_income   = 0
            self.sum_spending = 0
            self.sum_children = 0
            self.sum_age      = 0

    def absorb_point(self, x_scaled, raw_features):
        """Thêm 1 điểm mới vào CF"""
        if self.N == 0:
            self.__init__(x_scaled, raw_features)
            return
        self.N += 1
        self.LS += x_scaled
        self.SS += x_scaled**2
        self.sum_income   += raw_features["Income"]
        self.sum_spending += raw_features["Total_Spending"]
        self.sum_children += raw_features["Children"]
        self.sum_age      += raw_features["Customer_Age"]
     # tính chất gộp cụm Cf3 =  Cf1 + Cf2
    def absorb_cf(self, other):
        """Gộp 1 CF khác"""
        if self.N == 0:
            self.N = other.N
            self.LS = other.LS.copy()
            self.SS = other.SS.copy()
            self.sum_income   = other.sum_income
            self.sum_spending = other.sum_spending
            self.sum_children = other.sum_children
            self.sum_age      = other.sum_age
            return
        self.N += other.N
        self.LS += other.LS
        self.SS += other.SS
        self.sum_income   += other.sum_income
        self.sum_spending += other.sum_spending
        self.sum_children += other.sum_children
        self.sum_age      += other.sum_age

    def centroid(self):
        return self.LS / self.N

    def radius(self):
        centroid = self.centroid() 
        variance = (self.SS / self.N) - centroid**2
        return np.sqrt(np.sum(variance))
        # R = căn ((ss/n) - (ls/n)^2) -> trả về 1 số
    # tính trung bình các trường thuộc tính
    def mean_features(self):
        return {
            "Income_mean":   self.sum_income / self.N,
            "Spending_mean": self.sum_spending / self.N,
            "Children_mean": self.sum_children / self.N,
            "Age_mean":      self.sum_age / self.N,
        }

# =========================
# 3. Build CF Leaf (L1)
# =========================
def insert_point(cf_list, x_scaled, raw_features, threshold=BIRCH_THRESHOLD):
    if not cf_list:
        # điểm đầu tiên -> CF đầu tiên
        cf_list.append(CFSubcluster(x_scaled, raw_features))
        return

    # tìm CF gần nhất
    min_dist = float("inf")
    closest_cf = None
    for cf in cf_list:
        dist = np.linalg.norm(x_scaled - cf.centroid())
        if dist < min_dist:
            min_dist = dist
            closest_cf = cf

    # nếu gần hơn threshold thì gộp
    if min_dist <= threshold:
        closest_cf.absorb_point(x_scaled, raw_features)
    else:
        # nếu xa hơn threshold -> tạo CF mới
        cf_list.append(CFSubcluster(x_scaled, raw_features))


def build_CF_leaf(X_scaled, df, threshold=BIRCH_THRESHOLD):
    cf_list = []
    for i, row in df.iterrows():
        x_scaled = X_scaled[i]
        raw_features = {
            "Income": row["Income"],
            "Total_Spending": row["Total_Spending"],
            "Children": row["Children"],
            "Customer_Age": row["Customer_Age"]
        }
        insert_point(cf_list, x_scaled, raw_features, threshold)
    return cf_list


# =========================
# 4. Lọc CF Leaf
# =========================
def filter_CF(cf_list, min_cf_size=MIN_CF_SIZE, q=MAX_CF_RADIUS_Q):
    radii = [cf.radius() for cf in cf_list]
    cutoff = np.percentile(radii, q*100)
    return [
        cf for cf in cf_list
        if cf.N >= min_cf_size and cf.radius() <= cutoff
    ]

# =========================
# 5. Roll-up CF (Tự cài đặt)
# =========================
def rollup_CF(cf_list, threshold=ROLLUP_THRESHOLD):
    super_cf_list = []
    for cf in cf_list:
        if not super_cf_list:
            new_cf = CFSubcluster()
            new_cf.absorb_cf(cf)
            super_cf_list.append(new_cf)
            continue

        # tìm super CF gần nhất
        min_dist = float("inf")
        closest_cf = None
        for super_cf in super_cf_list:
            dist = np.linalg.norm(cf.centroid() - super_cf.centroid())
            if dist < min_dist:
                min_dist = dist
                closest_cf = super_cf

        # thử gộp
        temp_cf = CFSubcluster()
        temp_cf.absorb_cf(closest_cf)
        temp_cf.absorb_cf(cf)

        if temp_cf.radius() <= threshold:
            closest_cf.absorb_cf(cf)
        else:
            new_cf = CFSubcluster()
            new_cf.absorb_cf(cf)
            super_cf_list.append(new_cf)
    return super_cf_list

# =========================
# 6. Chạy thử
# =========================
cf_L1 = build_CF_leaf(X_scaled, df, threshold=BIRCH_THRESHOLD)
print("Số CF leaf (trước lọc):", len(cf_L1))

cf_L1_filtered = filter_CF(cf_L1, min_cf_size=MIN_CF_SIZE, q=MAX_CF_RADIUS_Q)
print("Số CF leaf (sau lọc):", len(cf_L1_filtered))

cf_L2 = rollup_CF(cf_L1_filtered, threshold=ROLLUP_THRESHOLD)
print("Số super CF (L2):", len(cf_L2))

# for idx, cf in enumerate(cf_L2[:5]):
#     print(f"\nSuper CF {idx}: N={cf.N}, radius={cf.radius():.3f}")
#     print("  Mean features:", cf.mean_features())
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# Gán nhãn CF cho từng khách hàng
# =========================
def assign_labels(X_scaled, cf_list):
    labels = []
    for x in X_scaled:
        # tìm CF gần nhất
        min_dist = float("inf")
        best_label = -1
        for idx, cf in enumerate(cf_list):
            dist = np.linalg.norm(x - cf.centroid())
            if dist < min_dist:
                min_dist = dist
                best_label = idx
        labels.append(best_label)
    return np.array(labels)

labels_L1 = assign_labels(X_scaled, cf_L1_filtered)
labels_L2 = assign_labels(X_scaled, cf_L2)

# =========================
# Vẽ scatter PCA 2D
# =========================
def plot_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab20", s=20, alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    plt.title(title + f" — PC1 {pca.explained_variance_ratio_[0]*100:.1f}% | PC2 {pca.explained_variance_ratio_[1]*100:.1f}%")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# =========================
# Vẽ CF L1 và CF L2
# =========================
plot_scatter(X_scaled, labels_L1, "BIRCH Micro-Clusters (CF L1, Demographics)")
plot_scatter(X_scaled, labels_L2, "BIRCH Super-Clusters (CF L2, Demographics)")


import math
import seaborn as sns

# =========================
# Chuẩn hóa CF -> DataFrame
# =========================
def cf_to_dataframe(cf_list, label_col="CF_Label"):
    rows = []
    for idx, cf in enumerate(cf_list):
        row = {"N": cf.N, label_col: idx}
        row.update(cf.mean_features())
        rows.append(row)
    return pd.DataFrame(rows)

# =========================
# Vẽ heatmap (phân trang)
# =========================
def plot_cf_heatmap(cf_list, features, title="CF Heatmap", page_size=20, label_col="CF_Label"):
    df_cf = cf_to_dataframe(cf_list, label_col=label_col)
    cf_no_extra = df_cf.set_index(label_col)[features]
    cf_counts = df_cf.set_index(label_col)["N"]

    n_pages = math.ceil(len(cf_no_extra) / max(page_size, 1))
    for i in range(n_pages):
        start, end = i * page_size, min((i + 1) * page_size, len(cf_no_extra))
        sl = cf_no_extra.iloc[start:end].copy()
        ylabels = [f"{idx} (N={int(cf_counts.loc[idx])})" for idx in sl.index]

        plt.figure(figsize=(12,6))
        ax = sns.heatmap(sl, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"size":8})
        ax.set_title(f"{title} (Trang {i+1}/{n_pages})")
        ax.set_xlabel("Feature"); ax.set_ylabel(label_col)
        ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        plt.show()
# # Với CF L1
# plot_cf_heatmap(cf_L1, ["Income_mean", "Spending_mean", "Children_mean", "Age_mean"],
#                 title="CF L1 trung bình", page_size=20, label_col="CF_Label")

# # Với CF L1
# plot_cf_heatmap(cf_L1_filtered, ["Income_mean", "Spending_mean", "Children_mean", "Age_mean"],
#                 title="CF L1 trung bình", page_size=20, label_col="CF_Label")

# # Với CF L2
# plot_cf_heatmap(cf_L2, ["Income_mean", "Spending_mean", "Children_mean", "Age_mean"],
#                 title="CF L2 trung bình", page_size=20, label_col="CF_Label")   