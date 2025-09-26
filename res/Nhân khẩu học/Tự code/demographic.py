# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# =========================
# 1. Đọc & chuẩn hóa dữ liệu
# =========================
df = pd.read_csv("dataset/data_cleaning/cleaned_dataset.csv")

FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]
X = df[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 2. CF Subcluster class
# =========================
class CFSubcluster:
    def __init__(self, x_scaled=None, raw_features=None):
        if x_scaled is not None:
            self.N = 1
            self.LS = np.array(x_scaled, dtype=float)
            self.SS = np.array(x_scaled, dtype=float)**2
            self.sum_income = raw_features["Income"]
            self.sum_spending = raw_features["Total_Spending"]
            self.sum_children = raw_features["Children"]
            self.sum_age = raw_features["Customer_Age"]
        else:
            self.N = 0
            self.LS = None
            self.SS = None
            self.sum_income = 0
            self.sum_spending = 0
            self.sum_children = 0
            self.sum_age = 0

    def absorb_point(self, x_scaled, raw_features):
        if self.N == 0:
            self.__init__(x_scaled, raw_features)
            return
        self.N += 1
        self.LS += x_scaled
        self.SS += x_scaled**2
        self.sum_income += raw_features["Income"]
        self.sum_spending += raw_features["Total_Spending"]
        self.sum_children += raw_features["Children"]
        self.sum_age += raw_features["Customer_Age"]

    def absorb_cf(self, other):
        if self.N == 0:
            self.N = other.N
            self.LS = other.LS.copy()
            self.SS = other.SS.copy()
            self.sum_income = other.sum_income
            self.sum_spending = other.sum_spending
            self.sum_children = other.sum_children
            self.sum_age = other.sum_age
            return
        self.N += other.N
        self.LS += other.LS
        self.SS += other.SS
        self.sum_income += other.sum_income
        self.sum_spending += other.sum_spending
        self.sum_children += other.sum_children
        self.sum_age += other.sum_age

    def centroid(self):
        return self.LS / self.N

    def radius(self):
        centroid = self.centroid()
        variance = (self.SS / self.N) - centroid**2
        return np.sqrt(np.sum(variance))

    def mean_features(self):
        return {
            "Income_mean": self.sum_income / self.N,
            "Spending_mean": self.sum_spending / self.N,
            "Children_mean": self.sum_children / self.N,
            "Age_mean": self.sum_age / self.N,
        }

# =========================
# 3. Xây dựng CF Leaf (L1)
# =========================
def insert_point(cf_list, x_scaled, raw_features, threshold=0.4):
    if not cf_list:
        cf_list.append(CFSubcluster(x_scaled, raw_features))
        return

    min_dist = float("inf")
    closest_cf = None
    for cf in cf_list:
        dist = np.linalg.norm(x_scaled - cf.centroid())
        if dist < min_dist:
            min_dist = dist
            closest_cf = cf

    temp_cf = CFSubcluster()
    temp_cf.absorb_cf(closest_cf)
    temp_cf.absorb_point(x_scaled, raw_features)

    if temp_cf.radius() <= threshold:
        closest_cf.absorb_point(x_scaled, raw_features)
    else:
        cf_list.append(CFSubcluster(x_scaled, raw_features))

def build_CF_leaf(X_scaled, df, threshold=0.4):
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
# 4. Lọc CF leaf
# =========================
def filter_CF(cf_list, min_cf_size=3):
    radii = [cf.radius() for cf in cf_list]
    cutoff = np.percentile(radii, 90)  # p90
    return [
        cf for cf in cf_list
        if cf.N >= min_cf_size and cf.radius() <= cutoff
    ]

# =========================
# 5. Roll-up bằng Agglomerative
# =========================
def rollup_CF(cf_list, threshold=0.8):
    centroids = np.array([cf.centroid() for cf in cf_list])
    # khoảng cách <= threshold -> coi như cùng cụm
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward"
    )
    labels = agg.fit_predict(centroids)

    super_cf_list = []
    for lbl in np.unique(labels):
        indices = np.where(labels == lbl)[0]
        new_cf = CFSubcluster()
        for i in indices:
            new_cf.absorb_cf(cf_list[i])
        super_cf_list.append(new_cf)
    return super_cf_list

# =========================
# 6. Chạy thử
# =========================
cf_L1 = build_CF_leaf(X_scaled, df, threshold=0.4)
print("Số CF leaf (trước lọc):", len(cf_L1))

cf_L1_filtered = filter_CF(cf_L1, min_cf_size=3)
print("Số CF leaf (sau lọc):", len(cf_L1_filtered))

cf_L2 = rollup_CF(cf_L1_filtered, threshold=0.6)
print("Số super CF (L2):", len(cf_L2))

for idx, cf in enumerate(cf_L2[:5]):
    print(f"\nSuper CF {idx}: N={cf.N}, radius={cf.radius():.3f}")
    print("  Mean features:", cf.mean_features())
