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
X = df[FEATURES].copy()

X["Income"] = np.log1p(X["Income"].clip(lower=0))
X["Total_Spending"] = np.log1p(X["Total_Spending"].clip(lower=0))

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
    def simulate_radius_after_absorb(self, x_scaled):
        """Tính bán kính giả định nếu thêm 1 điểm mới vào CF"""
        if self.N == 0:
            return 0.0
        
        N_new = self.N + 1
        LS_new = self.LS + x_scaled
        SS_new = self.SS + x_scaled**2
        centroid_new = LS_new / N_new
        variance_new = (SS_new / N_new) - centroid_new**2
        radius_new = np.sqrt(np.sum(variance_new))
        return radius_new

# =========================
# 3. Build CF Leaf (L1)
# =========================
def insert_point(cf_list, x_scaled, raw_features, threshold=BIRCH_THRESHOLD):
    if not cf_list:
        cf_list.append(CFSubcluster(x_scaled, raw_features))
        return

    # Tìm CF gần nhất
    min_dist = float("inf")
    closest_cf = None
    for cf in cf_list:
        dist = np.linalg.norm(x_scaled - cf.centroid())
        if dist < min_dist:
            min_dist = dist
            closest_cf = cf

    # Kiểm tra bán kính giả định nếu thêm điểm vào CF đó
    new_radius = closest_cf.simulate_radius_after_absorb(x_scaled)

    if new_radius <= threshold:
        # Nếu vẫn trong giới hạn → gộp vào CF đó
        closest_cf.absorb_point(x_scaled, raw_features)
    else:
        # Nếu vượt quá → tạo CF mới
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

'''
# =========================
# 7) KMEANS TRÊN CF L2 + HIỂN THỊ Z-SCORE
# =========================
from sklearn.cluster import KMeans

# Tạo DataFrame từ CF L2
df_cf_l2 = cf_to_dataframe(cf_L2, label_col="CF_Label")

# Chọn các feature trung bình để chạy KMeans
FEATURES_KM = ["Age_mean", "Children_mean", "Income_mean", "Spending_mean"]
X_cf_l2 = df_cf_l2[FEATURES_KM].values

# Chạy KMeans (ví dụ K = 6)
K = 6
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
df_cf_l2["KMeans_Label"] = kmeans.fit_predict(X_cf_l2)

print(f"\nSố cụm KMeans (K={K}):", df_cf_l2["KMeans_Label"].nunique())

# =========================
# Tính Z-score cho từng cụm
# =========================
cluster_means  = df_cf_l2.groupby("KMeans_Label")[FEATURES_KM].mean().sort_index()
cluster_median = df_cf_l2.groupby("KMeans_Label")[FEATURES_KM].median().sort_index()
cluster_std    = df_cf_l2.groupby("KMeans_Label")[FEATURES_KM].std().sort_index()
cluster_count  = df_cf_l2["KMeans_Label"].value_counts().sort_index().rename("Count")

global_mean = df_cf_l2[FEATURES_KM].mean()
global_std  = df_cf_l2[FEATURES_KM].std().replace(0, np.nan)

# Z-score = (mean - global_mean) / global_std
z_table = (cluster_means - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

# =========================
# Hiển thị kết quả Z-score
# =========================
print("\n===== Z-SCORE THEO CỤM KMEANS (CF L2) =====")
print(z_table.round(2))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(z_table.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title(f"Z-score trung bình theo cụm KMeans (CF L2, K={K})")
plt.xlabel("Cluster"); plt.ylabel("Feature")
plt.tight_layout()
plt.show()'''


# =========================
# 8) GÁN NHÃN KMEANS CHO TỪNG KHÁCH HÀNG (DỰA TRÊN CF L2)
# =========================
from sklearn.cluster import KMeans

# Tạo DataFrame từ CF L2
df_cf_l2 = cf_to_dataframe(cf_L2, label_col="CF_Label")

# Chọn các feature trung bình để chạy KMeans
FEATURES_KM = ["Age_mean", "Children_mean", "Income_mean", "Spending_mean"]
X_cf_l2 = df_cf_l2[FEATURES_KM].values

# Chạy KMeans (ví dụ K = 6)
K = 6
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
df_cf_l2["KMeans_Label"] = kmeans.fit_predict(X_cf_l2)
# 1️⃣ Lấy tâm cụm KMeans (centroids)
kmeans_centroids = kmeans.cluster_centers_

# 2️⃣ Lấy centroid thực của từng CF L2
cf_centroids = np.vstack([cf.centroid() for cf in cf_L2])

# 3️⃣ Gán mỗi CF L2 thuộc cụm nào (đã có trong df_cf_l2)
l2_to_k = dict(zip(range(len(cf_L2)), df_cf_l2["KMeans_Label"]))

# 4️⃣ Gán từng khách hàng thực (X_scaled) vào cụm gần nhất của KMeans
#    Bước 1: tìm CF L2 gần nhất
labels_L2_customer = assign_labels(X_scaled, cf_L2)
#    Bước 2: tra xem CF L2 đó thuộc cụm KMeans nào
labels_customer_kmeans = np.array([l2_to_k[l] for l in labels_L2_customer])

print("\nĐã gán nhãn KMeans cho toàn bộ khách hàng.")
print("Số cụm thực tế:", len(np.unique(labels_customer_kmeans)))

# =========================
# 9) VẼ PCA SCATTER KHÁCH HÀNG THEO CỤM KMEANS
# =========================
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pca_clusters(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=20, alpha=0.7)
    plt.colorbar(sc, label="KMeans Label")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

plot_pca_clusters(X_scaled, labels_customer_kmeans, f"KMeans Clusters dựa trên CF L2 (K={K})")

# =========================
# 10) TÍNH Z-SCORE CHO KHÁCH HÀNG (THEO CỤM KMEANS)
# =========================
df_customer = df.copy()
df_customer["KMeans_Label"] = labels_customer_kmeans

FEATURES_REAL = ["Customer_Age", "Children", "Income", "Total_Spending"]
cluster_means  = df_customer.groupby("KMeans_Label")[FEATURES_REAL].mean().sort_index()
cluster_median = df_customer.groupby("KMeans_Label")[FEATURES_REAL].median().sort_index()
cluster_std    = df_customer.groupby("KMeans_Label")[FEATURES_REAL].std().sort_index()
cluster_count  = df_customer["KMeans_Label"].value_counts().sort_index().rename("Count")

global_mean = df_customer[FEATURES_REAL].mean()
global_std  = df_customer[FEATURES_REAL].std().replace(0, np.nan)

z_table_real = (cluster_means - global_mean) / global_std
z_table_real = z_table_real.replace([np.inf, -np.inf], np.nan)

print("\n===== Z-SCORE KHÁCH HÀNG (GÁN TỪ KMEANS TRÊN CF L2) =====")
print(z_table_real.round(2))

plt.figure(figsize=(10,6))
sns.heatmap(z_table_real.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title(f"Z-score trung bình theo cụm KMeans (Khách hàng, K={K})")
plt.xlabel("Cluster"); plt.ylabel("Feature")
plt.tight_layout(); plt.show()

