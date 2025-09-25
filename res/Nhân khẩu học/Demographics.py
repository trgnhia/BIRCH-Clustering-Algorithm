import pandas as pd
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("dataset/data_cleaning/cleaned_input_dataset.csv")

# 2. Chọn các cột nhân khẩu học
features = ["Age", "Marital_Status", "Education", "Children", "Income", "Total_Spent"]
X = df[features]

# 3. Chuẩn hóa dữ liệu (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Khởi tạo và huấn luyện BIRCH
birch_model = Birch(threshold=1.0, n_clusters=None)
birch_model.fit(X_scaled)

# 5. Gán nhãn CF cho từng khách hàng
df["CF_Label"] = birch_model.labels_

# 6. Tính toán các chỉ số CF: N, LS, SS, Centroid, Radius
cf_list = []
for label, group in df.groupby("CF_Label"):
    points = group[features].values
    N = len(points)
    LS = np.sum(points, axis=0)               # Linear Sum
    SS = np.sum(points**2, axis=0)            # Square Sum
    centroid = LS / N                         # Centroid
    radius = np.sqrt(np.sum(SS / N - centroid**2))  # Radius toàn cụm
    
    cf_list.append({
        "CF_Label": label,
        "N": N,
        "LS": LS.tolist(),
        "SS": SS.tolist(),
        "Centroid": centroid.tolist(),
        "Radius": radius
    })

cf_extra_df = pd.DataFrame(cf_list)

# 7. Thống kê đặc trưng trung bình của từng CF (theo dữ liệu gốc)
cf_summary = df.groupby("CF_Label")[features].mean().reset_index()

# 8. Gộp thêm LS, SS, Centroid, Radius
cf_summary = cf_summary.merge(cf_extra_df, on="CF_Label")

# 9. Xuất kết quả
print("Tổng số CF được tạo ra:", len(cf_summary))
print(cf_summary.head())

cf_summary.to_csv("Demographics_CF_output.csv", index=False)

# =========================
# 10. TRỰC QUAN HÓA
# =========================

# 10.1 PCA scatter plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["CF_Label"], cmap="tab20", s=30, alpha=0.7)
plt.colorbar(scatter, label="CF Label")
plt.title("BIRCH Micro-Clusters (PCA 2D, StandardScaler)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 10.2 Heatmap chia trang (20 CF mỗi hình)
cf_no_extra = cf_summary.set_index("CF_Label")[features]
page_size = 20
n_pages = math.ceil(len(cf_no_extra) / page_size)

for i in range(n_pages):
    start, end = i * page_size, min((i + 1) * page_size, len(cf_no_extra))
    plt.figure(figsize=(12, 6))
    sns.heatmap(cf_no_extra.iloc[start:end], annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"size": 8})
    plt.title(f"Đặc trưng trung bình CF (Trang {i+1}/{n_pages})")
    plt.xlabel("Feature")
    plt.ylabel("CF Label")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

# 10.3 Bar chart số lượng khách hàng theo CF (chia trang 15 cụm mỗi hình)
counts = cf_summary.set_index("CF_Label")["N"]
page_size = 15
n_pages = math.ceil(len(counts) / page_size)

for i in range(n_pages):
    start, end = i * page_size, min((i + 1) * page_size, len(counts))
    plt.figure(figsize=(10, 5))
    counts.iloc[start:end].plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Số lượng khách hàng trong từng CF (Trang {i+1}/{n_pages})")
    plt.xlabel("CF Label")
    plt.ylabel("Số khách hàng")
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()


from sklearn.cluster import KMeans

# 1. Lấy các centroid từ CF (ở dạng gốc)
centroids = np.vstack(cf_summary["Centroid"].values)

# 2. Chuẩn hóa lại centroid trước khi chạy KMeans
centroids_scaled = scaler.fit_transform(centroids)

# 3. Chạy KMeans (ví dụ gom thành 4 cụm chính)
kmeans = KMeans(n_clusters=4, random_state=42)
cf_summary["KMeans_Label"] = kmeans.fit_predict(centroids_scaled)

# 4. Gán nhãn KMeans cho từng khách hàng dựa vào CF_Label
cf_map = dict(zip(cf_summary["CF_Label"], cf_summary["KMeans_Label"]))
df["KMeans_Label"] = df["CF_Label"].map(cf_map)

# =========================
# 5. Trực quan hóa
# =========================

# 5.1 PCA scatter plot toàn bộ khách hàng, tô theo KMeans
X_pca = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["KMeans_Label"], cmap="tab10", s=30, alpha=0.7)
plt.colorbar(scatter, label="KMeans Cluster")
plt.title("KMeans Clusters sau khi BIRCH (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 5.2 Bar chart số lượng khách hàng theo KMeans
plt.figure(figsize=(8, 5))
df["KMeans_Label"].value_counts().sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Số lượng khách hàng trong từng KMeans Cluster")
plt.xlabel("KMeans Cluster")
plt.ylabel("Số khách hàng")
plt.xticks(rotation=0)
plt.show()
