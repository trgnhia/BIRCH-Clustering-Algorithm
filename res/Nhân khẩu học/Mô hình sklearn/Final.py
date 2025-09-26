# -*- coding: utf-8 -*-
import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.colors import ListedColormap
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"

# Chọn K theo danh sách này (sẽ đánh giá và chọn K tốt nhất)
TRY_K_LIST = [4, 5, 6]

# BIRCH
BIRCH_THRESHOLD = 0.2       # nhỏ -> CF chặt hơn
MIN_CF_SIZE = 3              # lọc CF quá nhỏ trước KMeans (0 = không lọc)

# Biến số sử dụng
FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]

# Tiền xử lý
APPLY_LOG_TRANSFORM = True   # log1p cho Income/Total_Spent
LOG_COLS = ["Income", "Total_Spending"]

RANDOM_STATE = 42

# --- Thư mục output cùng cấp script ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Trường hợp chạy trong notebook/REPL
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_ver4_consistent"  # [CHANGED] tách folder để tránh ghi đè
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Xuất file
OUT_CF = str(OUT_DIR / "CF_summary.csv")
OUT_KMEANS_SUMMARY = str(OUT_DIR / "KMeans_summary.csv")
OUT_KMEANS_ZSCORE   = str(OUT_DIR / "KMeans_zscore.csv")

# =========================
# HÀM TIỆN ÍCH
# =========================
def load_and_prepare(df_path, features, log_cols=None, apply_log=False):
    df = pd.read_csv(df_path)
    X_raw = df[features].copy()        # [NEW] giữ bản raw để thống kê ở thang gốc
    X = df[features].copy()

    # xử lý thiếu
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in features:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    if apply_log and log_cols:
        for c in log_cols:
            if c in X.columns:
                X[c] = np.log1p(np.clip(X[c], a_min=0, a_max=None))  # [same as before]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # scale trên dữ liệu đã log (nếu có)

    return df, X_raw, X, X_scaled, scaler   # [CHANGED] trả thêm X_raw, X (log)

def build_cf_consistent(df, X_raw, X_log, X_scaled, features, threshold=1.0):
    """
    Tạo CF từ BIRCH trên KHÔNG GIAN X_SCALED (nhất quán).
    - Centroid_CF được tính TRỰC TIẾP trên X_SCALED (không cần scale thêm về sau).
    - Đồng thời lưu mean theo thang gốc (X_raw) để đọc hiểu (report).
    """
    birch = Birch(threshold=threshold, n_clusters=None)  # [same]
    birch.fit(X_scaled)
    labels = birch.labels_
    df = df.copy()
    df["CF_Label"] = labels

    # Gom theo CF_Label bằng index
    cf_rows = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        N = int(len(idx))
        # Centroid trong không gian chuẩn hoá (NHẤT QUÁN để dùng cho KMeans)
        centroid_scaled = X_scaled[idx].mean(axis=0)  # [NEW] centroid ở X_scaled
        # Radius trong không gian chuẩn hoá
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))  # [NEW]

        # Mean theo thang gốc để report (dễ hiểu)
        mean_raw = pd.Series(X_raw.iloc[idx].mean(axis=0), index=features)
        mean_log = pd.Series(X_log.iloc[idx].mean(axis=0), index=features)

        cf_rows.append({
            "CF_Label": lab,
            "N": N,
            "Centroid_scaled": centroid_scaled,  # mảng numpy (dùng nội bộ)
            "Radius_scaled": radius_scaled,
            **{f"{c}_mean_raw": mean_raw[c] for c in features},
            **{f"{c}_mean_log": mean_log[c] for c in features},
        })

    cf_summary = pd.DataFrame(cf_rows)

    # Bảng mean cho report heatmap (thang gốc)
    cf_mean_raw = df.groupby("CF_Label")[features].mean().reset_index()  # [same idea, raw]
    # Gộp để có cả mean_raw cột + N + centroid_scaled/radius_scaled
    cf_summary = cf_mean_raw.merge(cf_summary.drop(columns=[f"{c}_mean_raw" for c in features] +
                                                   [f"{c}_mean_log" for c in features]),
                                   on="CF_Label", how="left")
    return df, cf_summary

def kmeans_weighted_on_cf_scaled(cf_summary, n_clusters, random_state=42):
    """
    Chạy KMeans có trọng số TRỰC TIẾP TRÊN 'Centroid_scaled' (đÃ ở không gian X_scaled).
    => Không fit scaler mới (tránh lệch).
    """
    centroids_scaled = np.vstack(cf_summary["Centroid_scaled"].values)  # [CHANGED]
    weights = cf_summary["N"].values
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(centroids_scaled, sample_weight=weights)
    return km

def map_all_cf_to_klabels(df_with_cf, cf_summary, km_big, cf_summary_big, k, random_state):
    """
    Ánh xạ CF -> nhãn KMeans (macro). Nếu có CF nhỏ bị lọc, predict trên centroids_scaled.
    """
    cf_to_k = {}

    # Ánh xạ các CF lớn (đã train)
    labs_big = cf_summary_big["CF_Label"].values
    preds_big = km_big.labels_
    for lab, p in zip(labs_big, preds_big):
        cf_to_k[int(lab)] = int(p)

    # Nếu bị lọc: predict CF nhỏ trên cùng không gian (centroids_scaled)
    if len(cf_summary_big) < len(cf_summary):
        small = cf_summary[~cf_summary["CF_Label"].isin(labs_big)]
        if len(small) > 0:
            cent_small = np.vstack(small["Centroid_scaled"].values)
            preds_small = km_big.predict(cent_small)  # [CHANGED] dùng cùng mô hình & không gian
            for lab, p in zip(small["CF_Label"].values, preds_small):
                cf_to_k[int(lab)] = int(p)

    # Ánh xạ về từng khách
    macro = df_with_cf["CF_Label"].map(cf_to_k).values
    return cf_to_k, macro

def pca_scatter_from_scaled(X_scaled, labels, title, cmap="tab10"):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap,
                     s=22, alpha=0.7, edgecolors="none")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]:.1f}%)")
    plt.tight_layout(); plt.show()
    return X_pca, var, pca

# =========================
# 1) Đọc & tiền xử lý
# =========================
df, X_raw, X_log, X_scaled, scaler = load_and_prepare(
    INPUT_CSV, FEATURES, LOG_COLS, APPLY_LOG_TRANSFORM
)

# =========================
# 2) BIRCH -> CF (NHẤT QUÁN ở X_scaled)
# =========================
df, cf_summary = build_cf_consistent(df, X_raw, X_log, X_scaled, FEATURES, threshold=BIRCH_THRESHOLD)
print(f"Số CF tạo ra: {len(cf_summary)}")

# (tuỳ chọn) lọc CF nhỏ
if MIN_CF_SIZE > 0:
    kept = cf_summary["N"] >= MIN_CF_SIZE
    print(f"Lọc CF nhỏ: còn {kept.sum()}/{len(cf_summary)} CF (N >= {MIN_CF_SIZE})")
    cf_summary_big = cf_summary[kept].copy()
else:
    cf_summary_big = cf_summary.copy()

# Vẽ PCA theo CF (mỗi chấm = 1 KH, tô theo CF)
pca_scatter_from_scaled(X_scaled, df["CF_Label"], "BIRCH Micro-Clusters (PCA 2D)")

# =========================
# 2.b) Phân tích PCA (loadings, tương quan) trên X_scaled
# =========================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
explained = (pca.explained_variance_ratio_ * 100).round(2)
print("Explained variance (%):", explained.tolist())  # [PC1%, PC2%]

loadings = pd.DataFrame(
    pca.components_.T, index=FEATURES, columns=["PC1_loading", "PC2_loading"]
).sort_values("PC1_loading", ascending=False)
print("\n=== LOADINGS (trọng số feature -> PC) ===")
print(loadings.round(3))

contrib = (loadings**2)
contrib_pc = contrib.div(contrib.sum(axis=0), axis=1) * 100
print("\n=== ĐÓNG GÓP % của từng feature vào PC (chuẩn hoá) ===")
print(contrib_pc.round(1))

scores = pd.DataFrame(X_pca, columns=["PC1_score", "PC2_score"])
Z = pd.DataFrame(X_scaled, columns=FEATURES)
corr_pc = Z.join(scores).corr().loc[FEATURES, ["PC1_score", "PC2_score"]]
print("\n=== TƯƠNG QUAN(feature, PC score) ===")
print(corr_pc.round(2))

ax = loadings.plot(kind="bar", figsize=(8,4))
ax.set_title("PCA Loadings"); ax.set_ylabel("Loading")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# =========================
# Heatmap CF (kèm N) — mean ở thang gốc (để đọc hiểu)
# =========================
# cf_no_extra = cf_summary.set_index("CF_Label")[FEATURES]  # mean raw theo nhóm
# cf_counts = cf_summary.set_index("CF_Label")["N"]
# page_size = 20
# n_pages = math.ceil(len(cf_no_extra) / max(page_size, 1))
# for i in range(n_pages):
#     start, end = i * page_size, min((i + 1) * page_size, len(cf_no_extra))
#     sl = cf_no_extra.iloc[start:end].copy()
#     ylabels = [f"{idx} (N={int(cf_counts.loc[idx])})" for idx in sl.index]

#     plt.figure(figsize=(12,6))
#     ax = sns.heatmap(sl, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"size":8})
#     ax.set_title(f"Đặc trưng trung bình CF (Trang {i+1}/{n_pages})")
#     ax.set_xlabel("Feature"); ax.set_ylabel("CF Label")
#     ax.set_yticklabels(ylabels, rotation=0, fontsize=8)
#     plt.xticks(rotation=45, ha="right", fontsize=9)
#     plt.tight_layout(); plt.show()

# =========================
# 3) Chọn K tốt nhất (KMeans có trọng số TRÊN CF ở X_scaled)
# =========================
results = []
best = None
for k in TRY_K_LIST:
    # Train KMeans trên CF lớn (centroid_scaled, weight=N)
    km_big = kmeans_weighted_on_cf_scaled(cf_summary_big, k, random_state=RANDOM_STATE)  # [CHANGED]
    cf_to_k, macro = map_all_cf_to_klabels(df, cf_summary, km_big, cf_summary_big, k, RANDOM_STATE)

    if len(np.unique(macro)) < 2:
        sil, db = np.nan, np.nan
    else:
        sil = silhouette_score(X_scaled, macro)         # [same] đánh giá trên X_scaled
        db  = davies_bouldin_score(X_scaled, macro)     # [same]

    results.append((k, sil, db))
    if best is None or (not np.isnan(sil) and sil > best[1]):
        best = (k, sil, db, cf_to_k)

print("\n== Đánh giá K ==")
print(pd.DataFrame(results, columns=["K","Silhouette","DaviesBouldin"]).round(3))

# Dùng K tốt nhất
K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
df["KMeans_Label"] = df["CF_Label"].map(CF_TO_K)
print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")

# =========================
# 4) Báo cáo KMeans (trên khách hàng) — ở THANG GỐC để dễ hiểu
# =========================
kmeans_mean   = df.groupby("KMeans_Label")[FEATURES].mean().sort_index()
kmeans_median = df.groupby("KMeans_Label")[FEATURES].median().sort_index()
kmeans_std    = df.groupby("KMeans_Label")[FEATURES].std().sort_index()
kmeans_count  = df["KMeans_Label"].value_counts().sort_index().rename("Count")

summary_table = (
    kmeans_mean.add_suffix("_mean")
    .join(kmeans_median.add_suffix("_median"))
    .join(kmeans_std.add_suffix("_std"))
    .join(kmeans_count)
)

global_mean = df[FEATURES].mean()
global_std  = df[FEATURES].std().replace(0, np.nan)
z_table = (kmeans_mean - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

summary_table.reset_index().to_csv(OUT_KMEANS_SUMMARY, index=False)
z_table.reset_index().to_csv(OUT_KMEANS_ZSCORE, index=False)

print("\n===== TÓM TẮT CỤM (KMeans) =====")
print(summary_table.round(2))
print("\n===== Z-SCORE so với toàn bộ =====")
print(z_table.round(2))

# =========================
# 5) Lưu CF summary & trực quan
# =========================
# Khi lưu CSV không thể ghi mảng numpy đẹp, nên bỏ Centroid_scaled để gọn file:
cf_to_save = cf_summary.drop(columns=["Centroid_scaled"], errors="ignore")  # [NEW]
cf_to_save.to_csv(OUT_CF, index=False)

# =========================
# PCA theo KMeans (nhất quán: PCA trên X_scaled)
# =========================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_ * 100

unique_labels = np.unique(df["KMeans_Label"])
n_clusters = len(unique_labels)
cmap_custom = ListedColormap(plt.cm.get_cmap("tab10").colors[:n_clusters])

plt.figure(figsize=(10, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                 c=df["KMeans_Label"], cmap=cmap_custom,
                 s=22, alpha=0.7, edgecolors="none")
cbar = plt.colorbar(sc, ticks=range(n_clusters))
cbar.set_label("Cluster Label")
plt.title(f"KMeans Clusters (PCA 2D) — K={K_BEST}")
plt.xlabel(f"PC1 ({var[0]:.1f}%)")
plt.ylabel(f"PC2 ({var[1]:.1f}%)")
plt.tight_layout()
plt.show()

# =========================
# Heatmap z-score của cụm (chuẩn hoá theo cột giữa các cụm)
# =========================
plt.figure(figsize=(10, 6))
sns.heatmap(((kmeans_mean - kmeans_mean.mean())/kmeans_mean.std()).T,
            annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Z-score theo trung bình cụm (chuẩn hoá theo cột)")
plt.xlabel("KMeans Cluster"); plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Boxplot (Income/Total_Spent/Age) ở thang gốc
for feat in ["Income", "Total_Spending", "Age"]:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="KMeans_Label", y=feat, data=df, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=df, color="k", size=2, alpha=0.25)
    plt.title(f"Phân phối {feat} theo KMeans (K={K_BEST})")
    plt.tight_layout(); plt.show()

# Bar chart số lượng theo cụm
plt.figure(figsize=(8, 5))
kmeans_count.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Số lượng khách hàng theo KMeans Cluster")
plt.xlabel("KMeans Cluster")
plt.ylabel("Số khách hàng")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\n✅ Hoàn tất.")
print(f"- CF summary: {OUT_CF}")
print(f"- KMeans summary: {OUT_KMEANS_SUMMARY}")
print(f"- KMeans z-score: {OUT_KMEANS_ZSCORE}")

# =========================
# 6) Dự đoán cụm cho khách hàng mới (NHẤT QUÁN NHÃN)
# =========================
# Thay vì fit lại KMeans (sẽ sinh nhãn khác), ta gán theo 'macro-centroid' hiện có trong X_scaled.

# Ví dụ khách hàng mới (thang gốc)
new_customer = {
    "Age": 35,
    "Children": 2,
    "Income": 45000,
    "Total_Spending": 1200
}

# Tạo DataFrame & biến đổi như pipeline (log + scale)
new_df = pd.DataFrame([new_customer])
if APPLY_LOG_TRANSFORM:
    for c in LOG_COLS:
        if c in new_df.columns:
            new_df[c] = np.log1p(np.clip(new_df[c], a_min=0, a_max=None))
new_scaled = scaler.transform(new_df[FEATURES])

# Tính macro-centroid (trên X_scaled) theo nhãn đã có
macro_labels_sorted = sorted(df["KMeans_Label"].unique())
macro_centroids_scaled = np.vstack([
    X_scaled[df["KMeans_Label"].values == k].mean(axis=0)
    for k in macro_labels_sorted
])

# Gán cụm bằng nearest centroid trong X_scaled (nhãn đồng nhất)
dists = ((macro_centroids_scaled - new_scaled)**2).sum(axis=1)
pred_idx = int(np.argmin(dists))
pred_label = macro_labels_sorted[pred_idx]

print("\n===== NEW CUSTOMER =====")
print(new_customer)
print("→ Khách hàng mới thuộc cụm:", pred_label)

# Lấy đặc trưng của cụm đó từ summary_table (đã đúng nhãn)
cluster_features = summary_table.loc[pred_label]
print("\n=== ĐẶC TRƯNG CỤM", pred_label, "===")
print(cluster_features.round(2))
