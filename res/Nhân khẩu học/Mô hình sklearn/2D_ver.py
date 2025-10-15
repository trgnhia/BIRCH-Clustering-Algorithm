# -*- coding: utf-8 -*-
import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"

FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]

# BIRCH & KMeans
# các chỉ số đã được tính từ file best_params
BIRCH_THRESHOLD   = 0.40
ROLLUP_THRESHOLD  = 0.60
MIN_CF_SIZE       = 3
MAX_CF_RADIUS_Q   = 0.90
WEIGHT_EXP        = 1.6
K = 6
TRY_K_LIST        = [6]
RANDOM_STATE      = 42

DO_REFIT_ASSIGN   = True

# Xuất file
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_demographics_2D"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVAL_K          = str(OUT_DIR / "EvalK.csv")
OUT_CLUSTER_SUMMARY = str(OUT_DIR / "Cluster_Summary.csv")
OUT_CF_L1           = str(OUT_DIR / "CF_L1_summary.csv")
OUT_CF_L2           = str(OUT_DIR / "CF_L2_summary.csv")

# =========================
# HÀM TIỆN ÍCH
# =========================

# 1 , 2 , 3a đã được giải thích bên file lấy các chỉ số tốt nhất
def assert_columns_exist(df, cols, label="dataset"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in {label}: {missing}")

def birch_labels(X_scaled, threshold):
    return Birch(threshold=threshold, n_clusters=None).fit_predict(X_scaled)

def summarize_cf(df, cf_labels, X_scaled, features):
    tmp = df.copy()
    tmp["CF_Label"] = cf_labels
    rows = []
    for lab, idx in tmp.groupby("CF_Label").indices.items():
        idx = np.asarray(list(idx))
        N = int(len(idx))
        centroid_scaled = X_scaled[idx].mean(axis=0)
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))
        mean_raw = tmp.iloc[idx][features].mean()
        rows.append({
            "CF_Label": lab,
            "N": N,
            "Centroid_scaled": centroid_scaled,
            "Radius_scaled": radius_scaled,
            **{f"{c}_mean_raw": mean_raw[c] for c in features}
        })
    cf_summary = pd.DataFrame(rows)
    mean_raw_tbl = tmp.groupby("CF_Label")[features].mean().reset_index()
    cf_summary = mean_raw_tbl.merge(
        cf_summary.drop(columns=[f"{c}_mean_raw" for c in features]),
        on="CF_Label", how="left"
    )
    return tmp, cf_summary

def filter_cf(cf_summary, min_size=0, max_radius_q=None):
    mask = pd.Series(True, index=cf_summary.index)
    if min_size > 0:
        mask &= cf_summary["N"] >= min_size
    if max_radius_q is not None:
        thr = cf_summary["Radius_scaled"].quantile(max_radius_q)
        mask &= cf_summary["Radius_scaled"] <= thr
    return cf_summary[mask].copy()

def rollup_cf(cf_summary_l1, rollup_threshold):
    cents = np.vstack(cf_summary_l1["Centroid_scaled"].values)
    labels2 = Birch(threshold=rollup_threshold, n_clusters=None).fit_predict(cents)
    df_l1 = cf_summary_l1.copy()
    df_l1["L2_Label"] = labels2

    rows = []
    for lab, g in df_l1.groupby("L2_Label"):
        w = g["N"].values.astype(float)
        cents = np.vstack(g["Centroid_scaled"].values)
        w_sum = w.sum()
        centroid_l2 = (cents * w[:, None]).sum(axis=0) / max(w_sum, 1.0)
        rows.append({
            "L2_Label": int(lab),
            "N": int(w_sum),
            "Centroid_scaled": centroid_l2
        })
    cf_l2 = pd.DataFrame(rows)
    return df_l1, cf_l2

def kmeans_on_cf(cf_summary, n_clusters, weight_exp=1.0, random_state=42):
    cents = np.vstack(cf_summary["Centroid_scaled"].values)
    weights = (cf_summary["N"].astype(float).values) ** float(weight_exp)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    km.fit(cents, sample_weight=weights)
    return km

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def get_colormap(n_clusters):
    """Sinh colormap có đúng n_clusters màu, không bị trùng."""
    base_cmap = cm.get_cmap("tab20")  # có thể thay bằng "tab10", "Set3", "gist_ncar"...
    if hasattr(base_cmap, "colors"):
        base_colors = base_cmap.colors
        if n_clusters <= len(base_colors):
            colors = base_colors[:n_clusters]
        else:
            # nội suy thêm màu nếu cụm > số màu gốc
            colors = base_cmap(np.linspace(0, 1, n_clusters))
    else:
        colors = base_cmap(np.linspace(0, 1, n_clusters))
    return mcolors.ListedColormap(colors)


def pca_scatter(X_scaled, labels, title):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    n_clusters = len(np.unique(labels))
    cmap_custom = get_colormap(n_clusters)
    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap_custom,
                     s=22, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(sc); cbar.set_label("Label")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

# =========================
# 1) LOAD & TIỀN XỬ LÝ
# =========================
df = pd.read_csv(INPUT_CSV)
assert_columns_exist(df, FEATURES, "FEATURES")

# log transform cho Income, Total_Spending
X = df[FEATURES].copy()
X["Income"] = np.log1p(X["Income"].clip(lower=0))
X["Total_Spending"] = np.log1p(X["Total_Spending"].clip(lower=0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 2) BIRCH L1 + LỌC + ROLLUP L2
# =========================
cf_labels_l1 = birch_labels(X_scaled, threshold=BIRCH_THRESHOLD)
df_cf_l1, cf_l1 = summarize_cf(df[FEATURES], cf_labels_l1, X_scaled, FEATURES)
print(f"Số CF L1 tạo ra: {len(cf_l1)}")

cf_l1_filtered = filter_cf(cf_l1, min_size=MIN_CF_SIZE, max_radius_q=MAX_CF_RADIUS_Q)
print(f"Lọc CF L1: còn {len(cf_l1_filtered)}/{len(cf_l1)}")

df_l1_with_l2, cf_l2 = rollup_cf(cf_l1_filtered, rollup_threshold=ROLLUP_THRESHOLD)
print(f"Số super-CF L2 sau roll-up: {len(cf_l2)}")
def plot_cf_heatmap(cf_summary, features, page_size=20, title="CF Summary", label_col="CF_Label"):
    """
    Vẽ heatmap cho trung bình các feature của CF.
    - cf_summary: DataFrame đã có cột label_col, 'N', và các feature trung bình
    - features: list tên các cột feature
    - page_size: số CF trên mỗi trang
    - label_col: tên cột nhãn ('CF_Label' hoặc 'L2_Label')
    """
    if label_col not in cf_summary.columns:
        raise ValueError(f"[ERROR] Không tìm thấy cột {label_col} trong cf_summary")

    cf_no_extra = cf_summary.set_index(label_col)[features]
    cf_counts = cf_summary.set_index(label_col)["N"]

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


# Với CF L1
#plot_cf_heatmap(cf_l1, FEATURES, page_size=20, title="CF L1 trung bình trước lọc", label_col="CF_Label")

# Với CF L1
#plot_cf_heatmap(cf_l1_filtered, FEATURES, page_size=20, title="CF L1 trung bình sau lọc", label_col="CF_Label")

# Với CF L2
#plot_cf_heatmap(cf_l2, FEATURES, page_size=20, title="CF L2 trung bình", label_col="L2_Label")


pca_scatter(X_scaled, df_cf_l1["CF_Label"], "BIRCH Micro-Clusters (Demographics)")
df_cf_l1 = df_cf_l1.merge(
    df_l1_with_l2[["CF_Label", "L2_Label"]].drop_duplicates(),
    on="CF_Label", how="left"
)
pca_scatter(X_scaled, df_cf_l1["L2_Label"], "BIRCH Super-Clusters (Demographics)")

def pca_scatter_cf(cf_summary_l1_with_l2, title):
    """Vẽ PCA scatter cho CF L1 (mỗi chấm = centroid của CF L1), màu = L2_Label"""
    cents = np.vstack(cf_summary_l1_with_l2["Centroid_scaled"].values)
    labels = cf_summary_l1_with_l2["L2_Label"].values
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(cents)
    var = pca.explained_variance_ratio_ * 100
    n_clusters = len(np.unique(labels))
    cmap_custom = get_colormap(n_clusters)
    plt.figure(figsize=(10,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap=cmap_custom,
                     s=60, alpha=0.9, edgecolors="k")
    cbar = plt.colorbar(sc); cbar.set_label("L2_Label")
    plt.title(f"{title} — PC1 {var[0]:.1f}% | PC2 {var[1]:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()
pca_scatter_cf(df_l1_with_l2, "BIRCH Super-Clusters (Demographics) [mỗi chấm = CF L1]")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
explained = (pca.explained_variance_ratio_ * 100).round(2)
print("Explained variance (%):", explained.tolist())

# Loadings (mức đóng góp của từng feature vào PC)
loadings = pd.DataFrame(
    pca.components_.T,
    index=FEATURES,
    columns=["PC1_loading", "PC2_loading"]
).sort_values("PC1_loading", ascending=False)
print("\n=== LOADINGS (feature → PC) ===")
print(loadings.round(3))

# Đóng góp % (chuẩn hóa bình phương loadings)
contrib = (loadings**2)
contrib_pc = contrib.div(contrib.sum(axis=0), axis=1) * 100
print("\n=== ĐÓNG GÓP % của từng feature vào PC ===")
print(contrib_pc.round(1))

# Tương quan giữa feature và PC score
scores = pd.DataFrame(X_pca, columns=["PC1_score", "PC2_score"])
Z = pd.DataFrame(X_scaled, columns=FEATURES)
corr_pc = Z.join(scores).corr().loc[FEATURES, ["PC1_score", "PC2_score"]]
print("\n=== TƯƠNG QUAN (Feature, PC score) ===")
print(corr_pc.round(2))

# Vẽ biểu đồ loadings
ax = loadings.plot(kind="bar", figsize=(8,4))
ax.set_title("PCA Loadings (feature → PC)")
ax.set_ylabel("Loading")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# 3) KMEANS TRÊN L2
# =========================
# cf_l2_for_km = cf_l2.rename(columns={"L2_Label":"CF_Label"})
# rows, best = [], None
# for k in TRY_K_LIST:
#     km = kmeans_on_cf(cf_l2_for_km, k, weight_exp=WEIGHT_EXP, random_state=RANDOM_STATE)
#     l2_to_k = dict(zip(cf_l2_for_km["CF_Label"].values, km.labels_))
#     cf_to_k_used = dict(zip(df_l1_with_l2["CF_Label"].values,
#                             df_l1_with_l2["L2_Label"].map(l2_to_k).values))
#     cf_to_k_all = {}
#     cf_to_k_all.update(cf_to_k_used)
#     missing = cf_l1[~cf_l1["CF_Label"].isin(cf_to_k_used.keys())]
#     if len(missing) > 0:
#         preds = km.predict(np.vstack(missing["Centroid_scaled"].values))
#         for lab, kpred in zip(missing["CF_Label"].values, preds):
#             cf_to_k_all[int(lab)] = int(kpred)
#     macro = df_cf_l1["CF_Label"].map(cf_to_k_all).values
#     if len(np.unique(macro)) < 2:
#         s, db = np.nan, np.nan
#     else:
#         s  = silhouette_score(X_scaled, macro)
#         db = davies_bouldin_score(X_scaled, macro)
#     rows.append((k, s, db))
#     if best is None or (not np.isnan(s) and s > best[1]):
#         best = (k, s, db, cf_to_k_all)

# eval_df = pd.DataFrame(rows, columns=["K","Silhouette","DaviesBouldin"]).round(3)
# eval_df.to_csv(OUT_EVAL_K, index=False)
# print("\n== Đánh giá K =="); print(eval_df)

# K_BEST, SIL_BEST, DB_BEST, CF_TO_K = best
# df_cf_l1["KMeans_Label"] = df_cf_l1["CF_Label"].map(CF_TO_K)
# print(f"\nChọn K={K_BEST} | Silhouette={SIL_BEST:.3f} | DB={DB_BEST:.3f}")


# =========================
# 3) KMEANS TRÊN L2 (K cố định, không đánh giá s/db)
# =========================
cf_l2_for_km = cf_l2.rename(columns={"L2_Label": "CF_Label"})


km = kmeans_on_cf(cf_l2_for_km, K, weight_exp=WEIGHT_EXP, random_state=RANDOM_STATE)

# Gán nhãn KMeans cho CF L2
l2_to_k = dict(zip(cf_l2_for_km["CF_Label"].values, km.labels_))

# Gán nhãn từ L2 → L1 (CF)
cf_to_k_used = dict(zip(
    df_l1_with_l2["CF_Label"].values,
    df_l1_with_l2["L2_Label"].map(l2_to_k).values
))

# Bổ sung các CF bị thiếu (nếu có)
cf_to_k_all = {}
cf_to_k_all.update(cf_to_k_used)
missing = cf_l1[~cf_l1["CF_Label"].isin(cf_to_k_used.keys())]
if len(missing) > 0:
    preds = km.predict(np.vstack(missing["Centroid_scaled"].values))
    for lab, kpred in zip(missing["CF_Label"].values, preds):
        cf_to_k_all[int(lab)] = int(kpred)

# Gán nhãn KMeans cuối cùng cho toàn bộ khách hàng
CF_TO_K = cf_to_k_all
df_cf_l1["KMeans_Label"] = df_cf_l1["CF_Label"].map(CF_TO_K)

# print(f"\nKMeans hoàn tất với K={K_BEST}. Đã gán nhãn cho {df_cf_l1['KMeans_Label'].notna().sum()} khách hàng.")




# =========================
# 3b) REFINEMENT
# =========================
# tinh gọn , tính toán lại lần cuối
if DO_REFIT_ASSIGN:
    # Lấy danh sách nhãn KMeans đã gán cho CF L1, sắp xếp để ổn định thứ tự
    macro_labels_sorted = np.array(sorted(np.unique(df_cf_l1["KMeans_Label"]))) 

    macro_centroids = []
    # Với mỗi cụm macro (KMeans_Label), tính centroid dựa trên toàn bộ KH thực tế
    for k in macro_labels_sorted:
        idx = np.where(df_cf_l1["KMeans_Label"].values == k)[0]  # lấy index KH thuộc cụm k
        macro_centroids.append(X_scaled[idx].mean(axis=0))       # tính trung bình (centroid thực sự)
    
    # Gom tất cả centroid lại thành ma trận (số cụm x số chiều feature)
    macro_centroids = np.vstack(macro_centroids)

    # Tính khoảng cách bình phương từ mỗi KH tới từng macro-centroid
    # X_scaled[:, None, :]  : (n_kh, 1, n_feat)
    # macro_centroids[None, :, :] : (1, n_cluster, n_feat)
    # Hiệu -> (n_kh, n_cluster, n_feat)
    # Bình phương + sum(axis=2) -> (n_kh, n_cluster) ma trận khoảng cách
    d2 = ((X_scaled[:, None, :] - macro_centroids[None, :, :])**2).sum(axis=2)

    # Mỗi KH gán nhãn lại = cụm có centroid gần nhất (min distance)
    reassigned = macro_labels_sorted[np.argmin(d2, axis=1)]

    # Cập nhật nhãn mới vào dataframe
    df_cf_l1["KMeans_Label"] = reassigned

    # Nếu còn hơn 1 cụm, tính lại chỉ số đánh giá chất lượng
    if len(np.unique(reassigned)) > 1:
        s2 = silhouette_score(X_scaled, reassigned)         # Silhouette (càng cao càng tốt)
        db2 = davies_bouldin_score(X_scaled, reassigned)    # Davies-Bouldin (càng thấp càng tốt)
    else:
        s2, db2 = np.nan, np.nan   # nếu gom hết thành 1 cụm -> không tính được

    # In ra kết quả refinement
    print(f"[Refinement] Silhouette={s2:.3f} | DB={db2:.3f}")


# =========================
# 4) BÁO CÁO CỤM
# =========================
cluster_means   = df_cf_l1.groupby("KMeans_Label")[FEATURES].mean().sort_index()
cluster_median  = df_cf_l1.groupby("KMeans_Label")[FEATURES].median().sort_index()
cluster_std     = df_cf_l1.groupby("KMeans_Label")[FEATURES].std().sort_index()
cluster_count   = df_cf_l1["KMeans_Label"].value_counts().sort_index().rename("Count")

global_mean = df_cf_l1[FEATURES].mean()
global_std  = df_cf_l1[FEATURES].std().replace(0, np.nan)
z_table = (cluster_means - global_mean) / global_std
z_table = z_table.replace([np.inf, -np.inf], np.nan)

summary_table = (
    cluster_means.add_suffix("_mean")
    .join(cluster_median.add_suffix("_median"))
    .join(cluster_std.add_suffix("_std"))
    .join(cluster_count)
)
summary_table.reset_index().to_csv(OUT_CLUSTER_SUMMARY, index=False)

print("\n===== TÓM TẮT CỤM (mean/median/std + count) =====")
print(summary_table.round(2))
print("\n===== Z-SCORE so với toàn bộ =====")
print(z_table.round(2))

# =========================
# 5) TRỰC QUAN
# =========================
pca_scatter(X_scaled, df_cf_l1["KMeans_Label"], f"KMeans Clusters (K={len(cluster_count)}) — Demographics")

plt.figure(figsize=(12,6))
sns.heatmap(z_table.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Z-score nhân khẩu học theo cụm")
plt.xlabel("Cluster"); plt.ylabel("Feature")
plt.tight_layout(); plt.show()

for feat in FEATURES:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="KMeans_Label", y=feat, data=df_cf_l1, palette="Set2")
    sns.stripplot(x="KMeans_Label", y=feat, data=df_cf_l1, color="k", size=2, alpha=0.25)
    plt.title(f"Phân phối {feat} theo cụm (K={len(cluster_count)})")
    plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
cluster_count.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Số lượng khách hàng theo cụm KMeans")
plt.xlabel("KMeans Cluster"); plt.ylabel("Số khách hàng")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# =========================
# 6) LƯU CF SUMMARY
# =========================
cf_l1.drop(columns=["Centroid_scaled"], errors="ignore").to_csv(OUT_CF_L1, index=False)
cf_l2.to_csv(OUT_CF_L2, index=False)

print("\n✅ Hoàn tất.")
print(f"- EvalK: {OUT_EVAL_K}")
print(f"- Cluster Summary: {OUT_CLUSTER_SUMMARY}")
print(f"- CF L1 summary: {OUT_CF_L1}")
print(f"- CF L2 summary: {OUT_CF_L2}")

# =========================
# 7) DỰ ĐOÁN KHÁCH HÀNG MỚI
# =========================
new_customer = {
    "Customer_Age": 35,
    "Children": 2,
    "Income": 45000,
    "Total_Spending": 1200
}
new_df = pd.DataFrame([new_customer])
new_df["Income"] = np.log1p(new_df["Income"].clip(lower=0))
new_df["Total_Spending"] = np.log1p(new_df["Total_Spending"].clip(lower=0))
new_scaled = scaler.transform(new_df[FEATURES])

macro_labels_sorted = np.array(sorted(np.unique(df_cf_l1["KMeans_Label"])))
macro_centroids = np.vstack([
    X_scaled[df_cf_l1["KMeans_Label"].values == k].mean(axis=0)
    for k in macro_labels_sorted
])
dists = ((macro_centroids - new_scaled)**2).sum(axis=1)
pred_idx = int(np.argmin(dists))
pred_label = int(macro_labels_sorted[pred_idx])

print("\n===== NEW CUSTOMER =====")
print(new_customer)
print("→ Khách hàng mới thuộc cụm:", pred_label)
print("\n=== ĐẶC TRƯNG CỤM", pred_label, "===")
print(summary_table.loc[pred_label].round(2))
