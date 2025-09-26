# -*- coding: utf-8 -*-
import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"
PRODUCTS = ["MntWines","MntFruits","MntMeatProducts",
            "MntFishProducts","MntSweetProducts","MntGoldProds"]

BIRCH_LIST  = [0.2, 0.3, 0.4]   # giá trị thử threshold L1
ROLLUP_LIST = [0.5, 0.6, 0.7]   # giá trị thử roll-up L2
TRY_K_LIST  = [3, 4, 5, 6, 7]

MIN_CF_SIZE     = 3
MAX_CF_RADIUS_Q = 0.90
WEIGHT_EXP      = 1.6
RANDOM_STATE    = 42

# =========================
# HÀM TIỆN ÍCH (tái sử dụng từ code gốc, rút gọn)
# =========================
def build_mix_clr_and_logtotal(df, product_cols, eps=1e-6):
    total = df[product_cols].sum(axis=1)
    total_safe = total.replace(0, np.nan)
    mix = df[product_cols].div(total_safe, axis=0).fillna(0.0)
    log_mix = np.log(mix + eps)
    clr = log_mix.sub(log_mix.mean(axis=1), axis=0)
    log_total = np.log1p(total.clip(lower=0))
    return pd.concat([clr.add_prefix("clr_"), pd.Series(log_total, name="log_total")], axis=1)

def birch_labels(X_scaled, threshold):
    return Birch(threshold=threshold, n_clusters=None).fit_predict(X_scaled)

def summarize_cf(df_products, cf_labels, X_scaled):
    tmp = df_products.copy()
    tmp["CF_Label"] = cf_labels
    rows = []
    for lab, idx in tmp.groupby("CF_Label").indices.items():
        idx = np.asarray(list(idx))
        N = len(idx)
        centroid_scaled = X_scaled[idx].mean(axis=0)
        var_vec = X_scaled[idx].var(axis=0, ddof=0)
        radius_scaled = float(np.sqrt(var_vec.sum()))
        rows.append({"CF_Label": lab, "N": N, "Centroid_scaled": centroid_scaled,
                     "Radius_scaled": radius_scaled})
    return pd.DataFrame(rows)

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
    df_l1 = cf_summary_l1.copy(); df_l1["L2_Label"] = labels2
    rows = []
    for lab, g in df_l1.groupby("L2_Label"):
        w = g["N"].values.astype(float)
        cents = np.vstack(g["Centroid_scaled"].values)
        centroid_l2 = (cents * w[:, None]).sum(axis=0) / w.sum()
        rows.append({"L2_Label": lab, "N": int(w.sum()), "Centroid_scaled": centroid_l2})
    return df_l1, pd.DataFrame(rows)

def kmeans_on_cf(cf_summary, n_clusters, weight_exp=1.0, random_state=42):
    cents = np.vstack(cf_summary["Centroid_scaled"].values)
    weights = (cf_summary["N"].astype(float).values) ** float(weight_exp)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    km.fit(cents, sample_weight=weights)
    return km

# =========================
# MAIN
# =========================
df = pd.read_csv(INPUT_CSV)
X = build_mix_clr_and_logtotal(df, PRODUCTS)
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)

results = []
for bt in BIRCH_LIST:
    for rt in ROLLUP_LIST:
        # BIRCH L1
        cf_labels_l1 = birch_labels(X_scaled, threshold=bt)
        cf_l1 = summarize_cf(df[PRODUCTS], cf_labels_l1, X_scaled)
        cf_l1_filtered = filter_cf(cf_l1, min_size=MIN_CF_SIZE, max_radius_q=MAX_CF_RADIUS_Q)

        # Roll-up L2
        df_l1_with_l2, cf_l2 = rollup_cf(cf_l1_filtered, rollup_threshold=rt)
        cf_l2_for_km = cf_l2.rename(columns={"L2_Label":"CF_Label"})

        best_k, best_sil, best_db = None, -1, None
        for k in TRY_K_LIST:
            km = kmeans_on_cf(cf_l2_for_km, k, weight_exp=WEIGHT_EXP, random_state=RANDOM_STATE)
            preds = km.predict(X_scaled)  # predict trực tiếp tất cả điểm
            if len(np.unique(preds)) > 1:
                s = silhouette_score(X_scaled, preds)
                db = davies_bouldin_score(X_scaled, preds)
            else:
                s, db = np.nan, np.nan
            if not np.isnan(s) and s > best_sil:
                best_k, best_sil, best_db = k, s, db

        results.append({
            "BIRCH_THRESHOLD": bt,
            "ROLLUP_THRESHOLD": rt,
            "Best_K": best_k,
            "Silhouette": round(best_sil, 3),
            "DaviesBouldin": round(best_db, 3),
            "CF_L1": len(cf_l1),
            "CF_L1_filtered": len(cf_l1_filtered),
            "CF_L2": len(cf_l2)
        })

res_df = pd.DataFrame(results)
print("\n=== KẾT QUẢ THỬ NGHIỆM ===")
print(res_df)

# chọn phiên bản tốt nhất (theo Silhouette ↑, tie-break DB ↓)
best_row = res_df.sort_values(["Silhouette", "DaviesBouldin"], ascending=[False, True]).iloc[0]
print("\n>>> PHIÊN BẢN TỐT NHẤT:")
print(best_row)
