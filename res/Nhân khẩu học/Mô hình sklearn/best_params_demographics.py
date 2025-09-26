# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "dataset/data_cleaning/cleaned_dataset.csv"
FEATURES = ["Customer_Age", "Children", "Income", "Total_Spending"]

BIRCH_LIST  = [0.2, 0.3, 0.4]    # threshold L1
ROLLUP_LIST = [0.5, 0.6, 0.7]    # threshold L2 (roll-up)
TRY_K_LIST  = [3,4, 5, 6,7,8,9]          # số cluster thử

MIN_CF_SIZE     = 3
MAX_CF_RADIUS_Q = 0.90
WEIGHT_EXP      = 1.6
RANDOM_STATE    = 42

OUT_CSV = "best_params_demographics.csv"

# =========================
# HÀM TIỆN ÍCH
# =========================
def birch_labels(X_scaled, threshold):
    return Birch(threshold=threshold, n_clusters=None).fit_predict(X_scaled)

def summarize_cf(X_scaled, cf_labels):
    rows = []
    for lab in np.unique(cf_labels):
        idx = np.where(cf_labels == lab)[0]
        N = len(idx)
        centroid = X_scaled[idx].mean(axis=0)
        radius   = np.sqrt(X_scaled[idx].var(axis=0, ddof=0).sum())
        rows.append({"CF_Label": lab, "N": N,
                     "Centroid_scaled": centroid,
                     "Radius_scaled": radius})
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

def kmeans_on_cf(cf_summary, n_clusters):
    cents = np.vstack(cf_summary["Centroid_scaled"].values)
    weights = (cf_summary["N"].astype(float).values) ** WEIGHT_EXP
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    km.fit(cents, sample_weight=weights)
    return km

# =========================
# MAIN
# =========================
df = pd.read_csv(INPUT_CSV)
X = df[FEATURES].copy()
# log transform cho Income & Spending
X[["Income","Total_Spending"]] = np.log1p(X[["Income","Total_Spending"]].clip(lower=0))
X_scaled = StandardScaler().fit_transform(X)

results = []
for bt in BIRCH_LIST:
    for rt in ROLLUP_LIST:
        # CF L1
        cf_labels_l1 = birch_labels(X_scaled, threshold=bt)
        cf_l1 = summarize_cf(X_scaled, cf_labels_l1)
        cf_l1_filtered = filter_cf(cf_l1, MIN_CF_SIZE, MAX_CF_RADIUS_Q)

        # CF L2
        _, cf_l2 = rollup_cf(cf_l1_filtered, rollup_threshold=rt)
        cf_l2_for_km = cf_l2.rename(columns={"L2_Label": "CF_Label"})

        # thử các K
        best_k, best_sil, best_db = None, -1, None
        for k in TRY_K_LIST:
            km = kmeans_on_cf(cf_l2_for_km, k)
            preds = km.predict(X_scaled)   # predict toàn bộ KH
            if len(np.unique(preds)) > 1:
                s  = silhouette_score(X_scaled, preds)
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

best_row = res_df.sort_values(["Silhouette","DaviesBouldin"], ascending=[False,True]).iloc[0]
print("\n>>> PHIÊN BẢN TỐT NHẤT:")
print(best_row)

# Lưu CSV
res_df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Đã lưu toàn bộ kết quả vào {OUT_CSV}")
