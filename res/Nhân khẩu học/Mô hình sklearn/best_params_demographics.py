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

BIRCH_LIST  = [0.2, 0.3, 0.4]    # threshold L1 - ngưỡng giữa các khách hàng
ROLLUP_LIST = [0.5, 0.6, 0.7]    # threshold L2 (roll-up) - ngưỡng giữa các CFl1
TRY_K_LIST  = [3, 4, 5, 6, 7, 8, 9]          # số cluster thử

MIN_CF_SIZE     = 3 # số N tối thiểu trong 1 CF
MAX_CF_RADIUS_Q = 0.90 # bán kính tối đa
WEIGHT_EXP      = 1.6
RANDOM_STATE    = 42

OUT_CSV = "best_params_demographics.csv"

# =========================
# HÀM TIỆN ÍCH
# =========================
# Chạy hàm birch tiêu chuẩn 
def birch_labels(X_scaled, threshold):
    return Birch(threshold=threshold, n_clusters=None).fit_predict(X_scaled)

# Hàm này giúp tóm tắt từng CF
def summarize_cf(X_scaled, cf_labels):
    rows = []
    for lab in np.unique(cf_labels):                  # duyệt qua từng CF label (mỗi cụm CF)
        idx = np.where(cf_labels == lab)[0]           # lấy index của các điểm thuộc CF đó
        N = len(idx)                                  # số lượng điểm trong CF
        centroid = X_scaled[idx].mean(axis=0)         # vector trung tâm (centroid) tính theo trung bình
        radius   = np.sqrt(X_scaled[idx].var(axis=0, ddof=0).sum())  
                                                     # bán kính: sqrt(tổng phương sai các chiều)
        rows.append({"CF_Label": lab, 
                     "N": N,
                     "Centroid_scaled": centroid,
                     "Radius_scaled": radius})
    return pd.DataFrame(rows)                        # trả về bảng tóm tắt các CF

# Lọc các CF từ bảng tóm tắt cf_summary 
def filter_cf(cf_summary, min_size=0, max_radius_q=None):
    mask = pd.Series(True, index=cf_summary.index) # Đánh dấu taatrs cả = true : chưa bỏ cái nào
    if min_size > 0:
        mask &= cf_summary["N"] >= min_size         # Chỉ lấy các cụm có N >= Minsize
    if max_radius_q is not None:
        thr = cf_summary["Radius_scaled"].quantile(max_radius_q) #Lấy ngưỡng bán kính theo %
        mask &= cf_summary["Radius_scaled"] <= thr  # chỉ lấy các cụm có bán kính nhỏ hơn ngưỡng tức là các cụm đặc , mang nhiều đặc trưng , cụm có bán kính lớn sẽ bị loãng , mang đặc trưng có thể không đúng
    return cf_summary[mask].copy()



def rollup_cf(cf_summary_l1, rollup_threshold):
    # thực hiện birch trên các CFL1 thu được , cột chính là Centroid_scaled thay vì chạy toàn bộ thông tin CF ~ thông tin các khách hàng 
    # Centroid_scaled của các CFL1 đã sẽ được coi là tọa độ của các cụm con để phân thành các siêu cụm CFL2
    cents = np.vstack(cf_summary_l1["Centroid_scaled"].values)
    labels2 = Birch(threshold=rollup_threshold, n_clusters=None).fit_predict(cents)
    df_l1 = cf_summary_l1.copy(); df_l1["L2_Label"] = labels2 # gán nhãn CFL2 cho các CFL1 : CLF1 nào thuộc về CFL2 nào
    rows = []
    # Gom và tính lại các chỉ số
    for lab, g in df_l1.groupby("L2_Label"):
        w = g["N"].values.astype(float)
        cents = np.vstack(g["Centroid_scaled"].values)
        centroid_l2 = (cents * w[:, None]).sum(axis=0) / w.sum()
        rows.append({"L2_Label": lab, "N": int(w.sum()), "Centroid_scaled": centroid_l2})
    # trả về các CFL2
    return df_l1, pd.DataFrame(rows)

def kmeans_on_cf(cf_summary, n_clusters):
    cents = np.vstack(cf_summary["Centroid_scaled"].values) # lấy ra cột centroid_scaled từ CFL2
    weights = (cf_summary["N"].astype(float).values) ** WEIGHT_EXP # tính trọng số 
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20) # khởi chạy kmeans
    km.fit(cents, sample_weight=weights) # thực hiện huấn luyện 
    return km # trả về toàn bộ mô hình kmeans sau khi huấn luyện

# =========================
# MAIN
# =========================
df = pd.read_csv(INPUT_CSV) # Lấy file dataset 
X = df[FEATURES].copy() # lấy ra các cột cần thiết
# log transform cho Income & Spending

# Lọa bỏ các giá trị < 0 -> = 0
X[["Income","Total_Spending"]] = np.log1p(X[["Income","Total_Spending"]].clip(lower=0))

# Chuẩn hóa các trường về cùng 1 thang đo theo công thức z = (x - u )/ o - u : trung bình , o : độ lệch chuẩn
X_scaled = StandardScaler().fit_transform(X)

results = []
# lặp mọi cặp Threshold tại CFL1 và RollUp_Threshold tại CFL2
for bt in BIRCH_LIST:
    for rt in ROLLUP_LIST:
        # CF L1
        # Birch tại CFL1
        cf_labels_l1 = birch_labels(X_scaled, threshold=bt)
        #lấy ra bảng tóm tắt của các CF
        cf_l1 = summarize_cf(X_scaled, cf_labels_l1)

        # lọc dựa trên N tối thiểu và ngưỡng bán kính tối đa
        cf_l1_filtered = filter_cf(cf_l1, MIN_CF_SIZE, MAX_CF_RADIUS_Q)
        #Roll up ( gom ) CFl2 dựa trên những CFL1 đã được lọc và ngưỡng tối thiểu
        # CF L2
        _, cf_l2 = rollup_cf(cf_l1_filtered, rollup_threshold=rt)
        cf_l2_for_km = cf_l2.rename(columns={"L2_Label": "CF_Label"})

        # thử các K
        best_k, best_sil, best_db = None, -1, None
        # thực hiện lấy ra các chỉ số bao gồm K , Silhouette( ) , DaviesBouldin
        for k in TRY_K_LIST:
            km = kmeans_on_cf(cf_l2_for_km, k) # chạy kmeans với CFL2
            preds = km.predict(X_scaled)   # predict toàn bộ KH để đẩy khách hàng vào các cụm đã lấy được sau khi chạy với CFL2
            if len(np.unique(preds)) > 1:
                s  = silhouette_score(X_scaled, preds) # lấy ra chỉ số silhouette [-1,1] 
                #s(i) =( b(i) - a(i) )/  max(a(i),b(i)) với a(i) là độ gọn trong cụm , b(i) là khoảng cách trung bình tới tất cả các điểm trong cụm gần nhất
                #Silhouette score đo độ gọn trong cụm và độ tách biệt giữa các cụm.  Giá trị gần 1 = cụm rõ ràng, gần 0 = chồng chéo, <0 = phân cụm tệ.
                
                db = davies_bouldin_score(X_scaled, preds) #DBI đo sự cân bằng giữa độ gọn trong cụm và khoảng cách giữa cụm.Giá trị càng nhỏ càng tốt.
            else:
                s, db = np.nan, np.nan
            if not np.isnan(s) and s > best_sil: # tốt hơn thì lấy
                best_k, best_sil, best_db = k, s, db
        # hàm add được đặt ngoài vòng for tìm K tức là lúc này đã tìm được K , s và db tốt nhất tương ứng với từng cặp threshold
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
