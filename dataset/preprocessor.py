# clean_marketing_campaign_v5.py
import pandas as pd
import numpy as np
from datetime import datetime

# ========= CONFIG =========
INPUT_PATH = "marketing_campaign.csv"             # tệp gốc (thường TSV)
OUTPUT_PATH = "marketing_campaign_cleaned.csv"    # tệp sau khi làm sạch
REFERENCE_YEAR = 2025
ANALYSIS_DATE = datetime(2025, 9, 24)
# ==========================

# ====== COLUMN ORDER (hard-coded from your 2nd file) ======
COLUMN_ORDER = [
    "Age",
    "Education",          # đã thay bằng nhãn số
    "Marital_Status",     # đã thay bằng nhãn số
    "Children",
    "Income",
    "Days_Since_Enroll",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "Total_Spent",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "Total_Purchases",
    "Online_Ratio",
    "NumWebVisitsMonth",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Complain",
    "Response",
]

def normalize_education(x: str) -> str:
    """Gộp các biến thể 'Basic.*' về 'Basic'."""
    if pd.isna(x):
        return x
    x = str(x).strip()
    if x.startswith("Basic"):
        return "Basic"
    return x

def map_education_to_label(x: str):
    """
    Mapping:
    0: Basic | 1: Graduation | 2: 2n Cycle | 3: Master | 4: PhD
    """
    if pd.isna(x):
        return np.nan
    x = normalize_education(x)
    m = {"Basic": 0, "Graduation": 1, "2n Cycle": 2, "Master": 3, "PhD": 4}
    return m.get(x, np.nan)

def map_marital_to_label(x: str) -> int:
    """
    Mapping:
    0: Single | 1: Together | 2: Married | 3: Divorced | 4: Widow | 5: Other (YOLO/Absurd/Alone/khác)
    """
    if pd.isna(x):
        return 5
    x = str(x).strip()
    if x == "Single":
        return 0
    if x == "Together":
        return 1
    if x == "Married":
        return 2
    if x == "Divorced":
        return 3
    if x == "Widow":
        return 4
    if x in {"YOLO", "Absurd", "Alone"}:
        return 5
    return 5

def main():
    # 1) Đọc dữ liệu gốc (dataset marketing_campaign thường là TSV)
    df = pd.read_csv(INPUT_PATH, sep="\t")

    # 2) Loại bỏ bản ghi thiếu Income (KHÔNG điền median)
    df = df.dropna(subset=["Income"]).copy()

    # 3) Bỏ cột ít giá trị
    df.drop(columns=[c for c in ["ID", "Z_CostContact", "Z_Revenue"] if c in df.columns],
            inplace=True, errors="ignore")

    # 4) Year_Birth -> Age
    df["Age"] = REFERENCE_YEAR - df["Year_Birth"].astype(int)

    # 5) Dt_Customer -> Days_Since_Enroll (tính tới ANALYSIS_DATE)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
    df["Days_Since_Enroll"] = (ANALYSIS_DATE - df["Dt_Customer"]).dt.days

    # 6) Children = Kidhome + Teenhome, rồi xóa Kidhome/Teenhome
    df["Children"] = df["Kidhome"].astype(int) + df["Teenhome"].astype(int)
    df.drop(columns=["Kidhome", "Teenhome"], inplace=True, errors="ignore")

    # 7) Total_Spent từ 6 nhóm chi tiêu
    mnt_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    for c in mnt_cols:
        if c not in df.columns:
            df[c] = 0
    df["Total_Spent"] = df[mnt_cols].sum(axis=1)

    # 8) Total_Purchases từ các kênh mua
    purchase_cols = ["NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]
    for c in purchase_cols:
        if c not in df.columns:
            df[c] = 0
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    # 9) Online_Ratio = NumWebPurchases / Total_Purchases (safe divide)
    df["Online_Ratio"] = np.where(df["Total_Purchases"] > 0,
                                  df["NumWebPurchases"] / df["Total_Purchases"],
                                  0.0)

    # 10) Education/Marital_Status: thay thế HOÀN TOÀN bằng nhãn số
    df["Education"] = df["Education"].apply(map_education_to_label).astype("Int64")
    df["Marital_Status"] = df["Marital_Status"].apply(map_marital_to_label).astype("Int64")

    # 11) Bỏ các cột gốc đã chuyển đổi
    df.drop(columns=["Year_Birth", "Dt_Customer"], inplace=True, errors="ignore")

    # 12) KHÓA CỨNG: giữ đúng và chỉ đúng các cột trong COLUMN_ORDER, theo đúng thứ tự
    missing = [c for c in COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Các cột bắt buộc bị thiếu: {missing}")

    df = df[COLUMN_ORDER]  # chỉ giữ và theo đúng thứ tự cột mong muốn

    # 13) Lưu kết quả
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
