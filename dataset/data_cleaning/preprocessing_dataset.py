
import warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Tải và kiểm tra dữ liệu 
df_customer = pd.read_csv('dataset/origin/marketing_campaign.csv', sep='\t')

# Đọc 5 dòng đầu , mỗi dòng = 1 khách hàng
print(df_customer.head())

# Các cột trong dữ liệu
print(df_customer.columns) # x29 cột
# kiểu dữ liệu cho từng feature
print(df_customer.info())

#kích thước dataset
print(df_customer.shape) # (2240, 29)

"""
Phân tích các trường lấy được :
=== People:
ID: Unique identifier for each customer.
Year_Birth: Customer's birth year.
Education: Customer's education level.
Marital_Status: Customer's marital status.
Income: Customer's Yearly household income.
Kidhome: Number of children in the household.
Teenhome: Number of teenagers in the household.
Dt_Customer: Date of Customer enrollment.
Recency: Number of days since the last purchase.
Complain: 1 if the Customer complained in the last 2 years, 0 otherwise.

=== Products, amount spent on different product categories in the last 2 years:
MntWines: Various wine products.
MntFruits: Fresh fruit products.
MntMeatProducts: Fresh meat products.
MntFishProducts: Fresh fish products.
MntSweetProducts: Various candy and sweet products.
MntGoldProds: Various gold and precious metal products.

==Promotional Campaign:
NumDealsPurchases: Number of purchases made with a discount.
AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise.
AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise.
AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise.
AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise.
AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise.
Response: 1 if customer accepted the offer in the last campaign, 0 otherwise.

==Place:
NumWebPurchases: Number of purchases made through the company's website.
NumCatalogPurchases: Number of purchases made using a catalogue.
NumStorePurchases: Number of purchases made directly in stores.
NumWebVisitsMonth: Number of visits to company's web site in the last month.

==Others:
Z_CostContact: Constant value (likely a placeholder) related to customer contact cost.
Z_Revenue: Constant value (likely a placeholder) related to customer revenue.

"""

# kiểm tra tỉ trọng trong các feature 
for column in df_customer.columns:
    print(f"--- {column} ---")
    print(df_customer[column].value_counts(normalize=True))  # Tỉ trọng
    print("\n")
# Kiểm tra phân phối giá trị của các trường dữ liệu dạng số
print(df_customer[["Year_Birth", "Income", "Kidhome", "Teenhome", 
             "Recency", "MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds",
            "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
            "NumStorePurchases", "NumWebVisitsMonth"]].describe().T.round(2))
#Kiểm tra khách hàng trùng nhau dựa trên ID
# Check for any duplicated values based on selected features
# keep=False means marking all duplicates
duplicated_value = df_customer.duplicated(subset=["ID"], keep=False)
df_customer_duplicate = df_customer[duplicated_value].sort_values(by=["ID"])

# Show any duplicated values
print(df_customer_duplicate) # Không có ông cháu nào bị lặp

# Kiểm tra các giá trị bị thiếu
print(df_customer.isnull().sum()[df_customer.isnull().sum() > 0]) # income thiếu 20 ông cháu

print(df_customer[df_customer.isnull().any(axis=1)]) # in ra các dòng có giá trị bị thiếu

#điền vào chỗ trống = trung vị 
median_income = df_customer["Income"].median()
df_customer["Income"] = df_customer["Income"].fillna(median_income)

#kiểm tra lại
print(df_customer.isnull().sum()[df_customer.isnull().sum() > 0]) # không còn thiếu nữa

# Xử lý trường Marital_Status (tình trạng hôn nhân)
df_customer['Marital_Status'] = df_customer['Marital_Status'].replace({
    'Absurd': 'Single',
    'Alone': 'Single',
    'YOLO': 'Single'
}) # gộp các trạng thái lạ vào single

#Thêm cột tuổi
df_customer["Customer_Age"] = 2015 - df_customer["Year_Birth"] # Dữ liệu cũ nên lấy từ năm 2015

#Gom nhóm tuổi vào các bins
bins = [0, 29, 39, 49, 59, 120]
labels = ['<30', '30-39', '40-49', '50-59', '60+']
df_customer['Age_Group'] = pd.cut(df_customer['Customer_Age'], bins=bins, labels=labels)

#Kiểm tra lại
print(df_customer.isnull().sum()[df_customer.isnull().sum() > 0]) # có 1 ông cháu bị out (tuổi 120+)
print(df_customer[df_customer.isnull().any(axis=1)])  

# bỏ ông cháu có tuổi bị lệch quá xa
df_customer.dropna(subset=['Age_Group'], inplace=True)

# Thống kê tuổi bằng biểu đồ
# Counting the number or sum of each age group
age_counts = df_customer['Age_Group'].value_counts().sort_index()

# Barchart visualization of customer age groups
# Creating the barchart size and figure
plt.figure(figsize=(10,8))
ax = sns.barplot(x=age_counts.index, y=age_counts.values, palette='pastel')

# Adding 'sum' and 'percentage' labels inside the barchart
total = age_counts.sum()
for i, count in enumerate(age_counts):
    percentage = (count / total) * 100
    plt.text(i, count + 5, f'{count} ({percentage:.2f}%)', ha='center', va='bottom', fontsize=12)

# Adding titles for the barchart
plt.title("Customer Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# Tạo cột tổng chi tiêu 
df_customer['Total_Spending'] = df_customer[['MntWines', 'MntFruits', 'MntMeatProducts',
                                              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
# Thêm cột Children bằng cách cộng Kidhome và Teenhome
df_customer['Children'] = df_customer['Kidhome'] + df_customer['Teenhome']
# kiểm tra lại dataset
print(df_customer.shape) # 2230 x 33
print(df_customer.info()) 
print(df_customer.head())

# xóa các cột không cần thiết
columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue', 'Kidhome', 'Teenhome']
df_customer.drop(columns=columns_to_drop, axis=1, inplace=True)

print(df_customer.shape) 
print(df_customer.info()) 
print(df_customer.head())

# Lưu dataset đã làm sạch
df_customer.to_csv("dataset/data_cleaning/cleaned_dataset.csv", index=False)
print("✅ Saved cleaned dataset!")

# Phân loại các kiểu dữ liệu của các feature
numerical_variables = [
    'Income', 'Customer_Age',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency',
    'Total_Spending', 'Children'
]

# Relevant Categorical variables
categorical_variables = ['Education', 'Marital_Status', 'Age_Group']



def thousands_formatter(x, pos):
    abs_x = abs(x)
    if abs_x >= 1_000_000:
        formatted = f'{abs_x / 1_000_000:.0f}M'
    elif abs_x >= 1_000:
        formatted = f'{abs_x / 1_000:.0f}K'
    else:
        formatted = f'{int(abs_x)}'

    return f'-{formatted}' if x < 0 else formatted

formatter = FuncFormatter(thousands_formatter)
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Số boxplot mỗi figure
plots_per_figure = 3

# Tổng số biến số
total_plots = len(numerical_variables)

# Số figure cần tạo
num_figures = math.ceil(total_plots / plots_per_figure)

for fig_idx in range(num_figures):
    start = fig_idx * plots_per_figure
    end = min(start + plots_per_figure, total_plots)
    current_vars = numerical_variables[start:end]
    
    # Tạo figure mới
    fig, axes = plt.subplots(
        nrows=1, ncols=len(current_vars), figsize=(6 * len(current_vars), 5)
    )
    
    # Nếu chỉ có 1 subplot thì axes không phải list
    if len(current_vars) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, current_vars):
        sns.boxplot(data=df_customer, x=col, ax=ax)
        
        # Tính ngưỡng IQR
        Q1 = df_customer[col].quantile(0.25)
        Q3 = df_customer[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Vẽ ngưỡng
        ax.axvline(lower_bound, color="red", linestyle="--", label="Lower Bound")
        ax.axvline(upper_bound, color="blue", linestyle="--", label="Upper Bound")
        
        ax.set_title(f"{col}", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(formatter)
    
    # Chỉ thêm legend cho plot đầu tiên trong figure
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    plt.show()
