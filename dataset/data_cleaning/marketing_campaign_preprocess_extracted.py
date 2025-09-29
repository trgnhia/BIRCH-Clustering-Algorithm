# ------------------------------------------------------------
# Extracted from marketing-campaign-preprocess.ipynb
# This file concatenates all Python code cells in order.
# IPython magics and shell escapes have been commented out.
# ------------------------------------------------------------

# In[1]:
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

# In[2]:
# Tải dataset lên notebook Kaggle
df_customer = pd.read_csv('/kaggle/input/marketingcampaign/marketing_campaign.csv', sep='\t')

# Hiển thị 5 dòng đầu tiên của dataset
df_customer.head()

# In[3]:
# Hiển thị các tên cột của dataset
col_names = df_customer.columns
col_names

# In[4]:
# Hiển thị tóm tắt và kích thước của dataset
df_customer.info()
df_customer.shape

# In[5]:
# Đặt timeframe là ngày-tháng-năm với Python
df_customer['Dt_Customer'] = pd.to_datetime(df_customer['Dt_Customer'], dayfirst=True)

# Kiểm tra phạm vi ngày tháng đăng ký của khách hàng
print("Earliest date of customer enrollment is", df_customer['Dt_Customer'].min())
print("Latest date of customer enrollment is", df_customer['Dt_Customer'].max())

# In[6]:
# Đếm số lượng khách hàng đăng ký theo từng năm từ 2012 đến 2014
df_customer['Dt_Customer'].dt.year.value_counts().sort_index()

# In[7]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Year_Birth"
df_customer["Year_Birth"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[8]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Education"
df_customer["Education"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[9]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Marital_Status"
df_customer["Marital_Status"].value_counts(normalize=True).mul(100).round(2).sort_index()

# Kiểm tra các kết quả bên dưới, có một số giá trị bất thường của Marital_Status
# Chẳng hạn, Absurd, Alone và YOLO không có ý nghĩa gì cả

# In[10]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Income"
df_customer["Income"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[11]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Kidhome"
df_customer["Kidhome"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[12]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Teenhome"
df_customer["Teenhome"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[13]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Dt_Customer"
df_customer["Dt_Customer"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[14]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Recency"
df_customer["Recency"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[15]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Complain"
df_customer["Complain"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[16]:
# Hiển thị tỉ lệ của từng giá trị trong trường "MntWines"
df_customer["MntWines"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[17]:
# Hiển thị tỉ lệ của từng giá trị trong trường "MntFruits"
df_customer["MntFruits"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[18]:
# Hiển thị tỉ lệ của từng giá trị trong trường "MntMeatProducts"
df_customer["MntMeatProducts"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[19]:
# Hiển thị tỉ lệ của từng giá trị trong trường "MntFishProducts"
df_customer["MntFishProducts"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[20]:
# Hiển thị tỉ lệ của từng giá trị trong trường "MntSweetProducts"
df_customer["MntSweetProducts"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[21]:
# Hiển thị tỉ lệ của từng giá trị trong trường "MntGoldProds"
df_customer["MntGoldProds"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[22]:
# Hiển thị tỉ lệ của từng giá trị trong trường "NumDealsPurchases"
df_customer["NumDealsPurchases"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[23]:
# Hiển thị tỉ lệ của từng giá trị trong trường "AcceptedCmp1"
df_customer["AcceptedCmp1"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[24]:
# Hiển thị tỉ lệ của từng giá trị trong trường "AcceptedCmp2"
df_customer["AcceptedCmp2"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[25]:
# Hiển thị tỉ lệ của từng giá trị trong trường "AcceptedCmp3"
df_customer["AcceptedCmp3"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[26]:
# Hiển thị tỉ lệ của từng giá trị trong trường "AcceptedCmp4"
df_customer["AcceptedCmp4"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[27]:
# Hiển thị tỉ lệ của từng giá trị trong trường "AcceptedCmp5"
df_customer["AcceptedCmp5"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[28]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Response"
df_customer["Response"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[29]:
# Hiển thị tỉ lệ của từng giá trị trong trường "NumWebPurchases"
df_customer["NumWebPurchases"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[30]:
# Hiển thị tỉ lệ của từng giá trị trong trường "NumCatalogPurchases"
df_customer["NumCatalogPurchases"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[31]:
# Hiển thị tỉ lệ của từng giá trị trong trường "NumStorePurchases"
df_customer["NumStorePurchases"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[32]:
# Hiển thị tỉ lệ của từng giá trị trong trường "NumWebVisitsMonth"
df_customer["NumWebVisitsMonth"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[33]:
df_customer[["Year_Birth", "Income", "Kidhome", "Teenhome", 
             "Recency", "MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds",
            "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
            "NumStorePurchases", "NumWebVisitsMonth"]].describe().T.round(2)

# In[34]:
# Kiểm tra các bản ghi trùng lặp dựa trên trường ID
# keep=False để đánh dấu tất cả các bản ghi trùng lặp
duplicated_value = df_customer.duplicated(subset=["ID"], keep=False)
df_customer_duplicate = df_customer[duplicated_value].sort_values(by=["ID"])

# Hiển thị các bản ghi trùng lặp
df_customer_duplicate

# In[35]:
# Kiểm tra các cột thiếu dữ liệu và đếm số ô bị thiếu dữ liệu
df_customer.isnull().sum()[df_customer.isnull().sum() > 0]

# In[36]:
# Hiển thị các bản ghi có giá trị bị thiếu
df_customer[df_customer.isnull().any(axis=1)]

# In[37]:
median_income = df_customer["Income"].median()
df_customer["Income"] = df_customer["Income"].fillna(median_income)

# In[38]:
# Kiểm tra lại xem còn giá trị bị thiếu không
df_customer.isnull().sum()[df_customer.isnull().sum() > 0]

# In[39]:
df_customer.info()

# In[40]:
# Hiển thị các dòng chứa giá trị bất thường trong trường Marital_Status
unclear_marital_status = ['Absurd', 'Alone', 'YOLO']
df_unclear_marital_status = df_customer[df_customer['Marital_Status'].isin(unclear_marital_status)]
print(df_unclear_marital_status)

# In[41]:
# Đếm số lượng các giá trị bất thường trong trường Marital_Status
print(df_unclear_marital_status['Marital_Status'].value_counts())

# In[42]:
# Chuyển 7 giá trị bất thường thành Single
df_customer['Marital_Status'] = df_customer['Marital_Status'].replace({
    'Absurd': 'Single',
    'Alone': 'Single',
    'YOLO': 'Single'
})

# In[43]:
# Kiểm tra lại
df_unclear_marital_status = df_customer[df_customer['Marital_Status'].isin(['Absurd', 'Alone', 'YOLO'])]
print(df_unclear_marital_status['Marital_Status'].value_counts())

# In[44]:
# Thêm Customer_Age là một trường mới, lấy mốc thời gian xử lý dữ liệu là 2015 (ngay sau thời điểm đăng ký khách hàng mới nhất là 2014)
df_customer["Customer_Age"] = 2015 - df_customer["Year_Birth"]

# In[45]:
# Tính toán Customer_Age và sắp xếp chúng từ nhỏ đến lớn
df_customer['Customer_Age'].value_counts().sort_index()

# In[46]:
# Tính toán Year_Birth và sắp xếp chúng từ nhỏ đến lớn
df_customer['Year_Birth'].value_counts().sort_index()

# In[47]:
# Thêm cột Age_Group để phân loại khách hàng theo nhóm tuổi
bins = [0, 29, 39, 49, 59, 120]
labels = ['<30', '30–39', '40–49', '50–59', '60+']
df_customer['Age_Group'] = pd.cut(df_customer['Customer_Age'], bins=bins, labels=labels)

# In[48]:
df_customer.isnull().sum()[df_customer.isnull().sum() > 0]

# In[49]:
df_customer[df_customer.isnull().any(axis=1)]

# In[50]:
df_customer.dropna(subset=['Age_Group'], inplace=True)

# In[51]:
df_customer[df_customer.isnull().any(axis=1)]

# In[52]:
# Dataset hiện tại sau khi loại bỏ 3 dòng có giá trị NaN
df_customer.info()

# In[53]:
# Đếm số lượng của từng nhóm tuổi
age_counts = df_customer['Age_Group'].value_counts().sort_index()

# Biểu đồ thanh thể hiện các nhóm tuổi của khách hàng
plt.figure(figsize=(10,8))
bars = plt.bar(age_counts.index, age_counts.values, color='skyblue', alpha=0.8)

# Thêm giá trị và phần trăm vào từng thanh của biểu đồ
total = age_counts.sum()
for i, count in enumerate(age_counts):
    percentage = (count / total) * 100
    plt.text(i, count + 5, f'{count} ({percentage:.2f}%)', ha='center', va='bottom', fontsize=12)

# Thêm tiêu đề và nhãn cho biểu đồ
plt.title("Customer Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# In[54]:
df_customer['Total_Spending'] = df_customer[['MntWines', 'MntFruits', 'MntMeatProducts',
                                              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# In[55]:
# Hiển thị tỉ lệ của từng giá trị trong trường "Total_Spending"
df_customer["Total_Spending"].value_counts(normalize=True).mul(100).round(2).sort_index()

# In[56]:
# Tạo biến mới là Children bằng cách cộng Kidhome và Teenhome lại với nhau
df_customer['Children'] = df_customer['Kidhome'] + df_customer['Teenhome']

# In[57]:
# Chi tiết về dataset sau khi tiền xử lý
df_customer.info()

# In[58]:
df_customer.head()

# In[59]:
# Các biến số sẽ kiểm tra như sau:
numerical_variables = [
    'Income', 'Customer_Age',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency',
    'Total_Spending', 'Children'
]

# Các biến phân loại sẽ kiểm tra như sau:
categorical_variables = ['Education', 'Marital_Status', 'Age_Group']

# In[60]:
# Hàm định dạng số K và M (bao gồm cả giá trị âm)
# Ví dụ: chuyển 100,000 thành 100k và 1,000,000 thành 1M

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

# Danh sách các biến số
numerical_variables = ['Income', 'Customer_Age',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency',
    'Total_Spending', 'Children']

# Tạo biểu đồ hộp cho từng biến số
rows, cols = 5, 3 # (5 hàng x 3 cột)
plt.figure(figsize=(20, 20))

# Lặp lại qua từng biến để vẽ biểu đồ hộp
for i, col in enumerate(numerical_variables):
    ax = plt.subplot(rows, cols, i + 1)
    sns.boxplot(data=df_customer, x=col, ax=ax)

    # Tính toán IQR để xác định ngưỡng ngoại lệ
    Q1 = df_customer[col].quantile(0.25)
    Q3 = df_customer[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Thêm các đường thẳng dọc để biểu thị ngưỡng ngoại lệ
    ax.axvline(lower_bound, color="red", linestyle="--", label="Lower Bound")
    ax.axvline(upper_bound, color="blue", linestyle="--", label="Upper Bound")   
    
    ax.set_title(f"Distribution of {col} Variable", fontsize=15, fontweight="bold")
    ax.set_xlabel(col, fontsize=13)
    ax.set_ylabel("")  
    ax.xaxis.set_major_formatter(formatter)  # Áp dụng định dạng K và M cho trục x

    # Thêm chú thích chỉ cho biểu đồ đầu tiên để tránh lặp lại
    if i == 0:
        ax.legend(fontsize=11)

plt.tight_layout()
plt.show()

# In[61]:
# Danh sách các biến số
numerical_variables = ['Income', 'Customer_Age',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency',
    'Total_Spending', 'Children']

# Tạo biểu đồ histogram cho từng biến số
rows, cols = 5, 3 # (5 hàng x 3 cột)
plt.figure(figsize=(20, 20))

# Lặp lại qua từng biến để vẽ biểu đồ histogram
for i, col in enumerate(numerical_variables):
    ax = plt.subplot(rows, cols, i + 1)
    sns.histplot(data=df_customer, x=col, kde=True, bins=30, ax=ax, color='steelblue', edgecolor='black')
    
    ax.set_title(f'Distribution of {col} Variable', fontsize=15)
    ax.set_xlabel(col, fontsize=13)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=10)

    # Áp dụng định dạng số K và M (bao gồm cả giá trị âm)
    # Ví dụ: chuyển 100,000 thành 100k và 1,000,000 thành 1M
    ax.xaxis.set_major_formatter(formatter)

# Điều chỉnh bố cục để tránh chồng chéo
plt.tight_layout()
plt.show()

# In[62]:
# Đầu tiên, xác định các biến số cần kiểm tra
columns_to_check = [
    'Income', 'Customer_Age', 'MntWines', 'MntFruits',
    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
    'NumWebVisitsMonth', 'Recency', 'Total_Spending', 'Children'
]

# Hàm tính toán tỷ lệ phần trăm giá trị ngoại lai sử dụng phương pháp IQR
def calculate_outlier_percentage(df_customer, columns):
    outlier_data = []
    for col in columns:
        Q1 = df_customer[col].quantile(0.25)
        Q3 = df_customer[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_customer[(df_customer[col] < lower_bound) | (df_customer[col] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_percentage = (outlier_count / len(df_customer)) * 100
        outlier_data.append({
            'Variable': col,
            'Outlier Count': outlier_count,
            'Outlier Percentage (%)': round(outlier_percentage, 2)
        })
    return pd.DataFrame(outlier_data)

# Hiển thị tỷ lệ phần trăm giá trị ngoại lai cho các biến số đã chọn
outlier_stats = calculate_outlier_percentage(df_customer, columns_to_check)
print(outlier_stats)

# In[63]:
# Biến số cần lọc giá trị ngoại lai
# Các biến có hơn 5% giá trị ngoại lai bên trong chúng
variables_to_filter = [
    'MntFruits',
    'MntMeatProducts',
    'MntSweetProducts',
    'MntFishProducts',
    'MntGoldProds'
]

# Hàm Winsorization để lọc giá trị ngoại lai (thay thế các giá trị cực đoan bằng các giá trị ít cực đoan hơn)
def winsorize_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

# Áp dụng hàm Winsorization cho các biến số đã chọn
for col in variables_to_filter:
    winsorize_outliers(df_customer, col)

# In[64]:
# Kiểm tra lại xem còn giá trị ngoại lai không
for col in variables_to_filter:
    Q1 = df_customer[col].quantile(0.25)
    Q3 = df_customer[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    below_lower_bound = (df_customer[col] < lower_bound).sum()
    above_upper_bound = (df_customer[col] > upper_bound).sum()
    
    # Nếu không có giá trị ngoại lai, in ra thông báo
    print(f"{col}: Below lower bound = {below_lower_bound}, Above upper bound = {above_upper_bound}")

# In[65]:
df_customer.info()

# In[66]:
print(df_customer.columns.tolist())

# In[67]:
# Loại bỏ các cột không sử dụng cho phân tích và phân cụm dữ liệu
columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue', 'Kidhome', 'Teenhome']
df_customer.drop(columns=columns_to_drop, axis=1, inplace=True)

# In[68]:
# Copy và lưu dataframe đã được làm sạch và xử lý vào một dataframe mới
df_customer_filtered = df_customer.copy()

# In[69]:
df_customer_filtered.columns

# In[70]:
df_customer_filtered.info()

# In[71]:
df_customer_filtered.head()

# In[72]:
# Lưu dataset đã xử lý thành file CSV mới
df_customer_filtered.to_csv('/kaggle/working/customer_data_processed.csv', index=False)

# Hiển thị thông báo xác nhận
print("Dataset đã được lưu thành công!")
print(f"Tên file: customer_data_processed.csv")
print(f"Số lượng dòng: {df_customer_filtered.shape[0]:,}")
print(f"Số lượng cột: {df_customer_filtered.shape[1]:,}")
print(f"Kích thước file: {df_customer_filtered.memory_usage(deep=True).sum() / 1024:.2f} KB")

# In[73]:
# Các biến số sẽ sử dụng cho phân cụm dữ liệu
cluster_a_df = df_customer_filtered[['Age_Group', 'MntWines', 'MntFruits', 'MntMeatProducts',
                                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

# In[74]:
# Xem các giá trị duy nhất và tính tổng số giá trị của từng Products

summary = pd.DataFrame()

for col in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']:
    counts = df_customer_filtered[col].value_counts().sort_index()
    summary = pd.concat([summary, counts], axis=1)

summary.columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
print(summary)

# In[75]:
# Vì Age_group là một biến phân loại, tôi phải chuyển chúng thành số
# Python chỉ có thể 'đọc' các biến số, không phải biến phân loại
# Do đó, tôi sẽ biến đổi Age_Group thành các số bằng cách mã hóa các giá trị của nó

# Mã hóa Age_Group trong dataset df_customer_filtered
le = LabelEncoder()
df_customer_filtered['Age_Group_Encoded'] = le.fit_transform(df_customer_filtered['Age_Group'])

# Vì tôi đã có phiên bản mã hóa của Age_Group là Age_Group_Encoded, nên tôi sẽ sửa đổi các biến đã chọn cho phân cụm
cluster_a_df = df_customer_filtered[['Age_Group_Encoded', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

# In[76]:
# Tiếp theo tôi sẽ chuẩn hóa cluster_a_df
scaler = StandardScaler()
cluster_a_scaled = scaler.fit_transform(cluster_a_df)

