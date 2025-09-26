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
# Uploading the dataset into the Kaggle notebook
df_customer = pd.read_csv('dataset/origin/marketing_campaign.csv', sep='\t')

#Xử lý income = NaN lấy trung vị bù vào
median_income = df_customer["Income"].median()
df_customer["Income"] = df_customer["Income"].fillna(median_income)

# Absorbing 7 abnormal Marital_Status into 'Single'
df_customer['Marital_Status'] = df_customer['Marital_Status'].replace({
    'Absurd': 'Single',
    'Alone': 'Single',
    'YOLO': 'Single'
})

# Add Customer_Age as a new variable (column), based on current year 2025
df_customer["Customer_Age"] = 2015 - df_customer["Year_Birth"]

# Add variable (column) of Age Group into the chart (binning age groups)
bins = [0, 29, 39, 49, 59, 120]
labels = ['<30', '30-39', '40-49', '50-59', '60+']
df_customer['Age_Group'] = pd.cut(df_customer['Customer_Age'], bins=bins, labels=labels)
# xóa các items có tuổi bị lệch quá xa ( ngoài age_Group)
df_customer.dropna(subset=['Age_Group'], inplace=True)

# # Counting the number or sum of each age group
# age_counts = df_customer['Age_Group'].value_counts().sort_index()

# # Barchart visualization of customer age groups
# # Creating the barchart size and figure
# plt.figure(figsize=(10,8))
# ax = sns.barplot(x=age_counts.index, y=age_counts.values, palette='pastel')

# # Adding 'sum' and 'percentage' labels inside the barchart
# total = age_counts.sum()
# for i, count in enumerate(age_counts):
#     percentage = (count / total) * 100
#     plt.text(i, count + 5, f'{count} ({percentage:.2f}%)', ha='center', va='bottom', fontsize=12)

# # Adding titles for the barchart
# plt.title("Customer Distribution by Age Group")
# plt.xlabel("Age Group")
# plt.ylabel("Number of Customers")
# plt.tight_layout()
# plt.show()

df_customer['Total_Spending'] = df_customer[['MntWines', 'MntFruits', 'MntMeatProducts',
                                              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Add new variable called Children by combining variables Kidhome and Teenhome 
df_customer['Children'] = df_customer['Kidhome'] + df_customer['Teenhome']

# Variables that have more than 5% of outliers inside them
variables_to_filter = [
    'MntFruits',
    'MntMeatProducts',
    'MntSweetProducts',
    'MntFishProducts',
    'MntGoldProds'
]

# Winsorization function (replacing extreme values with less extreme ones)
def winsorize_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

# Apply Winsorization function to selected variables/columns
for col in variables_to_filter:
    winsorize_outliers(df_customer, col)
# Double check the outliers to see if Winsorization technique works or not
for col in variables_to_filter:
    Q1 = df_customer[col].quantile(0.25)
    Q3 = df_customer[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    below_lower_bound = (df_customer[col] < lower_bound).sum()
    above_upper_bound = (df_customer[col] > upper_bound).sum()
    
    # If the results for both below and upper are 0, then the technique worked
    print(f"{col}: Below lower bound = {below_lower_bound}, Above upper bound = {above_upper_bound}")
# Dropping variables/columns that are no longer needed (irrelevant)
columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue', 'Kidhome', 'Teenhome']
df_customer.drop(columns=columns_to_drop, axis=1, inplace=True)
df_customer.to_csv("dataset/data_cleaning/cleaned_dataset.csv", index=False)
print("✅ Saved cleaned dataset!")