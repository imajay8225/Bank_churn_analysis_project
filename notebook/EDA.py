# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\European_Bank.csv"

df = pd.read_csv(file_path)

# Churn Distribution

plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Geography vs Churn

plt.figure(figsize=(8,4))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title("Geography vs Churn")
plt.show()

# Gender vs Churn

plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Exited', data=df)
plt.title("Gender vs Churn")
plt.show()

# Active Member vs Churn

plt.figure(figsize=(6,4))
sns.countplot(x='IsActiveMember', hue='Exited', data=df)
plt.title("Active Member vs Churn")
plt.show()

# Products vs Churn

plt.figure(figsize=(6,4))
sns.countplot(x='NumOfProducts', hue='Exited', data=df)
plt.title("Products vs Churn")
plt.show()

# Age vs Churn

plt.figure(figsize=(6,4))
sns.boxplot(x='Exited', y='Age', data=df)
plt.title("Age vs Churn")
plt.show()

# Balance vs Churn

plt.figure(figsize=(6,4))
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title("Balance vs Churn")
plt.show()

# Correlation Heatmap

df_corr = df.copy()

df_corr = pd.get_dummies(
    df_corr,
    columns=['Geography','Gender'],
    drop_first=True
)

plt.figure(figsize=(14,8))

sns.heatmap(
    df_corr.corr(numeric_only=True),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()
