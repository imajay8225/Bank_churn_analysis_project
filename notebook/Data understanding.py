# ==========================================
# NOTEBOOK 01 : DATA UNDERSTANDING
# BANK CUSTOMER CHURN PREDICTION
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np

# Display Settings
pd.set_option('display.max_columns', None)

# ==========================================
# LOAD DATASET
# ==========================================

file_path = r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\European_Bank.csv"

df = pd.read_csv(file_path)

print("="*60)
print("DATASET LOADED SUCCESSFULLY")
print("="*60)

# ==========================================
# FIRST LOOK
# ==========================================

print("\nFIRST 5 RECORDS")
print(df.head())

print("\nLAST 5 RECORDS")
print(df.tail())

# ==========================================
# DATASET SHAPE
# ==========================================

print("\n" + "="*60)
print("DATASET SHAPE")
print("="*60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ==========================================
# COLUMN NAMES
# ==========================================

print("\n" + "="*60)
print("COLUMN NAMES")
print("="*60)

print(df.columns.tolist())

# ==========================================
# DATASET INFO
# ==========================================

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)

print(df.info())

# ==========================================
# STATISTICAL SUMMARY
# ==========================================

print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print(df.describe())

# ==========================================
# DATA TYPES
# ==========================================

print("\n" + "="*60)
print("DATA TYPES")
print("="*60)

print(df.dtypes)

# ==========================================
# MISSING VALUES
# ==========================================

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)

print(df.isnull().sum())

print("\nTotal Missing Values :",
      df.isnull().sum().sum())

# ==========================================
# DUPLICATE RECORDS
# ==========================================

print("\n" + "="*60)
print("DUPLICATE RECORDS")
print("="*60)

duplicates = df.duplicated().sum()

print("Duplicate Records :", duplicates)

# ==========================================
# UNIQUE VALUES
# ==========================================

print("\n" + "="*60)
print("UNIQUE VALUES IN EACH COLUMN")
print("="*60)

for col in df.columns:
    print(f"{col} : {df[col].nunique()}")

# ==========================================
# TARGET VARIABLE ANALYSIS
# ==========================================

print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS")
print("="*60)

print(df["Exited"].value_counts())

print("\nPercentage Distribution")

print(
    round(
        df["Exited"].value_counts(normalize=True)*100,
        2
    )
)

# ==========================================
# CHURN RATE
# ==========================================

print("\n" + "="*60)
print("CUSTOMER CHURN RATE")
print("="*60)

churn_rate = round(
    df["Exited"].mean()*100,
    2
)

print(f"Customer Churn Rate : {churn_rate}%")

# ==========================================
# CATEGORICAL FEATURES
# ==========================================

print("\n" + "="*60)
print("CATEGORICAL FEATURES")
print("="*60)

categorical_cols = df.select_dtypes(
    include="object"
).columns.tolist()

print(categorical_cols)

# ==========================================
# NUMERICAL FEATURES
# ==========================================

print("\n" + "="*60)
print("NUMERICAL FEATURES")
print("="*60)

numerical_cols = df.select_dtypes(
    exclude="object"
).columns.tolist()

print(numerical_cols)

# ==========================================
# GEOGRAPHY DISTRIBUTION
# ==========================================

print("\n" + "="*60)
print("GEOGRAPHY DISTRIBUTION")
print("="*60)

print(df["Geography"].value_counts())

# ==========================================
# GENDER DISTRIBUTION
# ==========================================

print("\n" + "="*60)
print("GENDER DISTRIBUTION")
print("="*60)

print(df["Gender"].value_counts())

# ==========================================
# PROJECT SUMMARY
# ==========================================

print("\n" + "="*60)
print("DATA UNDERSTANDING SUMMARY")
print("="*60)

print(f"""
Dataset Name      : European Bank Customer Churn Dataset
Total Records     : {df.shape[0]}
Total Columns     : {df.shape[1]}
Missing Values    : {df.isnull().sum().sum()}
Duplicate Records : {duplicates}
Target Variable   : Exited
Churn Rate        : {churn_rate}%

Business Objective:
Predict customer churn and identify high-risk
customers to improve customer retention strategies.
""")
