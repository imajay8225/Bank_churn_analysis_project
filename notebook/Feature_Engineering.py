# ==========================================
# FEATURE ENGINEERING
# ==========================================

import pandas as pd

file_path = r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\European_Bank.csv"

df = pd.read_csv(file_path)

# Remove useless columns

df.drop(
    columns=[
        "Year",
        "CustomerId",
        "Surname"
    ],
    inplace=True
)

# Encoding

df = pd.get_dummies(
    df,
    columns=[
        "Geography",
        "Gender"
    ],
    drop_first=True
)

# Feature Engineering

df["BalanceSalaryRatio"] = (
    df["Balance"] /
    (df["EstimatedSalary"] + 1)
)

df["ProductDensity"] = (
    df["NumOfProducts"] /
    (df["Tenure"] + 1)
)

df["EngagementProduct"] = (
    df["IsActiveMember"] *
    df["NumOfProducts"]
)

df["AgeTenure"] = (
    df["Age"] *
    df["Tenure"]
)

df["BalancePerProduct"] = (
    df["Balance"] /
    (df["NumOfProducts"] + 1)
)

print(df.head())

df.to_csv(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\processed_bank_data.csv",
    index=False
)

print("Processed Dataset Saved")
