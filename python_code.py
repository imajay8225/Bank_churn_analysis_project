# ==============================
# BANK CUSTOMER CHURN PREDICTION
# COMPLETE PROJECT CODE
# ==============================

# ------------------------------
# IMPORT LIBRARIES
# ------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import joblib

# ------------------------------
# LOAD DATASET
# ------------------------------

file_path = r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\European_Bank.csv"

df = pd.read_csv(file_path)

print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("="*50)
print("FIRST 5 ROWS")
print(df.head())

print("="*50)
print("MISSING VALUES")
print(df.isnull().sum())

print("="*50)
print("DUPLICATES")
print(df.duplicated().sum())

# ------------------------------
# CHURN RATE
# ------------------------------

churn_rate = df["Exited"].mean()*100

print("="*50)
print(f"CHURN RATE = {churn_rate:.2f}%")

# ------------------------------
# REMOVE USELESS COLUMNS
# ------------------------------

df.drop(
    columns=["Year", "CustomerId", "Surname"],
    inplace=True,
    errors="ignore"
)

# ------------------------------
# EDA
# ------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x="Geography", hue="Exited", data=df)
plt.title("Geography vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Gender", hue="Exited", data=df)
plt.title("Gender vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="IsActiveMember", hue="Exited", data=df)
plt.title("Active Member vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="NumOfProducts", hue="Exited", data=df)
plt.title("Products vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Age vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance vs Churn")
plt.show()

# ------------------------------
# ENCODING
# ------------------------------

df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------

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

# ------------------------------
# CORRELATION HEATMAP
# ------------------------------

plt.figure(figsize=(14,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# FEATURES & TARGET
# ------------------------------

X = df.drop("Exited", axis=1)
y = df["Exited"]

# ------------------------------
# TRAIN TEST SPLIT
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------
# SCALING
# ------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# MODELS
# ------------------------------

models = {

    "Logistic Regression":
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),

    "Decision Tree":
    DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(),

    "XGBoost":
    XGBClassifier(
        eval_metric="logloss"
    )
}

# ------------------------------
# MODEL EVALUATION
# ------------------------------

results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    if name == "Logistic Regression":

        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)

        prob = model.predict_proba(
            X_test_scaled
        )[:,1]

    else:

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        prob = model.predict_proba(
            X_test
        )[:,1]

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, prob)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

# ------------------------------
# RESULTS TABLE
# ------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

# ------------------------------
# BEST MODEL
# ------------------------------

best_model = models["Random Forest"]

# ------------------------------
# CONFUSION MATRIX
# ------------------------------

pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ------------------------------
# FEATURE IMPORTANCE
# ------------------------------

importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance":
    best_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP FEATURES")
print(importance.head(15))

importance.to_csv(
    "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.show()

# ------------------------------
# RISK SCORING
# ------------------------------

risk_score = best_model.predict_proba(X)[:,1]

def risk_category(x):

    if x < 0.30:
        return "Low"

    elif x < 0.70:
        return "Medium"

    else:
        return "High"

df["RiskScore"] = risk_score

df["RiskLevel"] = df["RiskScore"].apply(
    risk_category
)

# ------------------------------
# EXPORT FINAL DATA
# ------------------------------

output_path = r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\Final_Churn_Risk_Data.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nFinal Dataset Exported")

# ------------------------------
# SAVE MODEL
# ------------------------------

joblib.dump(
    best_model,
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\churn_model.pkl"
)

joblib.dump(
    scaler,
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\scaler.pkl"
)

print("\nModel Saved Successfully")
print("\nPROJECT COMPLETED SUCCESSFULLY")
