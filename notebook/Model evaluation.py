# ==========================================
# MODEL EVALUATION
# ==========================================

import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

df = pd.read_csv(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\processed_bank_data.csv"
)

from sklearn.model_selection import train_test_split

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = joblib.load(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\all_models.pkl"
)

scaler = joblib.load(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\scaler.pkl"
)

X_test_scaled = scaler.transform(X_test)

results = []

for name, model in models.items():

    if name == "Logistic Regression":

        pred = model.predict(X_test_scaled)

        prob = model.predict_proba(
            X_test_scaled
        )[:,1]

    else:

        pred = model.predict(X_test)

        prob = model.predict_proba(
            X_test
        )[:,1]

    results.append([

        name,

        accuracy_score(y_test,pred),

        precision_score(y_test,pred),

        recall_score(y_test,pred),

        f1_score(y_test,pred),

        roc_auc_score(y_test,prob)

    ])

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

print(results_df)

results_df.to_csv(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\model_comparison.csv",
    index=False
)

# Random Forest Evaluation

rf = models["Random Forest"]

pred = rf.predict(X_test)

cm = confusion_matrix(
    y_test,
    pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Random Forest Confusion Matrix")

plt.show()

# Feature Importance

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
    rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(15))

importance.to_csv(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\feature_importance.csv",
    index=False
)

print("Evaluation Completed")
