# ==========================================
# MODEL TRAINING
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

df = pd.read_csv(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\processed_bank_data.csv"
)

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {

    "Logistic Regression":
    LogisticRegression(
        class_weight='balanced',
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
        class_weight='balanced',
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(),

    "XGBoost":
    XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

for name, model in models.items():

    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

    else:

        model.fit(
            X_train,
            y_train
        )

print("All Models Trained Successfully")

joblib.dump(
    models,
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\all_models.pkl"
)

joblib.dump(
    scaler,
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\scaler.pkl"
)

print("Models Saved")# ==========================================
# MODEL TRAINING
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

df = pd.read_csv(
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\data\processed_bank_data.csv"
)

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {

    "Logistic Regression":
    LogisticRegression(
        class_weight='balanced',
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
        class_weight='balanced',
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(),

    "XGBoost":
    XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

for name, model in models.items():

    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

    else:

        model.fit(
            X_train,
            y_train
        )

print("All Models Trained Successfully")

joblib.dump(
    models,
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\all_models.pkl"
)

joblib.dump(
    scaler,
    r"C:\Users\AJAY PRATAP SINGH\OneDrive\Desktop\Bank_Churn_Project\models\scaler.pkl"
)

print("Models Saved")
