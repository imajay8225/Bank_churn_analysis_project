# 🏦 Bank Customer Churn Prediction Using Machine Learning and Explainable AI

<div align="center">

### Predicting Customer Attrition in Retail Banking with Interpretable Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC4B25?style=flat-square)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=flat-square)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#-license)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)]()

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-1-project-overview)
2. [Business Problem](#-2-business-problem)
3. [Objectives](#-3-objectives)
4. [Dataset Description](#-4-dataset-description)
5. [Technology Stack](#-5-technology-stack)
6. [Project Architecture](#-6-project-architecture)
7. [Exploratory Data Analysis](#-7-exploratory-data-analysis)
8. [Feature Engineering](#-8-feature-engineering)
9. [Machine Learning Models](#-9-machine-learning-models)
10. [Model Evaluation Metrics](#-10-model-evaluation-metrics)
11. [Explainable AI (SHAP)](#-11-explainable-ai-shap)
12. [Streamlit Dashboard](#-12-streamlit-dashboard)
13. [Project Workflow](#-13-project-workflow)
14. [Key Findings and Insights](#-14-key-findings-and-insights)
15. [Business Recommendations](#-15-business-recommendations)
16. [Installation Guide](#-16-installation-guide)
17. [Usage Instructions](#-17-usage-instructions)
18. [Folder Structure](#-18-folder-structure)
19. [Future Enhancements](#-19-future-enhancements)
20. [Author Information](#-20-author-information)
21. [License](#-21-license)
22. [Acknowledgements](#-22-acknowledgements)

---

## 📖 1. Project Overview

**Bank Customer Churn Prediction** is an end-to-end machine learning solution designed to identify retail banking customers who are at risk of closing their accounts (churning). The project goes beyond simple prediction — it incorporates **Explainable AI (XAI)** using **SHAP (SHapley Additive exPlanations)** to make model decisions transparent and business-interpretable, and ships with an **interactive Streamlit dashboard** for real-time risk scoring and what-if scenario analysis.

This project was developed as part of an **MBA Business Analytics capstone**, bridging the gap between technical machine learning implementation and actionable business strategy in the banking and financial services domain.

> 💡 **Why this matters:** Acquiring a new customer can cost banks 5–7x more than retaining an existing one. A reliable, explainable churn model empowers retention teams to act *before* a customer leaves — not after.

---

## 💼 2. Business Problem

Customer attrition is a critical concern for retail banks operating in highly competitive and saturated markets. Banks often discover a customer has churned only *after* the account has been closed, by which point retention efforts are no longer possible.

**Core challenges addressed by this project:**

| Challenge | Impact |
|---|---|
| Inability to identify at-risk customers early | Lost revenue and lifetime value |
| Lack of interpretability in "black-box" ML models | Low trust and adoption by business teams |
| No standardized risk scoring mechanism | Inefficient, untargeted retention campaigns |
| High cost of new customer acquisition | Shrinking profit margins |

This project addresses these gaps by building a **predictive, explainable, and actionable** churn detection system.

---

## 🎯 3. Objectives

- ✅ Predict the likelihood of customer churn using historical banking data
- ✅ Classify customers into risk segments (Low / Medium / High) for targeted intervention
- ✅ Compare multiple machine learning algorithms to identify the best-performing model
- ✅ Explain individual and global model predictions using SHAP for stakeholder trust
- ✅ Build an interactive dashboard for business users to explore predictions without writing code
- ✅ Translate analytical findings into concrete, actionable retention strategies

---

## 🗂️ 4. Dataset Description

The project uses the **European Bank Customer Dataset**, a widely used benchmark dataset in churn analytics containing customer demographic, account, and behavioral information.

| Attribute | Description |
|---|---|
| `CustomerId` | Unique customer identifier |
| `CreditScore` | Customer's credit score |
| `Geography` | Country of residence (France, Germany, Spain) |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Number of years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer holds a credit card |
| `IsActiveMember` | Active membership status |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target variable** — 1 if customer churned, 0 otherwise |

**Dataset characteristics:**
- 📊 ~10,000 customer records
- ⚖️ Imbalanced target class (churn vs. non-churn)
- 🌍 Multi-country representation (France, Germany, Spain)
- 🔢 Mix of numerical and categorical features

---

## 🛠️ 5. Technology Stack

<div align="center">

| Category | Tools / Libraries |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Data Manipulation** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat-square) ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |
| **Machine Learning** | ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/-XGBoost-EC4B25?style=flat-square) |
| **Explainability** | ![SHAP](https://img.shields.io/badge/-SHAP-8A2BE2?style=flat-square) |
| **Deployment / App** | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| **Model Persistence** | ![Joblib](https://img.shields.io/badge/-Joblib-4B8BBE?style=flat-square) |

</div>

---

## 🏗️ 6. Project Architecture

```
                       ┌──────────────────────────┐
                       │   Raw Customer Dataset    │
                       │  (European Bank Dataset)  │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │  Data Cleaning & Preprocessing   │
                  │  (Missing values, encoding,      │
                  │   scaling, outlier treatment)    │
                  └────────────────┬─────────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │   Exploratory Data Analysis      │
                  │   (Univariate, Bivariate, EDA)   │
                  └────────────────┬─────────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │      Feature Engineering         │
                  │ (New features, encoding, scaling)│
                  └────────────────┬─────────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │   Model Training & Evaluation    │
                  │ (LogReg, DT, RF, GBM, XGBoost)   │
                  └────────────────┬─────────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │       Model Comparison &         │
                  │      Best Model Selection        │
                  └────────────────┬─────────────────┘
                                    │
                        ┌───────────┴────────────┐
                        ▼                         ▼
          ┌──────────────────────┐   ┌──────────────────────────┐
          │  Explainable AI (SHAP)│   │  Customer Risk Scoring    │
          │  Global & Local       │   │  (Low / Medium / High)    │
          │  Interpretability     │   │                            │
          └───────────┬──────────┘   └─────────────┬─────────────┘
                       │                            │
                       └─────────────┬──────────────┘
                                      ▼
                       ┌──────────────────────────┐
                       │   Streamlit Dashboard     │
                       │ Prediction | Risk Level   │
                       │ SHAP Visuals | What-If    │
                       └──────────────────────────┘
```

---

## 📊 7. Exploratory Data Analysis

The EDA phase uncovers patterns, distributions, and relationships within the dataset to guide feature engineering and modeling decisions.

**Key analyses performed:**
- 📈 Churn distribution and class imbalance assessment
- 🌍 Churn rate by geography, gender, and age group
- 💰 Relationship between account balance, salary, and churn
- 🔁 Tenure and product holding patterns among churned vs. retained customers
- 🧮 Correlation heatmap to assess multicollinearity
- 📦 Outlier detection using box plots
- 🧊 Active vs. inactive membership impact on churn

**Sample insights visualized:**

| Visualization | Purpose |
|---|---|
| Churn distribution bar chart | Understand target class imbalance |
| Geography-wise churn rate | Identify high-risk regions |
| Age distribution by churn status | Detect age-related churn trends |
| Correlation heatmap | Identify multicollinearity among features |
| Box plots (Balance, Salary, Credit Score) | Detect outliers and skewness |

---

## 🧪 8. Feature Engineering

To enhance model performance and capture deeper behavioral signals, the following feature engineering steps were applied:

- 🔢 **Categorical Encoding** — One-Hot Encoding for `Geography`, Label Encoding for `Gender`
- ⚖️ **Feature Scaling** — StandardScaler applied to numerical features (Balance, Salary, Credit Score)
- 🆕 **Derived Features:**
  - `BalanceSalaryRatio` — Account balance relative to estimated salary
  - `TenureByAge` — Customer tenure normalized by age
  - `IsSeniorCustomer` — Flag for customers above a defined age threshold
  - `ProductsPerTenure` — Engagement intensity indicator
- 🎯 **Class Imbalance Handling** — Techniques such as class weighting / resampling to address the skewed churn distribution
- 🧹 **Multicollinearity Check** — Variance Inflation Factor (VIF) used to drop/refine redundant features

---

## 🤖 9. Machine Learning Models

Multiple classification algorithms were trained and benchmarked to identify the optimal churn prediction model:

| Model | Type | Key Characteristic |
|---|---|---|
| **Logistic Regression** | Linear | Baseline interpretable model |
| **Decision Tree** | Tree-based | Captures non-linear relationships |
| **Random Forest** | Ensemble (Bagging) | Reduces overfitting, robust performance |
| **Gradient Boosting** | Ensemble (Boosting) | Sequential error correction |
| **XGBoost** | Ensemble (Boosting) | High-performance, regularized boosting |

Each model was trained using a **train-test split with stratification**, validated using **cross-validation**, and tuned via **hyperparameter optimization (GridSearchCV / RandomizedSearchCV)**.

---

## 📐 10. Model Evaluation Metrics

Given the class imbalance inherent in churn datasets, models were evaluated using a holistic set of metrics rather than accuracy alone:

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Of predicted churners, how many actually churned |
| **Recall (Sensitivity)** | Of actual churners, how many were correctly identified |
| **F1 Score** | Harmonic balance between Precision and Recall |
| **ROC-AUC** | Model's ability to distinguish churners from non-churners across thresholds |

### 📋 Sample Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.79 | 0.58 | 0.41 | 0.48 | 0.77 |
| Decision Tree | 0.81 | 0.60 | 0.55 | 0.57 | 0.74 |
| Random Forest | 0.86 | 0.74 | 0.61 | 0.67 | 0.85 |
| Gradient Boosting | 0.86 | 0.75 | 0.63 | 0.69 | 0.86 |
| **XGBoost (Best)** | **0.87** | **0.77** | **0.66** | **0.71** | **0.88** |

> ⚠️ *Note: Values shown are representative/sample results. Replace with your actual model outputs after training.*

---

## 🔍 11. Explainable AI (SHAP)

A black-box model — however accurate — has limited business value if stakeholders can't trust or understand its decisions. This project integrates **SHAP (SHapley Additive exPlanations)** to deliver both **global** and **local** interpretability.

**SHAP implementation includes:**

- 🌐 **Global Interpretability** — SHAP Summary Plots to rank overall feature importance across all customers
- 🔬 **Local Interpretability** — SHAP Force/Waterfall plots to explain *individual* customer predictions
- 📊 **Dependence Plots** — Reveal how specific features (e.g., Age, NumOfProducts) influence churn probability
- 🧠 **Business Translation** — Converts technical SHAP values into plain-language risk drivers for non-technical stakeholders

**Typical top churn drivers identified:**
1. Age
2. Number of Products Held
3. Account Activity Status
4. Geography
5. Account Balance

---

## 📱 12. Streamlit Dashboard

An interactive **Streamlit web application** was built to make the model accessible to business and retention teams — no coding required.

**Dashboard Features:**

| Feature | Description |
|---|---|
| 🔮 **Customer Churn Prediction** | Input customer details and get real-time churn probability |
| 🚦 **Risk Level Classification** | Customers tagged as Low / Medium / High risk |
| 📊 **Feature Importance Visualization** | Interactive SHAP-based importance charts |
| 📈 **Probability Distribution Analysis** | View churn probability distribution across the customer base |
| 🎛️ **What-If Simulator** | Adjust customer attributes (e.g., tenure, products, activity) to see real-time impact on churn risk |

```
┌─────────────────────────────────────────────────────┐
│                 STREAMLIT DASHBOARD                   │
├───────────────┬───────────────┬───────────────────────┤
│  Prediction   │  Risk Scoring │   SHAP Explainability   │
│     Tab       │      Tab      │           Tab           │
├───────────────┴───────────────┴───────────────────────┤
│              What-If Scenario Simulator                │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 13. Project Workflow

```
1️⃣  Data Collection      →  Load European Bank Customer Dataset
2️⃣  Data Preprocessing   →  Clean, encode, scale features
3️⃣  EDA                  →  Analyze patterns & churn drivers
4️⃣  Feature Engineering  →  Create predictive derived features
5️⃣  Model Training       →  Train 5 ML algorithms
6️⃣  Model Evaluation     →  Compare using Accuracy/Precision/Recall/F1/ROC-AUC
7️⃣  Model Selection      →  Select best-performing model (XGBoost)
8️⃣  Explainability       →  Apply SHAP for global & local insights
9️⃣  Risk Scoring         →  Segment customers into risk tiers
🔟  Dashboard Deployment →  Build & launch Streamlit app
```

---

## 🔑 14. Key Findings and Insights

- 👴 **Age** is one of the strongest predictors — older customers show disproportionately higher churn rates
- 🌍 **Geography matters** — customers in certain regions (e.g., Germany) historically show elevated churn
- 💳 **Inactive members** churn at significantly higher rates than active ones
- 📦 **Customers with more than 2-3 products** show distinct churn behavior, often linked to dissatisfaction or product overload
- 💰 **High balance, low engagement customers** represent a high-value, high-risk segment requiring priority attention
- ⚖️ Ensemble models (**XGBoost, Gradient Boosting**) significantly outperform linear baselines on this dataset

---

## 💡 15. Business Recommendations

Based on model outputs and SHAP-driven insights, the following strategic actions are recommended for the bank's retention and relationship management teams:

1. **🎯 Prioritize High-Risk Segments** — Deploy retention offers specifically to customers flagged as "High Risk" by the model
2. **🤝 Re-engagement Campaigns** — Target inactive members with personalized engagement programs (loyalty rewards, financial advisory calls)
3. **🧓 Age-Targeted Retention** — Design retirement and senior-focused banking products to address age-driven churn
4. **📦 Product Bundling Review** — Reassess product cross-sell strategies for customers holding 3+ products showing dissatisfaction signals
5. **🌍 Region-Specific Strategy** — Investigate service quality and competitive pressure in high-churn geographies
6. **📊 Continuous Monitoring** — Operationalize the model via the dashboard for monthly churn-risk reviews by relationship managers

---

## ⚙️ 16. Installation Guide

### Prerequisites
- Python 3.9 or higher
- pip / conda package manager
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bank-customer-churn-prediction.git

# 2. Navigate into the project directory
cd bank-customer-churn-prediction

# 3. Create a virtual environment
python -m venv venv

# 4. Activate the virtual environment
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` (sample)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
plotly
streamlit
joblib
```

---

## 🚀 17. Usage Instructions

### Run the Jupyter Notebooks (Analysis & Model Training)
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

### Train the Model via Script
```bash
python src/train_model.py
```

### Launch the Streamlit Dashboard
```bash
streamlit run app/app.py
```

Once launched, the dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## 📁 18. Folder Structure

```
bank-customer-churn-prediction/
│
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/                # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── risk_scoring.py
│
├── models/
│   └── best_model.pkl            # Serialized trained model (Joblib)
│
├── app/
│   ├── app.py                    # Streamlit dashboard entry point
│   └── components/               # Dashboard UI components
│
├── reports/
│   ├── figures/                  # Saved plots & visualizations
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚧 19. Future Enhancements

- 🔄 Integrate real-time data pipelines for live churn scoring
- 🧠 Experiment with deep learning models (ANN, TabNet) for performance benchmarking
- ☁️ Deploy the dashboard on cloud platforms (AWS / Azure / Streamlit Cloud)
- 📩 Add automated email/SMS alert system for high-risk customer flags
- 🔐 Implement role-based access control for dashboard users
- 📊 Incorporate customer lifetime value (CLV) modeling alongside churn prediction
- 🌐 Expand dataset to include transactional and complaint-log data for richer features

---

## 👤 20. Author Information

**[Your Full Name]**
MBA — Business Analytics
📧 Email: your.email@example.com
🔗 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com)
🐙 GitHub: [github.com/yourusername](https://github.com)

> This project was developed as part of an MBA Business Analytics capstone to demonstrate applied machine learning and explainable AI capabilities in the banking and financial services domain.

---

## 📄 21. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

## 🙏 22. Acknowledgements

- 📚 European Bank Customer Dataset contributors and the open-source data community
- 🧰 Open-source libraries: Scikit-learn, XGBoost, SHAP, Streamlit, and the broader Python data science ecosystem
- 🎓 Faculty and mentors of the MBA Business Analytics program for guidance and feedback
- 💡 Inspiration drawn from real-world banking analytics and customer retention case studies

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Made with 🧠 + ☕ for the Banking & Financial Analytics domain**

</div>
