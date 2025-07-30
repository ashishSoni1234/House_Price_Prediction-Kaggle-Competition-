 
# 🏠 House Price Prediction - Kaggle Competition

This project focuses on building a **stacked regression model** for predicting house prices based on various housing features. It is part of the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) Kaggle competition.

> 🏅 **Rank Achieved:** 615 out of 4531 participants (Top 14%)

---

## 📊 Problem Statement

Given a dataset of residential homes in Ames, Iowa, the objective is to predict the final price of each home using machine learning regression techniques. The dataset contains both numerical and categorical features describing different aspects of the houses.

---

## 🔧 Tech Stack Used

- Python 3
- NumPy, Pandas
- Scikit-learn (for modeling and preprocessing)
- XGBoost
- Matplotlib / Seaborn (for EDA and visualization)
- 

---

## 🧹 Data Preprocessing

- Handled missing values using:
  - Categorical: `'None'` or mode
  - Numerical: 0 or median
- Applied **label encoding** and **one-hot encoding** on categorical features.
- Applied **Box-Cox transformation** to reduce skewness in highly skewed numerical columns.
- Performed **feature scaling** using `StandardScaler()` inside pipelines.

---

## 🏗️ Models Used

### Base Models (with scaling pipelines):

- Ridge Regression (`alpha=10`)
- Lasso Regression (`alpha=0.0005`)
- ElasticNet (`alpha=0.001`, `l1_ratio=0.9`)
- XGBoost Regressor (`n_estimators=500`, `learning_rate=0.02`, etc.)

### Meta Model:

- Lasso Regression with `alpha` tuned using `GridSearchCV`.

### Final Estimator:

✅ `StackingRegressor` from Scikit-learn with `passthrough=True` and `cv=5`

---

## ⚙️ Feature Engineering

- Derived features:
  - `HouseAge = YrSold - YearBuilt`
  - `RemodAge = YrSold - YearRemodAdd`
  - `TotalBathrooms`, `TotalPorchSF`, `TotalSF`, `TotalOutsideSF` etc.
- Dropped irrelevant or highly sparse features.
- Transformed ordinal features into numerical scale.

---

## 🧪 Model Evaluation

### Metrics:
- Root Mean Squared Error (RMSE) on log-transformed target (`log1p(SalePrice)`)

| Dataset         | RMSE       |
|-----------------|------------|
| Training RMSE   | ~0.0869     |
| Test RMSE       | ~0.1304   |
| CV RMSE (5-fold)| ~0.1338    |

---
## kaggle notebook link : [https://www.kaggle.com/code/ashish2k22a318/housepriceprediction-rank-615-4531](url)
