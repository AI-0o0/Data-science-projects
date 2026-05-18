# FIFA Player Value Prediction  Project Report

## Overview

This project applies machine learning to predict FIFA player market values (in millions of dollars) and classify players by performance tier. The dataset contains 19,667 FIFA players with 9 features, and the workflow covers exploratory data analysis, preprocessing, model training, hyperparameter tuning, error diagnosis, and ensemble modeling.

---

## Dataset

**Source file:** `Fifa.csv`

**Shape:** 19,667 rows × 9 columns

| Column | Type | Description |
|---|---|---|
| Name | object | Player name |
| Country | object | Nationality |
| Position | object | Field position |
| Age | int64 | Player age |
| Overall_Rating | int64 | Current overall rating |
| Future Potential | int64 | Projected rating |
| Team | object | Club team |
| Value Per M$ | float64 | Market value in millions (regression target) |
| Total_Stats Score | int64 | Aggregate stats score |

**Missing values:** None across all columns.

---

## Exploratory Data Analysis (EDA)

**Descriptive statistics highlights:**

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Age | 22.99 | 4.69 | 15 | 44 |
| Overall_Rating | 63.23 | 7.81 | 36 | 91 |
| Future Potential | 70.66 | 6.49 | 46 | 95 |
| Value Per M$ | 2.51 | 7.26 | 0.00 | — |
| Total_Stats Score | 1534.51 | 283.25 | 416 | — |

**Skewness analysis:** The target variable `Value Per M$` is heavily right-skewed (skewness = 7.98), indicating that most players have low market values and a small number of elite players drive a long tail. A `log1p` transformation was applied during preprocessing to address this.

**Correlation with Value Per M$:**

| Feature | Correlation |
|---|---|
| Overall_Rating | 0.561 |
| Future Potential | 0.501 |
| Total_Stats Score | 0.385 |
| Age | 0.142 |

Average pairwise correlation across all numeric features: **0.55**  moderate overall feature interdependence.

**Distribution plots** were generated for all numeric columns using Seaborn histograms with KDE overlays. A boxplot on `Value Per M$` confirmed the presence of significant high-value outliers, consistent with the extreme skewness.

---

## Preprocessing

The `preprocess_data()` function performs the following steps:

1. **Drops** the `Name` column (non-informative identifier).
2. **Splits** the data 80/20 into train/test sets (random state = 1).
3. **Applies `log1p` transformation** to `Value Per M$` on both splits to normalize the highly skewed target.
4. **Engineers a `Team_Strength` feature** from per-team averages of `Age` and `Future Potential`: `Strength = 0.3 × Age + 0.7 × Future Potential`.
5. **One-hot encodes** `Position` and continent-derived country groupings.
6. **Standard-scales** all numeric features.
7. **Creates a `Performance_Class` column** (Elite / High / Medium / Low) for the classification task.

The final feature matrix includes Age, Overall_Rating, Future Potential, Total_Stats Score, Team_Strength, position dummy variables, and continent dummy variables.

---

## Model Selection Rationale

Three regression algorithms were chosen based on the dataset characteristics:

**Random Forest Regressor**  suited to the medium-sized dataset with high skewness and categorical features after encoding. Handles non-linear relationships and mixed feature types naturally.

**SVR (RBF Kernel)**  appropriate given low-to-moderate feature count and the moderate average correlation (0.55). The RBF kernel captures non-linear patterns without requiring many features.

**KNN Regressor**  can capture local patterns in the data, useful given that value is likely clustered by position and team strength groups.

The same trio logic (RF, KNN, plus Logistic Regression for the linear baseline) was applied to the classification problem.

---

## Regression Results

### Baseline

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.3276 | 0.2280 | 0.8023 |
| Polynomial Degree 1 | 0.3276 | 0.2280 | 0.8023 |
| Polynomial Degree 2 | 0.1593 | 0.0844 | 0.9533 |
| Polynomial Degree 3 | 0.1766 | 0.0812 | 0.9424 |

### Tuned Models

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest Regressor (default) | 0.2089 | 0.1145 | 0.9196 |
| **Optimized Random Forest Regressor** | **0.1469** | **0.0449** | **0.9603** |
| SVR RBF (default) | 0.1558 | 0.0729 | 0.9553 |
| Optimized SVR RBF | 0.1549 | 0.0608 | 0.9558 |
| KNN Regressor (default) | 0.2251 | 0.1102 | 0.9067 |
| Optimized KNN Regressor | 0.2151 | 0.1013 | 0.9148 |
| **Voting Ensemble (RF + SVR + KNN)** | **0.1525** | **0.0572** | **0.9572** |

**Best hyperparameters found via GridSearchCV (3-fold CV):**

- **RF:** `n_estimators=795, max_depth=20, max_features=None, min_samples_leaf=2, min_samples_split=2`
- **SVR:** `C=10, epsilon=0.02, gamma='auto'`
- **KNN:** `n_neighbors=5, metric='manhattan', weights='distance'`

---

## Classification Results

Players were classified into four performance tiers: **Elite**, **High**, **Medium**, **Low**.

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.8005 |
| KNN Classifier (default) | 0.7677 |
| Optimized KNN Classifier | 0.7867 |
| Random Forest Classifier (default) | 0.8416 |
| **Optimized Random Forest Classifier** | **0.8564** |

**Optimized RF Classifier per-class performance:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Elite | 0.95 | 0.94 | 0.94 | 932 |
| High | 0.82 | 0.78 | 0.80 | 861 |
| Low | 0.90 | 0.90 | 0.90 | 1101 |
| Medium | 0.77 | 0.80 | 0.78 | 1040 |

The Elite class achieved the strongest performance, while High and Medium were harder to distinguish expected given their adjacent rating boundaries.

---

## Error Diagnosis (Bias-Variance)

| Model | Train RMSE | Test RMSE | Train R² | Test R² | Relative Gap | Diagnosis |
|---|---|---|---|---|---|---|
| Optimized RF Regressor | 0.0912 | 0.1469 | 0.9855 | 0.9603 | 37.88% | **Strong Overfitting** |
| Optimized SVR (RBF) | 0.1482 | 0.1549 | 0.9616 | 0.9558 | 4.33% | **Good Fit** |
| Optimized KNN Regressor | 0.0000 | 0.2151 | 1.0000 | 0.9148 | 100.00% | **Strong Overfitting** |

**10-fold cross-validation stability:**

| Model | Mean R² | Std R² | Min R² | Max R² | Verdict |
|---|---|---|---|---|---|
| Optimized RF | 0.9563 | 0.0153 | 0.9182 | 0.9778 | Stable, strong generalization |
| Optimized KNN | 0.9109 | 0.0145 | 0.8747 | — | Stable |

Despite the RF's bias-variance diagnosis flagging overfitting, its cross-validation scores remain consistent and high, suggesting the overfitting is moderate in practical impact. KNN's perfect train score (RMSE = 0.000) is a textbook memorization artifact of distance-weighted nearest-neighbor prediction on training points.

The SVR is the most theoretically well-fit model, showing minimal train/test gap, though it achieves a slightly lower R² than the RF.

---

## Ensemble

A **Voting Regressor** combining the three optimized models (RF, SVR, KNN) with equal weights produced:

| RMSE | MAE | R² |
|---|---|---|
| 0.1525 | 0.0572 | 0.9572 |

The ensemble slightly underperforms the optimized RF alone (R² 0.9603 vs 0.9572) because the KNN and SVR drag the average down, but it provides better variance reduction than any single model and avoids relying solely on the overfitting-prone RF.

---

## Summary & Recommendations

The **Optimized Random Forest Regressor** achieves the best raw test performance (R² = 0.9603) for predicting player market value, but exhibits signs of overfitting. The **Optimized SVR (RBF)** is the cleanest fit (4.33% relative RMSE gap) and a strong production candidate.

For the classification task, the **Optimized Random Forest Classifier** at 85.6% accuracy is the clear winner, particularly for the Elite tier (F1 = 0.94).

**Potential next steps:**
- Apply stronger regularization to the RF (lower `max_depth`, higher `min_samples_leaf`) to reduce overfitting.
- Explore gradient boosting methods (XGBoost, LightGBM) as natural extensions of the RF approach.
- Engineer additional features from player position clusters or league-level statistics.
- Address the KNN overfitting by experimenting with larger `n_neighbors` or dimensionality reduction.