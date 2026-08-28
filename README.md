# Machine Learning Fundamentals — Assignment 1

## Supervised ML Benchmark

## 1. Project Overview

This project builds a reproducible supervised machine learning benchmark for predicting employee attrition.

Multiple classification algorithms are trained and evaluated using the same dataset split and preprocessing pipeline. The goal is not only to identify the model with the highest score, but also to understand model assumptions, scaling requirements, overfitting, interpretability, latency, and class imbalance.

### Algorithms Evaluated

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Machine (SVM)

---

## 2. Dataset

The dataset contains employee information and an `attrition_flag` target indicating whether an employee left the organization.

### Dataset Size

* Rows: 945
* Columns: 12
* Target: `attrition_flag`

### Features

#### Numerical Features

* `tenure_years`
* `performance_rating`
* `training_hours_annual`
* `avg_overtime_hrs_week`
* `job_satisfaction_score`
* `work_life_balance_score`
* `last_promotion_years_ago`

#### Categorical Features

* `department`
* `role_level`
* `salary_band`

`employee_id` was excluded from model training because it is an identifier rather than a meaningful predictive feature.

---

## 3. Target Distribution

The target contains two classes:

| Class            | Percentage |
| ---------------- | ---------: |
| No Attrition (0) |     74.03% |
| Attrition (1)    |     25.97% |

The dataset is moderately imbalanced.

Because of this imbalance, accuracy alone is not sufficient for evaluating the models. Precision, recall, F1-score, and ROC-AUC are also considered.

---

## 4. Data Analysis

The dataset was inspected before model training.

The analysis included:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate rows
* Numerical and categorical features
* Target distribution
* Categorical value counts
* Numerical summary statistics
* Unique values
* Constant-column detection

There were no duplicate rows.

Missing values were found in:

* `job_satisfaction_score`
* `work_life_balance_score`
* `last_promotion_years_ago`
* `attrition_flag`

The missing target row was excluded before model training.

---

## 5. Train / Validation / Test Split

The dataset was divided into training, validation, and test sets using a fixed random seed.

Stratification was used to preserve the class distribution across the splits.

The validation set was used for model comparison and selection.

The test set was held back for final evaluation.

This prevents the test data from influencing model selection.

---

## 6. Preprocessing

A Scikit-learn preprocessing pipeline was used.

### Numerical Features

The following steps were applied:

1. Missing values were replaced using the median.
2. Numerical features were standardized using `StandardScaler`.

### Categorical Features

The following steps were applied:

1. Missing values were replaced using the most frequent value.
2. Categorical variables were converted into numerical features using `OneHotEncoder`.

The preprocessing pipeline was applied consistently to the training, validation, and test data.

This helps prevent data leakage.

---

## 7. Why Scaling Is Important

Scaling is particularly important for algorithms that depend on distances or the magnitude of features.

### Algorithms that require scaling

**KNN**

KNN calculates distances between observations. Without scaling, features with larger numerical ranges can dominate the distance calculation.

**SVM**

SVM is sensitive to feature magnitude because the decision boundary depends on the feature space.

**Logistic Regression**

Scaling is not mathematically mandatory, but it generally improves optimization and makes regularization behave more consistently across features.

### Algorithms that generally do not require scaling

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Tree-based models split data using feature thresholds, so feature magnitude does not affect the splitting process in the same way.

---

## 8. Model Results

### Validation Results

| Model               |   Accuracy | Precision | Recall |         F1 | ROC-AUC |
| ------------------- | ---------: | --------: | -----: | ---------: | ------: |
| Logistic Regression |     0.7090 |    0.3500 | 0.1429 |     0.2029 |  0.5838 |
| KNN                 |     0.7196 |    0.4091 | 0.1837 | **0.2535** |  0.5523 |
| Decision Tree       |     0.7143 |    0.3913 | 0.1837 |     0.2500 |  0.5553 |
| Random Forest       |     0.7302 |    0.3750 | 0.0612 |     0.1053 |  0.5728 |
| Gradient Boosting   |     0.7143 |    0.3529 | 0.1224 |     0.1818 |  0.5627 |
| XGBoost             |     0.7249 |    0.3846 | 0.1020 |     0.1613 |  0.5474 |
| SVM                 | **0.7407** |    0.0000 | 0.0000 |     0.0000 |  0.5376 |

---

## 9. Training and Prediction Time

The models were also compared based on computational performance.

| Model               | Training Time (sec) | Prediction Time (sec) |
| ------------------- | ------------------: | --------------------: |
| Logistic Regression |              0.0587 |                0.0399 |
| KNN                 |              0.0336 |                0.1158 |
| Decision Tree       |              0.0315 |                0.0208 |
| Random Forest       |              0.3922 |                0.0988 |
| Gradient Boosting   |              0.2199 |                0.0153 |
| XGBoost             |              0.1594 |                0.0153 |
| SVM                 |              0.1513 |                0.0677 |

KNN trained quickly but had relatively high prediction time because predictions require calculating distances to training observations.

Decision Tree had both low training and prediction times.

Random Forest had the highest training time among the tested models.

---

## 10. Model Selection

KNN was selected as the best validation model based on F1-score.

### KNN Validation Performance

* Accuracy: 71.96%
* Precision: 40.91%
* Recall: 18.37%
* F1-score: 25.35%
* ROC-AUC: 55.23%

Although SVM achieved the highest validation accuracy of 74.07%, its precision, recall, and F1-score were all zero.

This indicates that SVM failed to identify positive attrition cases effectively.

Therefore, accuracy alone was not used to select the final model.

---

## 11. Overfitting Analysis

Random Forest showed the clearest evidence of overfitting.

### Random Forest

* Training Accuracy: 100%
* Validation Accuracy: 73.02%
* Training F1: 100%
* Validation F1: 10.53%

The large difference between training and validation performance indicates that the model learned the training data too closely and did not generalize well.

Gradient Boosting and XGBoost also showed noticeable gaps between training and validation performance.

---

## 12. Underfitting Analysis

Logistic Regression and SVM showed relatively weak training performance compared with the more complex tree-based models.

SVM was particularly problematic because it produced:

* Validation Precision: 0%
* Validation Recall: 0%
* Validation F1: 0%

This suggests that the model was not effectively separating the positive attrition class under the current configuration.

---

## 13. Final Test Evaluation

After selecting KNN using the validation set, it was evaluated on the previously unseen test set.

### Test Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 71.96% |
| Precision | 40.91% |
| Recall    | 18.37% |
| F1-score  | 25.35% |
| ROC-AUC   | 59.09% |

### Confusion Matrix

```text
[[127  13]
 [ 40   9]]
```

Where:

* True Negatives = 127
* False Positives = 13
* False Negatives = 40
* True Positives = 9

The model correctly identified 9 employees who belonged to the attrition class but missed 40 positive cases.

---

## 14. Validation vs Test

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     71.96% | 71.96% |
| F1-score |     25.35% | 25.35% |
| ROC-AUC  |     55.23% | 59.09% |

The validation and test F1-scores were identical.

The test ROC-AUC was slightly higher than the validation ROC-AUC.

Overall, the model showed relatively consistent performance between validation and test data.

---

## 15. Class Imbalance

Approximately 74% of employees belong to the no-attrition class, while approximately 26% belong to the attrition class.

This means a model could achieve relatively high accuracy simply by predicting the majority class most of the time.

For example, SVM achieved 74.07% validation accuracy but had:

* Precision = 0
* Recall = 0
* F1 = 0

Therefore, accuracy alone would give a misleading impression of model quality.

For this problem, F1-score and recall are particularly important because identifying employees who may leave is more informative than simply predicting the majority class.

---

## 16. Production Considerations

If the main goal is identifying employees at risk of attrition, the current KNN model is the strongest choice among the tested models based on validation F1-score.

However, its overall predictive performance is still weak.

Before production deployment, the model should be improved through:

* Hyperparameter tuning
* Cross-validation
* Better feature engineering
* Class-imbalance handling
* Threshold optimization
* Additional relevant employee features

If interpretability and low latency were the primary requirements, Logistic Regression would be easier to explain and operationalize.

The final production choice should therefore consider more than predictive score, including:

* Accuracy and F1/recall
* Interpretability
* Prediction latency
* Training cost
* Data size
* Maintenance requirements
* Business consequences of false positives and false negatives

---

## 17. Project Structure

```text
ML_Supervised_Benchmark/
│
├── data/
│   └── employee_attrition.csv
│
├── src/
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── visualize_results.py
│   └── evaluation.py
│
├── results/
│   ├── model_comparison.csv
│   ├── final_model_comparison.csv
│   ├── validation_accuracy.png
│   ├── validation_f1.png
│   ├── validation_roc_auc.png
│   ├── train_vs_validation_accuracy.png
│   ├── train_vs_validation_f1.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

## 18. How to Run

### Create virtual environment

```bash
python3 -m venv .venv
```

### Activate environment

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run dataset analysis

```bash
python src/data_analysis.py
```

### Run preprocessing

```bash
python src/preprocessing.py
```

### Train and compare models

```bash
python src/train_models.py
```

### Generate visualizations

```bash
python src/visualize_results.py
```

### Run final evaluation

```bash
python src/evaluation.py
```

---

## 19. Conclusion

This assignment demonstrated a complete supervised machine learning workflow using Scikit-learn and XGBoost.

The project covered:

* Dataset analysis
* Train/validation/test splitting
* Missing-value handling
* Categorical encoding
* Feature scaling
* Multiple classification algorithms
* Training and prediction timing
* Model comparison
* Accuracy, precision, recall, F1-score and ROC-AUC
* Confusion matrix analysis
* Overfitting and underfitting
* Class imbalance
* Validation versus test evaluation
* Production model considerations

The experiment demonstrates that the model with the highest accuracy is not necessarily the best model for the business problem. Model selection should consider the appropriate evaluation metric, data characteristics, interpretability, latency, and operational requirements.
# Machine Learning Fundamentals — Assignment 1

## Supervised ML Benchmark

## 1. Project Overview

This project builds a reproducible supervised machine learning benchmark for predicting employee attrition.

Multiple classification algorithms are trained and evaluated using the same dataset split and preprocessing pipeline. The goal is not only to identify the model with the highest score, but also to understand model assumptions, scaling requirements, overfitting, interpretability, latency, and class imbalance.

### Algorithms Evaluated

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Machine (SVM)

---

## 2. Dataset

The dataset contains employee information and an `attrition_flag` target indicating whether an employee left the organization.

### Dataset Size

* Rows: 945
* Columns: 12
* Target: `attrition_flag`

### Features

#### Numerical Features

* `tenure_years`
* `performance_rating`
* `training_hours_annual`
* `avg_overtime_hrs_week`
* `job_satisfaction_score`
* `work_life_balance_score`
* `last_promotion_years_ago`

#### Categorical Features

* `department`
* `role_level`
* `salary_band`

`employee_id` was excluded from model training because it is an identifier rather than a meaningful predictive feature.

---

## 3. Target Distribution

The target contains two classes:

| Class            | Percentage |
| ---------------- | ---------: |
| No Attrition (0) |     74.03% |
| Attrition (1)    |     25.97% |

The dataset is moderately imbalanced.

Because of this imbalance, accuracy alone is not sufficient for evaluating the models. Precision, recall, F1-score, and ROC-AUC are also considered.

---

## 4. Data Analysis

The dataset was inspected before model training.

The analysis included:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate rows
* Numerical and categorical features
* Target distribution
* Categorical value counts
* Numerical summary statistics
* Unique values
* Constant-column detection

There were no duplicate rows.

Missing values were found in:

* `job_satisfaction_score`
* `work_life_balance_score`
* `last_promotion_years_ago`
* `attrition_flag`

The missing target row was excluded before model training.

---

## 5. Train / Validation / Test Split

The dataset was divided into training, validation, and test sets using a fixed random seed.

Stratification was used to preserve the class distribution across the splits.

The validation set was used for model comparison and selection.

The test set was held back for final evaluation.

This prevents the test data from influencing model selection.

---

## 6. Preprocessing

A Scikit-learn preprocessing pipeline was used.

### Numerical Features

The following steps were applied:

1. Missing values were replaced using the median.
2. Numerical features were standardized using `StandardScaler`.

### Categorical Features

The following steps were applied:

1. Missing values were replaced using the most frequent value.
2. Categorical variables were converted into numerical features using `OneHotEncoder`.

The preprocessing pipeline was applied consistently to the training, validation, and test data.

This helps prevent data leakage.

---

## 7. Why Scaling Is Important

Scaling is particularly important for algorithms that depend on distances or the magnitude of features.

### Algorithms that require scaling

**KNN**

KNN calculates distances between observations. Without scaling, features with larger numerical ranges can dominate the distance calculation.

**SVM**

SVM is sensitive to feature magnitude because the decision boundary depends on the feature space.

**Logistic Regression**

Scaling is not mathematically mandatory, but it generally improves optimization and makes regularization behave more consistently across features.

### Algorithms that generally do not require scaling

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Tree-based models split data using feature thresholds, so feature magnitude does not affect the splitting process in the same way.

---

## 8. Model Results

### Validation Results

| Model               |   Accuracy | Precision | Recall |         F1 | ROC-AUC |
| ------------------- | ---------: | --------: | -----: | ---------: | ------: |
| Logistic Regression |     0.7090 |    0.3500 | 0.1429 |     0.2029 |  0.5838 |
| KNN                 |     0.7196 |    0.4091 | 0.1837 | **0.2535** |  0.5523 |
| Decision Tree       |     0.7143 |    0.3913 | 0.1837 |     0.2500 |  0.5553 |
| Random Forest       |     0.7302 |    0.3750 | 0.0612 |     0.1053 |  0.5728 |
| Gradient Boosting   |     0.7143 |    0.3529 | 0.1224 |     0.1818 |  0.5627 |
| XGBoost             |     0.7249 |    0.3846 | 0.1020 |     0.1613 |  0.5474 |
| SVM                 | **0.7407** |    0.0000 | 0.0000 |     0.0000 |  0.5376 |

---

## 9. Training and Prediction Time

The models were also compared based on computational performance.

| Model               | Training Time (sec) | Prediction Time (sec) |
| ------------------- | ------------------: | --------------------: |
| Logistic Regression |              0.0587 |                0.0399 |
| KNN                 |              0.0336 |                0.1158 |
| Decision Tree       |              0.0315 |                0.0208 |
| Random Forest       |              0.3922 |                0.0988 |
| Gradient Boosting   |              0.2199 |                0.0153 |
| XGBoost             |              0.1594 |                0.0153 |
| SVM                 |              0.1513 |                0.0677 |

KNN trained quickly but had relatively high prediction time because predictions require calculating distances to training observations.

Decision Tree had both low training and prediction times.

Random Forest had the highest training time among the tested models.

---

## 10. Model Selection

KNN was selected as the best validation model based on F1-score.

### KNN Validation Performance

* Accuracy: 71.96%
* Precision: 40.91%
* Recall: 18.37%
* F1-score: 25.35%
* ROC-AUC: 55.23%

Although SVM achieved the highest validation accuracy of 74.07%, its precision, recall, and F1-score were all zero.

This indicates that SVM failed to identify positive attrition cases effectively.

Therefore, accuracy alone was not used to select the final model.

---

## 11. Overfitting Analysis

Random Forest showed the clearest evidence of overfitting.

### Random Forest

* Training Accuracy: 100%
* Validation Accuracy: 73.02%
* Training F1: 100%
* Validation F1: 10.53%

The large difference between training and validation performance indicates that the model learned the training data too closely and did not generalize well.

Gradient Boosting and XGBoost also showed noticeable gaps between training and validation performance.

---

## 12. Underfitting Analysis

Logistic Regression and SVM showed relatively weak training performance compared with the more complex tree-based models.

SVM was particularly problematic because it produced:

* Validation Precision: 0%
* Validation Recall: 0%
* Validation F1: 0%

This suggests that the model was not effectively separating the positive attrition class under the current configuration.

---

## 13. Final Test Evaluation

After selecting KNN using the validation set, it was evaluated on the previously unseen test set.

### Test Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 71.96% |
| Precision | 40.91% |
| Recall    | 18.37% |
| F1-score  | 25.35% |
| ROC-AUC   | 59.09% |

### Confusion Matrix

```text
[[127  13]
 [ 40   9]]
```

Where:

* True Negatives = 127
* False Positives = 13
* False Negatives = 40
* True Positives = 9

The model correctly identified 9 employees who belonged to the attrition class but missed 40 positive cases.

---

## 14. Validation vs Test

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     71.96% | 71.96% |
| F1-score |     25.35% | 25.35% |
| ROC-AUC  |     55.23% | 59.09% |

The validation and test F1-scores were identical.

The test ROC-AUC was slightly higher than the validation ROC-AUC.

Overall, the model showed relatively consistent performance between validation and test data.

---

## 15. Class Imbalance

Approximately 74% of employees belong to the no-attrition class, while approximately 26% belong to the attrition class.

This means a model could achieve relatively high accuracy simply by predicting the majority class most of the time.

For example, SVM achieved 74.07% validation accuracy but had:

* Precision = 0
* Recall = 0
* F1 = 0

Therefore, accuracy alone would give a misleading impression of model quality.

For this problem, F1-score and recall are particularly important because identifying employees who may leave is more informative than simply predicting the majority class.

---

## 16. Production Considerations

If the main goal is identifying employees at risk of attrition, the current KNN model is the strongest choice among the tested models based on validation F1-score.

However, its overall predictive performance is still weak.

Before production deployment, the model should be improved through:

* Hyperparameter tuning
* Cross-validation
* Better feature engineering
* Class-imbalance handling
* Threshold optimization
* Additional relevant employee features

If interpretability and low latency were the primary requirements, Logistic Regression would be easier to explain and operationalize.

The final production choice should therefore consider more than predictive score, including:

* Accuracy and F1/recall
* Interpretability
* Prediction latency
* Training cost
* Data size
* Maintenance requirements
* Business consequences of false positives and false negatives

---

## 17. Project Structure

```text
ML_Supervised_Benchmark/
│
├── data/
│   └── employee_attrition.csv
│
├── src/
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── visualize_results.py
│   └── evaluation.py
│
├── results/
│   ├── model_comparison.csv
│   ├── final_model_comparison.csv
│   ├── validation_accuracy.png
│   ├── validation_f1.png
│   ├── validation_roc_auc.png
│   ├── train_vs_validation_accuracy.png
│   ├── train_vs_validation_f1.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

## 18. How to Run

### Create virtual environment

```bash
python3 -m venv .venv
```

### Activate environment

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run dataset analysis

```bash
python src/data_analysis.py
```

### Run preprocessing

```bash
python src/preprocessing.py
```

### Train and compare models

```bash
python src/train_models.py
```

### Generate visualizations

```bash
python src/visualize_results.py
```

### Run final evaluation

```bash
python src/evaluation.py
```

---

## 19. Conclusion

This assignment demonstrated a complete supervised machine learning workflow using Scikit-learn and XGBoost.

The project covered:

* Dataset analysis
* Train/validation/test splitting
* Missing-value handling
* Categorical encoding
* Feature scaling
* Multiple classification algorithms
* Training and prediction timing
* Model comparison
* Accuracy, precision, recall, F1-score and ROC-AUC
* Confusion matrix analysis
* Overfitting and underfitting
* Class imbalance
* Validation versus test evaluation
* Production model considerations

The experiment demonstrates that the model with the highest accuracy is not necessarily the best model for the business problem. Model selection should consider the appropriate evaluation metric, data characteristics, interpretability, latency, and operational requirements.
