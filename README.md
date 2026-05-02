# Credit Risk and Loan Default Prediction

Machine learning system that predicts the probability of loan default using the Home Credit Default Risk dataset from Kaggle.

---

## Results

| Metric | Value |
|---|---|
| Dataset size | 307,511 applications |
| Features used | 55 |
| Best model | LightGBM |
| ROC-AUC (baseline) | 0.7012 |
| ROC-AUC (tuned) | 0.7526 |
| Defaulter Recall at 0.5 threshold | 7% |
| Defaulter Recall at 0.19 threshold | 53% |
| Class distribution | 91.9% repaid / 8.1% defaulted |

Accuracy is not used as an evaluation metric. A model that predicts everyone repays achieves 91.9% accuracy while catching zero defaulters. ROC-AUC and Recall are the correct metrics for this problem.

---

## File Structure

```
credit-risk-ml/
|
├── data/
│   ├── raw/                        # Original Kaggle CSV files, never modified
│   ├── processed/                  # Cleaned and transformed data files
│   │   ├── cleaned_data.csv        # Output of preprocessing notebook
│   │   ├── featured_data.csv       # Output of feature engineering notebook
│   │   ├── final_data.csv          # Output of feature selection notebook
│   │   └── splits/
│   │       ├── X_train.csv         # Training features after SMOTE
│   │       ├── X_test.csv          # Test features, untouched
│   │       ├── y_train.csv         # Training labels after SMOTE
│   │       └── y_test.csv          # Test labels, untouched
│   └── external/                   # Any additional external data
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Data cleaning and encoding
│   ├── 03_feature_engineering.ipynb# Creating financial features
│   ├── 04_feature_selection.ipynb  # Selecting best features
│   ├── 05_modelling.ipynb          # Model training and evaluation
│   ├── 06_tuning.ipynb             # Hyperparameter optimization
│   └── 07_shap.ipynb               # Model explainability
│
├── models/
│   └── lgbm_best_model.pkl         # Saved tuned LightGBM model
│
├── app/
│   └── app.py                      # Streamlit web application
│
├── src/                            # Python modules and utilities
├── .env.example                    # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks

**01_eda.ipynb**
Loads the raw dataset and performs exploratory analysis. Covers target variable distribution, missing value analysis, correlation with default, and key anomalies such as the DAYS_EMPLOYED placeholder value of 365243.

**02_preprocessing.ipynb**
Cleans the raw data. Drops columns with more than 40% missing values while protecting EXT_SOURCE columns, fixes the DAYS_EMPLOYED anomaly, imputes remaining missing values, encodes categorical variables, and removes duplicates. Output is cleaned_data.csv.

**03_feature_engineering.ipynb**
Creates 10 business-relevant financial features including credit-to-income ratio, EMI burden ratio, loan term, goods-to-credit ratio, employment stability ratio, income per family member, and EXT_SOURCE aggregates. Output is featured_data.csv.

**04_feature_selection.ipynb**
Reduces features from 184 to 55 using three methods: variance threshold filter, correlation filter at 0.85, and Random Forest feature importance. Output is final_data.csv.

**05_modelling.ipynb**
Splits data 80/20 with stratification, applies SMOTE only on training data, trains four models (Logistic Regression, Random Forest, XGBoost, LightGBM), evaluates using ROC-AUC, and tunes the decision threshold using precision-recall analysis.

**06_tuning.ipynb**
Uses Optuna to run 30 trials of hyperparameter search on LightGBM with 3-fold cross validation. Improves ROC-AUC from 0.7012 to 0.7526. Saves the best model using joblib.

**07_shap.ipynb**
Computes SHAP values using TreeExplainer on 5000 test samples. Generates global feature importance plots and individual prediction explanations for high risk and low risk applicants.

---

## Model Comparison

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.6378 |
| Random Forest | 0.6927 |
| XGBoost | 0.7006 |
| LightGBM baseline | 0.7012 |
| LightGBM tuned | 0.7526 |

---

## Setup

```bash
git clone https://github.com/ChickenSwab/credit-risk-ml.git
cd credit-risk-ml

python -m venv venv --without-pip
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

pip install -r requirements.txt
```

Download application_train.csv from the Home Credit Default Risk competition on Kaggle and place it in data/raw/.
https://www.kaggle.com/competitions/home-credit-default-risk/data

Run notebooks in order from 01 to 07, then launch the app:

```bash
cd app
streamlit run app.py
```

---

## Tech Stack

Python, pandas, numpy, scikit-learn, XGBoost, LightGBM, imbalanced-learn, Optuna, SHAP, joblib, Streamlit, MLflow
