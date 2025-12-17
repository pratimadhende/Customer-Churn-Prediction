# Customer Churn Prediction ML Project

## Project Overview
This project predicts whether a customer will churn (leave a service) using machine learning. It uses the *Telco Customer Churn dataset* and implements a *Random Forest Classifier* to identify customers most likely to churn. The project is designed to be *industry-level*, clean, and ready for professional presentation.

---

## Dataset
- *Source:* [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- *Description:* The dataset contains customer information such as demographics, account details, services subscribed, and churn status.
- *File used:* churn.csv

---

## Features Used
- Numeric features: tenure, MonthlyCharges, TotalCharges, etc.
- Categorical features: gender, Partner, Dependents, Contract, PaymentMethod, etc.
- All categorical variables are *one-hot encoded* for machine learning.

---

## Project Steps
1. *Data Loading:* Safely load CSV from the project folder.
2. *Data Cleaning:* Handle missing values and convert data types.
3. *Feature Encoding:* Encode categorical variables using one-hot encoding.
4. *Train-Test Split:* 80% training, 20% testing.
5. *Model Training:* Random Forest Classifier.
6. *Evaluation:*
   - Accuracy score
   - Classification report
   - Confusion matrix
7. *Visualization:*
   - Confusion matrix heatmap
   - Top 5 features affecting customer churn

---

## Requirements
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn