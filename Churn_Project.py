""""
Customer Churn Prediction for a Telecom / Service Company + Data-Driven Insights
Author: Pratima Dhende
Objectives: To develop an accurate churn prediction model and use explainability techniques 
to identify the key factors driving customer churn

"""
# -------------------
# Import Libraries 
# -------------------

import pandas as pd
import numpy as np
import seaborn as sns 
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier

# print("Loading Dataset....")
# df=pd.read_csv("Telco_Churn.csv")
csv_file = "Telco_Churn.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, csv_file)


print("Loading data from:", file_path)
df = pd.read_csv(file_path)
print("Dataset loaded. Shape:", df.shape)
print(df.head())


# print("Dataset Loaded.")
# print(df.head())

# --------------------------------
# Data Cleaning and Preprocessing
# --------------------------------
print("Data Cleaning....")
df=df.replace(" ",np.nan)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df["TotalCharges"].median(), inplace=True)

df=df.drop("customerID", axis=1)
# OR We use instead of this
# df.drop("CustomerID", axis=1, inplace=True)

for col in df.columns:
    df[col].dtype=="object"
    df[col]=LabelEncoder().fit_transform(df[col])

print("Cleaning Completed")

# ------
# Split
# ------
X=df.drop("Churn",axis=1)
y=df["Churn"]

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# ----------------
# Model Trainning
# ----------------
print("Model Tranning...")

model=RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Train Completed")

# ----------
# Evalution
# ----------
Prediction=model.predict(X_test)

print("Accurate Score: ",accuracy_score(y_test,Prediction))
print("Classification Report: \n",classification_report(y_test,Prediction))

cm=confusion_matrix(y_test,Prediction)

plt.figure(figsize=(12,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion_matrix.png",dpi=300,bbox_inches="tight")
plt.show()

# -------------------
# Feature Importance
# -------------------
importance=pd.Series(model.feature_importances_, index=X.columns)
importance=importance.sort_values(ascending=False)
plt.figure(figsize=(10,8))
importance[ :10].plot(kind="bar")
plt.title("Top 10 Feature Importance")
plt.ylabel("feature score")
plt.savefig("feature_importance.png",dpi=300,bbox_inches="tight")
plt.show()

print("'n Top 10 factors influencing churn: ")
print(importance[:10])

# churn count plot(target distribution)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.savefig("Churn Distribution.png",dpi=300,bbox_inches="tight")
plt.show()

# category feature vs churn
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title("Churn by Contract Type")
plt.xticks(rotation=45)
plt.savefig("Feature vs churn.png",dpi=300,bbox_inches="tight")
plt.show()

# histogram / distribution plot
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True)
plt.title("Monthly Charges Distribution by Churn")
plt.savefig("Distribution_plot.png",dpi=300,bbox_inches="tight")
plt.show()

# box plot(numerical feature vs churn)
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title("Tenure vs Churn")
plt.savefig("numerical feature vs churn.png",dpi=300,bbox_inches="tight")
plt.show()