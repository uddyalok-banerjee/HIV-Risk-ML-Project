# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:25:05 2024

@author: ubanerje
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, confusion_matrix

file_path = 'C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Structured_Model_Dataset_04112024_AD.xlsx'

df = pd.read_excel(file_path)

# Display the first few rows of the DataFrame to inspect the data
print(df.head())

df.dropna(inplace=True)

categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and len(df[col].unique()) > 2]


one_hot_encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity
encoded_cols = pd.DataFrame(one_hot_encoder.fit_transform(df[categorical_cols]))
encoded_cols.columns = one_hot_encoder.get_feature_names(categorical_cols)

# Drop original categorical columns from DataFrame
df.drop(columns=categorical_cols, inplace=True)

# Concatenate original DataFrame with encoded columns
df_encoded = pd.concat([df, encoded_cols], axis=1)

# Split the data into features (predictors) and the target variable (label)
X = df_encoded.drop(columns=['label'])  # Features (predictors)
y = df_encoded['label']  # Target variable (label)

# Step 4: Feature Engineering
# If needed, perform additional feature engineering here

# Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, confusion_matrix

# Step 6: Choose Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Step 7-9: Train, Evaluate Models, and Calculate Metrics
for name, model in models.items():
    print("Training", name)
    model.fit(X_train, y_train)
    
    print("Evaluating", name)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate sensitivity and specificity from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Print evaluation metrics for the test dataset
    print("Evaluation Metrics for", name)
    print("Accuracy:", accuracy)
    print("ROC-AUC Score:", roc_auc)
    print("PR-AUC Score:", pr_auc)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Confusion Matrix:")
    print(cm)
    print()