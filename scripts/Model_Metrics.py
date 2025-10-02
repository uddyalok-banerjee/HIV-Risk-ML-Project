# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:17:42 2023

@author: ubanerje
"""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    fbeta_score,
    classification_report,
)



data = pd.read_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/CNN_Predictions_2.csv', sep=',')


# Extracting Prediction and True Value columns
predictions = data['Prediction']
true_values = data['TrueValue']

# Creating confusion matrix
conf_matrix = confusion_matrix(true_values, predictions)

# Calculating True Positive, True Negative, False Positive, False Negative
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

# Calculating Sensitivity (Recall)
sensitivity = TP / (TP + FN)

# Calculating Specificity
specificity = TN / (TN + FP)

# Calculating False Positive Rate (FPR)
fpr = FP / (FP + TN)

# Calculating False Negative Rate (FNR)
fnr = FN / (TP + FN)

# Calculating True Positive Rate (TPR)
tpr = TP / (TP + FN)

# Calculating True Negative Rate (TNR)
tnr = TN / (TN + FP)

# Calculating Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
roc_auc = roc_auc_score(true_values, predictions)

# Calculating Precision-Recall Curve and Area Under the Curve (AUC-PR)
precision, recall, _ = precision_recall_curve(true_values, predictions)
pr_auc = auc(recall, precision)

# Calculating Accuracy
accuracy = accuracy_score(true_values, predictions)

# Calculating F1 Score
f1 = f1_score(true_values, predictions)

# Calculating Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(true_values, predictions)

# Calculating Precision and Recall
precision = precision_score(true_values, predictions)
recall = recall_score(true_values, predictions)

# Calculating Balanced Accuracy
balanced_accuracy = balanced_accuracy_score(true_values, predictions)

# Calculating F2 Score
beta = 2  # You can adjust the beta parameter as needed
f2 = fbeta_score(true_values, predictions, beta=beta)

#Calculating Support
support_class_0 = TN + FP  # Support for class 0
support_class_1 = TP + FN



from sklearn.metrics import precision_score, recall_score

# Calculating Precision and Recall by Class
precision_by_class = precision_score(true_values, predictions, average=None)
recall_by_class = recall_score(true_values, predictions, average=None)

# Precision and Recall for Class 0 and Class 1
precision_class_0 = precision_by_class[0]
precision_class_1 = precision_by_class[1]
recall_class_0 = recall_by_class[0]
recall_class_1 = recall_by_class[1]



# Calculating F1 Score by Class
f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)


# Print the results
print("Confusion Matrix:")
print(conf_matrix)
print("\nSensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("False Positive Rate (FPR):", fpr)
print("False Negative Rate (FNR):", fnr)
print("True Positive Rate (TPR):", tpr)
print("True Negative Rate (TNR):", tnr)
print("\nArea Under the Receiver Operating Characteristic Curve (AUC-ROC):", roc_auc)
print("Area Under the Precision-Recall Curve (AUC-PR):", pr_auc)
print("\nAccuracy:", accuracy)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient (MCC):", mcc)
print("Precision:", precision)
print("Recall:", recall)
print("Balanced Accuracy:", balanced_accuracy)
print(f"F{beta} Score:", f2)


# Calculating Precision and Recall with macro-average
precision_macro = precision_score(true_values, predictions, average='macro')
recall_macro = recall_score(true_values, predictions, average='macro')

# Calculating Precision and Recall with weighted-average
precision_weighted = precision_score(true_values, predictions, average='weighted')
recall_weighted = recall_score(true_values, predictions, average='weighted')

# Print the results
print("\nPrecision (Macro):", precision_macro)
print("Recall (Macro):", recall_macro)
print("\nPrecision (Weighted):", precision_weighted)
print("Recall (Weighted):", recall_weighted)
