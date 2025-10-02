# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:12:13 2024

@author: ubanerje
"""

# Precision for Class 0 and Class 1
precision_class_0 = 0.8924
precision_class_1 = 0.4

# Recall for Class 0 and Class 1
recall_class_0 = 0.8185
recall_class_1 = 0.5507

# Calculating Macro-average Precision
macro_precision = (precision_class_0 + precision_class_1) / 2

# Calculating Macro-average Recall
macro_recall = (recall_class_0 + recall_class_1) / 2

print("Macro-average Precision:", macro_precision)
print("Macro-average Recall:", macro_recall)


# Support for Class 0 and Class 1
support_class_0 = 314
support_class_1 = 69

# Calculating Weighted-average Precision
weighted_precision = (precision_class_0 * support_class_0 + precision_class_1 * support_class_1) / (support_class_0 + support_class_1)

# Calculating Weighted-average Recall
weighted_recall = (recall_class_0 * support_class_0 + recall_class_1 * support_class_1) / (support_class_0 + support_class_1)

print("Weighted-average Precision:", weighted_precision)
print("Weighted-average Recall:", weighted_recall)
