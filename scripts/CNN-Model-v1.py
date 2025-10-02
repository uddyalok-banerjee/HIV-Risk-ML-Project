# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:17:29 2023

@author: ubanerje
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import auc, roc_curve, average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Brihat's architecture elements
embed = 100
weight = 1
epochs = 30
batch = 48
hidden = 256
out_size = 256
dropout = 0.5
kernel = 3
stride = 1
lr = 1e-4

# Load your dataset
# Replace 'your_data.csv' with the actual path to your dataset
data = pd.read_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Rebuild_CUIS_Test.csv', sep='|')

# Replace 'CUIS' with the actual column name that contains your CUIS data
X_cuis = data['CUIS'].values
y = data['M_Class'].values

# Encode class labels (convert binary classes to 0 and 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenize the CUIS data
# Replace 'max_cuis' with the desired number of unique CUIS to consider
max_cuis = 10000
tokenizer = Tokenizer(num_words=max_cuis)
tokenizer.fit_on_texts(X_cuis)
X_sequences = tokenizer.texts_to_sequences(X_cuis)

# Pad sequences to ensure equal length for the input data
# Replace 'max_cuis_sequence_length' with the desired sequence length
max_cuis_sequence_length = 3000
X_sequences = pad_sequences(X_sequences, maxlen=max_cuis_sequence_length)

# Split the data into training (80%) and holdout test (20%)
X_train, X_holdout, y_train, y_holdout = train_test_split(X_sequences, y, test_size=0.2, random_state=42)

# Build a new NLP model with the desired input shape
model = Sequential()
model.add(Embedding(input_dim=max_cuis, output_dim=embed, input_length=max_cuis_sequence_length))
model.add(Dropout(dropout))
model.add(Conv1D(hidden, kernel, activation='relu', strides=stride))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(out_size, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=epochs, batch_size=batch)

# Evaluate the model on the holdout test set (the 20% holdout set)
y_pred = model.predict(X_holdout)

# Create a list of various cutoff values to evaluate
cutoffs = np.arange(.2, .6, 0.05)

# Create dictionaries to store metrics for each cutoff
precision_values_class0 = {}
recall_values_class0 = {}
f1_values_class0 = {}
support_values_class0 = {}

precision_values_class1 = {}
recall_values_class1 = {}
f1_values_class1 = {}
support_values_class1 = {}

# Compute metrics for each cutoff and store the results
for cutoff in cutoffs:
    y_pred_binary = (y_pred > cutoff).astype(int)
    
    # Metrics for class 0
    precision_values_class0[cutoff] = precision_score(y_holdout, y_pred_binary, pos_label=0)
    recall_values_class0[cutoff] = recall_score(y_holdout, y_pred_binary, pos_label=0)
    f1_values_class0[cutoff] = f1_score(y_holdout, y_pred_binary, pos_label=0)
    support_values_class0[cutoff] = np.sum(y_pred_binary == 0)
    
    # Metrics for class 1
    precision_values_class1[cutoff] = precision_score(y_holdout, y_pred_binary, pos_label=1)
    recall_values_class1[cutoff] = recall_score(y_holdout, y_pred_binary, pos_label=1)
    f1_values_class1[cutoff] = f1_score(y_holdout, y_pred_binary, pos_label=1)
    support_values_class1[cutoff] = np.sum(y_pred_binary == 1)

# Print the evaluation metrics for each cutoff
for cutoff in cutoffs:
    print(f"Threshold: {cutoff:.2f}")
    print("Class 0 - Precision:", precision_values_class0[cutoff])
    print("Class 0 - Recall:", recall_values_class0[cutoff])
    print("Class 0 - F1 Score:", f1_values_class0[cutoff])
    print("Class 0 - Support:", support_values_class0[cutoff])
    
    print("Class 1 - Precision:", precision_values_class1[cutoff])
    print("Class 1 - Recall:", recall_values_class1[cutoff])
    print("Class 1 - F1 Score:", f1_values_class1[cutoff])
    print("Class 1 - Support:", support_values_class1[cutoff])
    print()

# Calculate ROC AUC and PR AUC for the holdout test set
fpr, tpr, _ = roc_curve(y_holdout, y_pred)
roc_auc = auc(fpr, tpr)
pr_auc = average_precision_score(y_holdout, y_pred)

# Print the evaluation metrics for the holdout test set
print("Holdout Test ROC AUC: {:.2f}".format(roc_auc))
print("Holdout Test PR AUC: {:.2f}".format(pr_auc))

model.save('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/CNN_Model.keras')

from tensorflow.keras.models import load_model
model = load_model('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/CNN_Model.keras')

# Load the tokenizer and other parameters used during training
max_cuis = 10000  
max_cuis_sequence_length = 3000  
tokenizer = Tokenizer(num_words=max_cuis)

# Load the foreign dataset
foreign_data = pd.read_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Annotated_CUIS_Complete.csv', sep='|')

# Preprocess the CUIs in the foreign dataset
X_foreign_cuis = foreign_data['CUIS'].values
tokenizer.fit_on_texts(X_foreign_cuis)
X_foreign_sequences = tokenizer.texts_to_sequences(X_foreign_cuis)
X_foreign_sequences = pad_sequences(X_foreign_sequences, maxlen=max_cuis_sequence_length)

# Make predictions
y_foreign_pred = model.predict(X_foreign_sequences)

# Set the cutoff threshold
cutoff_threshold = 0.25

# Apply the cutoff to convert probabilities to binary labels
y_foreign_pred_binary = (y_foreign_pred > cutoff_threshold).astype(int)

# Combine predictions with HSP Account IDs
predictions_df = pd.DataFrame({
    'HSP_Account_ID': foreign_data['HSP_ACCOUNT_ID'],
    'Predicted_Label': y_foreign_pred_binary.flatten()
})

# Display the predictions
print(predictions_df)

predictions_df.to_excel('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/prediction_comparison.xlsx', index=False)













