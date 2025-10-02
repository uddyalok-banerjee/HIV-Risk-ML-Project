# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:02:51 2024

@author: ubanerje
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import auc, roc_curve, average_precision_score
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

# Brihat's architecture elements
embed = 100
weight = 1
epochs = 15
batch = 48
hidden = 256
out_size = 256
dropout = 0.6
kernel = 3
stride = 1
lr = 1e-4

# Load your dataset
data = pd.read_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Rebuild_CUIS_Test.csv', sep='|')

X_cuis = data['CUIS'].values
y = data['M_Class'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

max_cuis = 10000
tokenizer = Tokenizer(num_words=max_cuis)
tokenizer.fit_on_texts(X_cuis)
X_sequences = tokenizer.texts_to_sequences(X_cuis)

max_cuis_sequence_length = 3000
X_sequences = pad_sequences(X_sequences, maxlen=max_cuis_sequence_length)

X_train, X_holdout, y_train, y_holdout = train_test_split(X_sequences, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=max_cuis, output_dim=embed, input_length=max_cuis_sequence_length))
model.add(Dropout(dropout))
model.add(Conv1D(hidden, kernel, activation='relu', strides=stride, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization()) 
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(out_size, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization()) 
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model on the training data and validate on the holdout set
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch, 
                    validation_data=(X_holdout, y_holdout))

# Extract loss and accuracy for training and validation sets
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Create subplots to plot loss and accuracy curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot training and validation loss
ax[0].plot(range(epochs), train_loss, label='Training Loss')
ax[0].plot(range(epochs), val_loss, label='Validation Loss')
ax[0].set_title('Loss Curve')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot training and validation accuracy
ax[1].plot(range(epochs), train_acc, label='Training Accuracy')
ax[1].plot(range(epochs), val_acc, label='Validation Accuracy')
ax[1].set_title('Accuracy Curve')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

# Display the plots
fig.savefig('learning_curve4.png')
plt.show()

#Plots with raw samples

iterations_per_epoch = np.ceil(len(X_train) / batch).astype(int)
cumulative_samples = np.arange(1, epochs + 1) * iterations_per_epoch * batch

# Create subplots to plot loss and accuracy curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot training and validation loss
ax[0].plot(cumulative_samples, train_loss, label='Training Loss')
ax[0].plot(cumulative_samples, val_loss, label='Validation Loss')
ax[0].set_title('Loss Curve')
ax[0].set_xlabel('Cumulative Samples Seen')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot training and validation accuracy
ax[1].plot(cumulative_samples, train_acc, label='Training Accuracy')
ax[1].plot(cumulative_samples, val_acc, label='Validation Accuracy')
ax[1].set_title('Accuracy Curve')
ax[1].set_xlabel('Cumulative Samples Seen')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

fig.savefig('learning_curve3.png')
plt.show()



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
