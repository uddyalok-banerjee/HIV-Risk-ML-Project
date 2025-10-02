# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:58:51 2024

@author: ubanerje
"""

import pandas
import os

data = pd.read_excel('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Structured_Model_Dataset_07012024.xlsx')

df = data



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data


# Separate features (X) and target (y)
X = df.drop(['Label'], axis=1)
y = df['Label']

# Extract encounter IDs
encounter_ids = X['ENCOUNTER_ID']

# Drop encounter IDs from features
X = X.drop(['ENCOUNTER_ID'], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test, encounter_ids_train, encounter_ids_test = train_test_split(
    X, y, encounter_ids, test_size=0.2, random_state=42
)

# Define preprocessing for categorical features
categorical_features = ['Race', 'Gender_Identity', 'Insurance', 'Employment_Status']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define preprocessing for numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_transformer = SimpleImputer(strategy='mean')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models to evaluate
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42))
]

best_model = None
best_accuracy = 0

# Fit and evaluate each model
for name, model in models:
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} accuracy: {accuracy:.4f}')
    
    # Track the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

print(f'Best model: {best_model.named_steps["model"]}')



# Predict on the test set
y_pred = best_model.predict(X_test)

# Create a dataframe with encounter IDs, actual labels, and predicted labels
results_df = pd.DataFrame({
    'ENCOUNTER_ID': encounter_ids_test,
    'Actual_Label': y_test,
    'Predicted_Label': y_pred
})

# Save the dataframe as an Excel file
results_df.to_excel('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/predictions.xlsx', index=False)

print('Predictions saved to predictions.xlsx')

# Creating confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


from sklearn.model_selection import GridSearchCV

# Define the pipeline with preprocessing and random forest classifier
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Define the parameter grid
param_grid_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Perform grid search cross-validation
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search_rf.best_params_)
print("Best CV Score:", grid_search_rf.best_score_)

# Get the best model
best_model_rf = grid_search_rf.best_estimator_
