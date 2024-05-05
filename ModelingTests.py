import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.metrics import *

data = pd.read_csv('proj-data.csv')

#BLOCO CHATGPT

data.dropna(inplace=True)  # Drop rows with missing values or use imputation techniques

# Encode categorical variables
data = pd.get_dummies(data, columns=['diagnoses'])

# Feature scaling (if necessary)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#NAO FAÇO A MINIMA O QUE É SUPOSTO SER ISTO
data[['numerical_column1', 'numerical_column2']] = scaler.fit_transform(data[['numerical_column1', 'numerical_column2']]) 

X = data.drop('target_column', axis=1)   # Features
y = data['target_column']                # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize models
model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = SVC()

# Train models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Make predictions
preds1 = model1.predict(X_test)
preds2 = model2.predict(X_test)
preds3 = model3.predict(X_test)

# Evaluate models
print("Logistic Regression Accuracy:", accuracy_score(y_test, preds1))
print("Random Forest Accuracy:", accuracy_score(y_test, preds2))
print("Support Vector Machine Accuracy:", accuracy_score(y_test, preds3))

# Example for Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

#FIM DO BLOCO DO CHAT GPT