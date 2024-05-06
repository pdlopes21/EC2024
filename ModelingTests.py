import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.metrics import *

data = pd.read_csv('proj-data.csv')

#BLOCO CHATGPT

# Remover as linhas com pouca informação
data.dropna(thresh=3, subset=['T3 measured:', 'TT4 measured:', 'T4U measured:', 'FTI measured:', 'TBG measured:'], inplace=True)

# Remover as colunas com pouca informação
data.dropna(axis=1, thresh=3669, inplace=True)

# Remover as colunas que indicam se algo foi medido ou não e a que tem a indentificação
columns_to_drop = data.filter(like='measured').columns
columns_to_drop.append('[record identification]')
data.drop(columns_to_drop, axis=1, inplace=True)

# Remover linhas com homens grávidos
data = data[~((data['gender'] == 'M') & (data['pregnant'] == "t"))]

def transform_diagnoses(df):
    
    hyperthyroid_conditions = ['A', 'B', 'C', 'D']
    hypothyroid_conditions = ['E', 'F', 'G', 'H']
    binding_protein = ['I', 'J']
    general_health = ['K']
    replacement_therapy = ['L', 'M', 'N']
    antithyroid_treatment = ['O', 'P', 'Q']
    other = ['R', 'S', 'T']
    
    df['diagnoses'].replace(hyperthyroid_conditions, 'hyperthyroid', inplace=True)
    df['diagnoses'].replace(hypothyroid_conditions, 'hypothyroid', inplace=True)
    df['diagnoses'].replace(binding_protein, 'binding protein', inplace=True)
    df['diagnoses'].replace(general_health, 'general health', inplace=True)
    df['diagnoses'].replace(replacement_therapy, 'replacement therapy', inplace=True)
    df['diagnoses'].replace(antithyroid_treatment, 'antithyroid treatment', inplace=True)
    df['diagnoses'].replace(other, 'miscellaneous', inplace=True)
    
    return df

# Encode categorical variables
data = transform_diagnoses(data)

binary_cols = ['on thyroxine:', 'query on thyroxine:', 'on antithyroid medication:', 
               'sick:', 'pregnant:', 'thyroid surgery:', 'I131 treatment:', 'query hypothyroid:',
               'query hyperthyroid:', 'lithium:', 'goitre:', 'tumor:', 'hypopituitary:', 'psych:',
               'referral source:', 'diagnoses']

# Trocar os t e f e os diagnosticos para valores
data = pd.get_dummies(data, columns=binary_cols)

# Feature scaling (if necessary)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#NAO FAÇO A MINIMA O QUE É SUPOSTO SER ISTO
data[['age','TSH','T3', 'TT4', 'T4U', 'FTI', 'TBG']] = scaler.fit_transform(data) 

X = data.drop('diagnoses', axis=1)   # Features
y = data['diagnoses']                # Target variable

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

