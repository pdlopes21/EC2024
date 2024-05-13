import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

data = pd.read_csv('proj-data.csv', na_values='?')

# Remover as colunas que indicam se algo foi medido ou não, a coluna com a indentificação e colunas com muitos valores ausentes
data.drop(data.filter(like='measured').columns, axis=1, inplace=True)
data.drop('[record identification]', axis=1, inplace=True)

hyperthyroid_conditions = ['A', 'B', 'C', 'D']
hypothyroid_conditions = ['E', 'F', 'G', 'H']
binding_protein = ['I', 'J']
general_health = ['K']
replacement_therapy = ['L', 'M', 'N']
discordant = ['R']
none = ['-']

for i in range(len(data)):
    if data.at[i, "diagnoses"] in hyperthyroid_conditions :
        data.at[i, "diagnoses"] = 1
    elif data.at[i, "diagnoses"] in hypothyroid_conditions :
        data.at[i, "diagnoses"] = 2
    elif data.at[i, "diagnoses"] in binding_protein :
        data.at[i, "diagnoses"] = 3
    elif data.at[i, "diagnoses"] in general_health :
        data.at[i, "diagnoses"] = 4
    elif data.at[i, "diagnoses"] in replacement_therapy :
        data.at[i, "diagnoses"] = 5
    elif data.at[i, "diagnoses"] in discordant :
        data.at[i, "diagnoses"] = 6
    elif data.at[i, "diagnoses"] in none :
        data.at[i, "diagnoses"] = 7 
    else:
        data.at[i, "diagnoses"] = 8 

data.replace('f', 0, inplace=True)
data.replace('t', 1, inplace=True)

X = data.iloc[:,:-1]

# Remover linhas com poucos valores medidos
X.dropna(axis=1, thresh=5500, inplace=True) # 5500 porque é ~75% do número total
X.drop('sex:',axis=1,inplace=True) #O Pidgey discorda
X.drop('referral source:',axis=1,inplace=True)

y = data.iloc[: , -1:]
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

#SCALER
scaler = StandardScaler()
scaler.fit(X_train)
Xt_train=scaler.fit_transform(X_train)
Xt_test=scaler.fit_transform(X_test)

#FEATURE SELECTION
N,M = Xt_train.shape

rfr=RandomForestRegressor(random_state=0)
sel = SelectFromModel(estimator=rfr,threshold=0.015)
y_train = y_train.squeeze().ravel()
y_test = y_test.squeeze().ravel()
sel.fit(Xt_train, y_train)

print("Default threshold: ", sel.threshold_)

features=sel.get_support()
Features_selected =np.arange(M)[features]

print("The features selected are columns: ", Features_selected)

nX_train=sel.transform(Xt_train)
nX_test=sel.transform(Xt_test)

score = make_scorer(matthews_corrcoef)

rfc = RandomForestClassifier(random_state=123)      
rfc = rfc.fit(Xt_train,y_train)
std = np.std([t.feature_importances_ for t in rfc.estimators_], axis=0)
importances = rfc.feature_importances_
features = np.argsort(importances)[::-1]

#Em caso de problemas, voltar a testar com dois RFC's
#rfc2 = RandomForestClassifier(random_state=123)      
#rfc2 = rfc2.fit(Xt_train,y_train)
#importances = rfc2.feature_importances_
#features = np.argsort(importances)[::-1]
 
for f in range(features.size):
    print("%d: Feature %d (%f)" % (f + 1, features[f],  importances[features[f]]))
 
#Plot the feature importances list:
plt.figure(1, figsize=(8,5))
plt.xticks(range(X.shape[1]), X.columns,rotation=90)
plt.bar(range(Xt_train.shape[1]), importances[features],color="b", yerr=std[features])
plt.xlim([-1, Xt_train.shape[1]])
plt.show()

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

def present_statistics(y_test, preds):
    print("Statistics:")
    print("The Precision is: %7.4f" % precision_score(y_test, preds, average='weighted'))
    print("The Accuracy is: %7.4f" % accuracy_score(y_test, preds))
    print("The Recall is: %7.4f" % recall_score(y_test, preds, average='weighted'))
    print("The F1 score is: %7.4f" % f1_score(y_test, preds, average='weighted'))
    print("The Matthews correlation coefficient is: %7.4f" % matthews_corrcoef(y_test, preds))
    print("-------------------------------------------------------------")