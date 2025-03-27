import numpy as np 
import pandas as pd
 
baseDeDados = pd.read_csv('../data/admission.csv', delimiter=';')
X = baseDeDados.iloc[:, :-1].values

#recebe as variavies dependentes (admissao baseada nas notas)
Y = baseDeDados.iloc[:, -1].values

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit_transform(X[:, 1:])

from sklearn.preprocessing import LabelEncoder

#transforma campos nao numericos em variaveis categoricas (rotulos)
label_enconder_X = LabelEncoder() 
X[:, 0] = label_enconder_X.fit_transform(X[:, 0])


#one hot enconding - transforma variaveis categoricas em valores binarios
D = pd.get_dummies(X[:, 0])
X = X[:, 1:]
X = np.insert(X, 0, D.values, axis=1)

from sklearn.model_selection import train_test_split
#separacao dos dados em 20% para testes e 80% para treinos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print(X_test)