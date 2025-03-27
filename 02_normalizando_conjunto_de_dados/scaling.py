import numpy as np 
import pandas as pd 

print("Carrengo a base de dados...")
baseDeDados = pd.read_csv("../data/admission.csv", delimiter=";")
X = baseDeDados.iloc[:, :-1].values
Y = baseDeDados.iloc[:, -1].values 

print("Preenchendo os dados que estao faltando...")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer = imputer.fit_transform(X[:, 1:])
print("ok!")

print("Computando rotulação...")
from sklearn.preprocessing import LabelEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

X = X[:, 1:]
D = pd.get_dummies(X[:, 0])
X = np.insert(X, 0, D.values, axis=1)
print("ok!")

print("Separando conjuntos de teste e treino...")
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2)
print("ok!")

print("Computando normalização...")
from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
XTrain = scaleX.fit_transform(XTrain)
XTest = scaleX.fit_transform(XTest)
print("ok!")
