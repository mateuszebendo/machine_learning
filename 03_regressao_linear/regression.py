import numpy as np
import pandas as pd 

def loadDataset(filename):
    baseDeDados = pd.read_csv(filename, delimiter=";")
    #carrega todas as linhas, todas as colunas menos a ultima
    X = baseDeDados.iloc[:, :-1].values
    #carrega todas as linhas, carrega a ultima coluna
    Y = baseDeDados.iloc[:, -1].values 
    return X, Y 

def fillMissingData(X): 
    from sklearn.impute import SimpleImputer 
    #instancia um simpleImputer, indica que os valores faltantes serao NaN e usa media como estrategia para calcula-los
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    #aplica o fit_transform em todas linhas e todas as colunas de X
    return imputer.fit_transform(X)

def computeCategorization(X):
    from sklearn.preprocessing import LabelEncoder
    labelenconder_X = LabelEncoder() 
    #transforma a primeira coluna da tabela em variaveis categoricas
    X[:, 0] = labelenconder_X.fit_transform(X[:, 0])
    
    #one hot enconding (transformar variaveis categoricas em valores binarios para nao interferir nos calculos do modelo)
    D = pd.get_dummies(X[:, 0])
    X = np.hstack((D.values, X[:, 1:]))
    return X

def splitTrainTest(X, Y, testSize): 
    from sklearn.model_selection import train_test_split
    #cria tabelas de treino e teste
    XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=testSize, random_state=42)
    return XTrain, XTest, yTrain, yTest

def computeScaling(train, test):
    from sklearn.preprocessing import StandardScaler
    #standardScaler -> usa metodos matematicos para padronizar os dados, fazendo-os ficar na mesma escala, assim valores maiores nao dominarao o modelo
    scaleX = StandardScaler()
    train = scaleX.fit_transform(train)
    test = scaleX.transform(test)
    return train, test

def computeLinearRegressionModel(XTrain, yTrain, XTest, yTest):
    from sklearn.linear_model import LinearRegression
    #linearRegression -> a regressao linear se baseia em prever um valor com base em conjunto de dados
    #ela estabelece uma relacao matematica entre uma variavel dependente e uma ou mais independentes
    #assim encontrando a melhor linha reta dentro de um grafico que descreve a relacao entre as variaveis
    regressor = LinearRegression()
    regressor.fit(XTrain, yTrain)
    yPred = regressor.predict(XTest)
    print(str(yTest) + "\n" + str(yPred))
    
def runLinearRegressionExample(filename): 
    X, Y = loadDataset(filename=filename)
    X[:, 1:] = fillMissingData(X[:, 1:])
    Y = Y.reshape(-1, 1)
    Y = fillMissingData(Y)
    Y = Y.ravel()   
    X = computeCategorization(X)
    XTrain, XTest, yTrain, yTest = splitTrainTest(X, Y, 0.2)
    computeLinearRegressionModel(XTrain, yTrain, XTest, yTest)
    
if __name__ == "__main__": 
    runLinearRegressionExample(filename="../data/svbr.csv")