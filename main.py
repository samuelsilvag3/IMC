import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt


def CarregarDataset(path):
    names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    df = pd.read_csv(path, names=names)
    return df

def TratamentoDeDados(df):
    """
    Realiza o pré-processamento dos dados carregados.

    Sugestões para o tratamento dos dados:
        * Utilize `df.head()` para visualizar as primeiras linhas e entender a estrutura.
        * Verifique a presença de valores ausentes e faça o tratamento adequado.
        * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
    
    Dicas adicionais:
        * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
        * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
    """
    print(df.head())
    print(df.info())
    for col in df.columns.tolist():
        print("Faltando na coluna {}: {}".format(col, df[col].isnull().sum()))
    
    return df

def Treinamento(trateddf):
    """
    Treina o modelo de machine learning.

    Detalhes:
        * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
        * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
        * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
    
    Nota: Esta função deve ser ajustada conforme o modelo escolhido.
    """
    X = np.array(trateddf[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
    y = np.array(trateddf['Species'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVC()
    svm.fit(X_train, y_train)
    resultado = Teste(svm, X_test, y_test)
    return resultado

def Teste(treinamento, X_test, y_test):
    """
    Avalia o desempenho do modelo treinado nos dados de teste.

    Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
    como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
    """
    return treinamento.score(X_test, y_test)

def Train():
    
    df = CarregarDataset("iris.data")  # Carrega o dataset especificado.

    trateddf = TratamentoDeDados(df)
    resultado = Treinamento(trateddf)
    print("Resultado: {}".format(resultado))

    #Treinamento()  # Executa o treinamento do modelo
    #Teste()

# Recomenda-se criar ao menos dois modelos (e.g., Regressão Linear e SVM) para comparar o desempenho.
# A biblioteca já importa LinearRegression e SVC, mas outras escolhas de modelo são permitidas.

Train()
