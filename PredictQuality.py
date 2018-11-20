from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from numpy import sum
from sklearn.naive_bayes import GaussianNB


# Gravando dataframes em variaveis com o pandas
df = pd.read_csv('winequality-red.csv')
Y_df = df['quality']
X_df = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

# Otimizadores de features
label_quality = LabelEncoder()
sc = StandardScaler()

# Transformando output de inteiro para binaria, a fim de melhorar performance
bins = (0, 6.5, 10)
group_names = ['bad', 'good']
Y_df = pd.cut(Y_df, bins = bins, labels = group_names)
Y_df = label_quality.fit_transform(Y_df)

# Estabelecendo 80% dos dados para treino e 20% para teste
tamanho_de_treino = int(0.8 * len(Y_df))
tamanho_de_teste = 100 - tamanho_de_treino

# Criando arrays de treino e teste	
X_treino = X_df.values[0:tamanho_de_treino]
X_treino = sc.fit_transform(X_treino)	
Y_treino = Y_df[0:tamanho_de_treino]

X_teste = X_df.values[tamanho_de_treino:(len(Y_df))]
X_teste = sc.fit_transform(X_teste)	
Y_teste = Y_df[tamanho_de_treino:(len(Y_df))]

# Treinando dados com o MLP Classifier com 11 neuronios de entrada, 11 escondidos e 2 de saida
classifier = MLPClassifier(activation='relu', max_iter=10000, hidden_layer_sizes=(11,11,2))
classifier.fit(X_treino, Y_treino)

# Armazenando resultados finais para comparacoes
resultado = classifier.predict(X_teste)
soma_erros = sum([resultado,Y_teste], axis=0)

# Print de resultados finais
print('-----------------------------------------------')
print('score:', classifier.score(X_teste, Y_teste))
print('predictions: \n', resultado)
print('expected: \n', Y_teste)
print('sum (0 and 2 are correct): \n', soma_erros)
print('Foram detectados ' + str(len(np.where(soma_erros==1)[0])) + ' erros em ' + str(len(Y_teste)) + ' amostras.')

# Prints para teste de naives
print('------------------------------------------------')
print(' Agora, testamos com o naive bayes')
modelo = GaussianNB()
modelo.fit(X_teste, Y_teste)
resultado = modelo.predict(X_teste)
acertos = 0
for i in range(0, len(resultado)):
	if Y_teste[i] == resultado[i]:
		acertos=acertos+1

total_de_elementos = len(resultado)
taxa_de_acerto = 100.0 * acertos / total_de_elementos
print("Taxa de acerto de " + str(taxa_de_acerto) + "%")
print(" ------------- Taxa de acertos -----------------")
print(acertos)
print("------------------------------------------------")


