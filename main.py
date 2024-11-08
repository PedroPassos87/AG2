import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('wholesale.csv', encoding="utf-8")

#Converter os valores presentes no conjunto de dados para numeros inteiros
df["Channel"] = df["Channel"].replace({"HoReCa": 0, "Retail": 1})
df["Region"] = df["Region"].replace({"Lisbon": 0, "Oporto": 1, "Other": 2})

#Reordenar as colunas do conjunto de dados
column = list(df.columns)
column.append(column.pop(0))
df = df.reindex(columns= column)


#5 -Separar o conjunto de dados em duas partes: 80% para treinamento e 20% para testes.
X = df.drop(columns=["Channel"])
y = df["Channel"]
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_treino)
X_test_scaled = scaler.transform(X_teste)

# Definir e treinar o modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
mlp.fit(X_train_scaled, y_treino)

# Predizer os resultados
y_pred = mlp.predict(X_test_scaled)

# Avaliar o modelo
accuracy = accuracy_score(y_teste, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}")

# Relatório de Classificação
print(classification_report(y_teste, y_pred))