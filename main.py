import pandas as pd;
import numpy as np;

df = pd.read_csv('wholesale.csv', encoding="utf-8")

#Converter os valores presentes no conjunto de dados para numeros inteiros
df["Channel"] = df["Channel"].replace({"HoReCa": 0, "Retail": 1})
df["Region"] = df["Region"].replace({"Lisbon": 0, "Oporto": 1, "Other": 2})