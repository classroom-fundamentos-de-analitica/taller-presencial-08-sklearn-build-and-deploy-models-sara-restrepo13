#Pickle - Salvar objetos en disco duro en formato binario. Guarda en binario modelo despues de estimado
"""Build, deploy and access a model using scikit-learn"""

import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]
#Como no todas las columnas son importantes para hacer estimación, ent creamos variable features que extrae subconjunto de columnas del df original

target = df[["price"]] #Variable explicada - columna precios

estimator = LinearRegression() #Crear modelo de regresión lineal, devuelve objeto en limpio
estimator.fit(features, target) #Llamamos metodo fit que calcula coef optimos del modelo

print(estimator.coef_) #Las a
print(estimator.intercept_) #Las b

with open("house_predictor.pickle", "wb") as file: #Abrimos archivo para escritura en binario
    pickle.dump(estimator, file) #Vaciar estimator en archivo file