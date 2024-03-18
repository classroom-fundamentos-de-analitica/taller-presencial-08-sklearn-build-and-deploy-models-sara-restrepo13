import pickle

with open("house_predictor.pickle", "rb") as file: #Abrimos archivo para escritura en binario
    loaded_model=pickle.load(file)

print(loaded_model.coef_)
print(loaded_model.intercept_)