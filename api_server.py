"""API server example"""

#
# Usage from command line:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
#

import pickle

import pandas as pd
from flask import Flask, request #Flask es paquete de python que permite app tipo cliente-servidor donde el servidor es flask

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"


FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["POST"])  #Cuando entre en el slash del final de la pag web le corra la funcion index
def index():
    """API function"""

    args = request.json
    filt_args = {key: [int(args[key])] for key in FEATURES} #Hace comprenhension en diccionario, recupera unicamente argumentos que coincidan con la clave FEATURES
    df = pd.DataFrame.from_dict(filt_args) #LO CONVERTIMOS EN DATAFRAME DE PANDAS

    with open("house_predictor.pickle", "rb") as file:
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(df) #Pronostique el precio de la casa y lo devuelve

    return str(prediction[0][0]) #El resultado de predict que es un num lo devuelvo como str para poder visualizarlo


if __name__ == "__main__":
    app.run(debug=True)