import os
import json
import pandas as pd
import pickle

import mlflow

from flask import Flask, request, jsonify

PUBLIC_SERVER_IP = os.getenv("PUBLIC_SERVER_IP", False)

print(f"... Connecting to MLFlow Server on {PUBLIC_SERVER_IP} ...")

# MLFLOW_TRACKING_URI = 'sqlite:///../mlops-project.db'
MLFLOW_TRACKING_URI = f"http://{PUBLIC_SERVER_IP}:5001"
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_name = "Auction-car-prices-prediction"
model_uri = f"models:/{model_name}/production"

if PUBLIC_SERVER_IP:
    print("... Loading prediction model from production stage ...")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

else:
    print("... Loading pre-saved model ...")

    with open("model/model.pkl", "rb") as f_in:
        model = pickle.load(f_in)

def prediction(record):
    record_df = pd.DataFrame([record])
    price_prediction = model.predict(record_df)

    return float(price_prediction[0])

print("... Ready to requests ...")

app = Flask("Car-price-prediction-service")

@app.route('/prediction', methods = ['POST'])
def prediction_endpoint():
    
    record = request.get_json()
    print("\nEstimating of the car with following characteristics\n", json.dumps(record, indent=4))
    
    price_prediction = prediction(record)
    print(price_prediction)
    result = {
        "price_estimation": price_prediction
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

