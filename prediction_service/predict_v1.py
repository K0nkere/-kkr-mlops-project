import os
import json
import pandas as pd

import mlflow

import requests
from flask import Flask, request, jsonify

from pymongo import MongoClient

PUBLIC_SERVER_IP = os.getenv("PUBLIC_SERVER_IP")
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

MLFLOW_TRACKING_URI = f"http://{PUBLIC_SERVER_IP}:5001"

print(f"... Connecting to MLFlow Server on {MLFLOW_TRACKING_URI} ...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

model_name = "Auction-car-prices-prediction"
model_uri = f"models:/{model_name}/production"

print("... Loading prediction model from production stage ...")

model = mlflow.pyfunc.load_model(model_uri=model_uri)

def prediction(record):
    record_df = pd.DataFrame([record])
    price_prediction = model.predict(record_df)

    return float(price_prediction[0])

def save_to_db(record, prediction):
    
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    
    rec = record.copy()
    rec['prediction'] = prediction

    requests.post(
        url=f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/car-prices",
        headers={"Content-Type": "application/json"},
        data = json.dumps([rec])
        )

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
    
    save_to_db(record, price_prediction)
    send_to_evidently_service(record, price_prediction)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

