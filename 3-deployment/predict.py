import json
import pandas as pd

import mlflow

from flask import Flask, request, jsonify

MLFLOW_TRACKING_URI = 'sqlite:///../mlops-project.db'
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_name = "Auction-car-prices-prediction"
model_uri = f"models:/{model_name}/production"

model = mlflow.pyfunc.load_model(model_uri=model_uri)

def prediction(record):
    record_df = pd.DataFrame([record])
    price_prediction = model.predict(record_df)

    return price_prediction[0]


app = Flask("Car-price-prediction")

@app.route('/prediction', methods = ['POST'])
def prediction_endpoint():
    
    record = request.get_json()
    print("\nEstimating of the car with following characteristics\n", json.dumps(record, indent=4))
    
    price_prediction = prediction(record)
    result = {
        "price_estimation": price_prediction
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

