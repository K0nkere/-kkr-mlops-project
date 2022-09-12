"""
Flask app for prediction service
"""

import os

import model_service
from flask import Flask, jsonify, request

PUBLIC_SERVER_IP = os.getenv("PUBLIC_SERVER_IP")
EVIDENTLY_SERVICE_ADDRESS = os.getenv(
    'EVIDENTLY_SERVICE', 'http://127.0.0.1:5000'
)
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

predict_service = model_service.init(
    PUBLIC_SERVER_IP=PUBLIC_SERVER_IP,
    EVIDENTLY_SERVICE_ADDRESS=EVIDENTLY_SERVICE_ADDRESS,
    MONGODB_ADDRESS=MONGODB_ADDRESS,
)

print("... Ready to requests ...")

app = Flask("Car-price-prediction-service")


@app.route('/prediction', methods=['POST'])
def prediction_endpoint():
    """
    Entrypoint for Flask application, gets request and returns prediction, work is based on previously loaded model_service
    """

    record = request.get_json()

    result = predict_service.endpoint(record)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
