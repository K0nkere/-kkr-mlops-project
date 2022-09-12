"""
Backend functions for Flask prediction service
"""

import json

import mlflow
import pandas as pd
import requests
from pymongo import MongoClient


class MockModel:
    """
    Mock model for unit test purposes
    """

    def __init__(self):
        pass

    def predict(self, _):
        """
        Imitate prediction with constanta
        """

        return [25000.0]

    def transform(self, **kwargs):
        """
        Imitate transformation of features
        """

        return kwargs


class ModelService:
    """
    Initialization class for predict service
    """

    def __init__(
        self,
        PUBLIC_SERVER_IP=None,
        EVIDENTLY_SERVICE_ADDRESS=None,
        MONGODB_ADDRESS=None,
    ):

        self.PUBLIC_SERVER_IP = PUBLIC_SERVER_IP
        self.EVIDENTLY_SERVICE_ADDRESS = EVIDENTLY_SERVICE_ADDRESS
        self.MONGODB_ADDRESS = MONGODB_ADDRESS

        mongo_client = MongoClient(self.MONGODB_ADDRESS)
        db = mongo_client.get_database("prediction_service")
        self.collection = db.get_collection("data")

    def load_model(self):
        """
        Connects to bucket and load model from production stage. If it fails - gets the Mock model
        """

        MLFLOW_TRACKING_URI = f"http://{self.PUBLIC_SERVER_IP}:5001"
        model_name = "Auction-car-prices-prediction"

        print(f"... Connecting to MLFlow Server on {MLFLOW_TRACKING_URI} ...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if self.PUBLIC_SERVER_IP:

            model_uri = f"models:/{model_name}/production"

            print("... Loading prediction model from production stage ...")

            model = mlflow.pyfunc.load_model(model_uri=model_uri)

            versions = mlflow.MlflowClient(
                MLFLOW_TRACKING_URI
            ).get_latest_versions(name=model_name, stages=["Production"])

            version = versions[0].version
            run_id = versions[0].run_id
            print(f"Version: {version} Run_id: {run_id}")

        else:
            print("... Using mock model for the test purposes... ")
            model = MockModel()

        return model

    def prediction(self, record):
        """
        Gets record, returns prediction based on loaded model
        """
        model = self.load_model()

        record_df = pd.DataFrame([record])
        price_prediction = model.predict(record_df)

        return float(price_prediction[0])

    def save_to_db(self, record, prediction):
        """
        Saves evidently report to mongoDB
        """

        rec = record.copy()
        rec['prediction'] = prediction
        self.collection.insert_one(rec)

    def send_to_evidently_service(self, record, prediction):
        """
        Send a record with prediction to Evidently for online monitoring
        """
        rec = record.copy()
        rec['prediction'] = prediction

        requests.post(
            url=f"{self.EVIDENTLY_SERVICE_ADDRESS}/iterate/car-prices",
            headers={"Content-Type": "application/json"},
            data=json.dumps([rec]),
            timeout=60,
        )

    def endpoint(self, record):
        """
        Gets record, launch prediction process, saves results to mongoDB and sends to Evidently
        """

        print(
            "\nEstimating of the car with following characteristics\n",
            json.dumps(record, indent=4),
        )

        price_prediction = self.prediction(record)
        print(price_prediction)
        result = {"price_estimation": price_prediction}

        self.save_to_db(record, price_prediction)
        self.send_to_evidently_service(record, price_prediction)

        return result


def init(
    PUBLIC_SERVER_IP: str, EVIDENTLY_SERVICE_ADDRESS: str, MONGODB_ADDRESS: str
):
    """
    Init
    """

    predict_service = ModelService(
        PUBLIC_SERVER_IP=PUBLIC_SERVER_IP,
        EVIDENTLY_SERVICE_ADDRESS=EVIDENTLY_SERVICE_ADDRESS,
        MONGODB_ADDRESS=MONGODB_ADDRESS,
    )

    return predict_service
