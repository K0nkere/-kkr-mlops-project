
import json
import pandas as pd

import mlflow

import requests

from pymongo import MongoClient

class MockModel:
        def __init__(self):
            pass

        def predict(self, _):

            return [25000.0]
        
        def transform(self, **kwargs):
            return kwargs

class ModelService:
    def __init__(self, PUBLIC_SERVER_IP = None,
            EVIDENTLY_SERVICE_ADDRESS = None,
            MONGODB_ADDRESS = None):

        self.PUBLIC_SERVER_IP = PUBLIC_SERVER_IP
        self.EVIDENTLY_SERVICE_ADDRESS = EVIDENTLY_SERVICE_ADDRESS
        self.MONGODB_ADDRESS = MONGODB_ADDRESS
    
        mongo_client = MongoClient(self.MONGODB_ADDRESS)
        db = mongo_client.get_database("prediction_service")
        self.collection = db.get_collection("data")


    def load_model(self):
        MLFLOW_TRACKING_URI = f"http://{self.PUBLIC_SERVER_IP}:5001"
        model_name = "Auction-car-prices-prediction"

        print(f"... Connecting to MLFlow Server on {MLFLOW_TRACKING_URI} ...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if self.PUBLIC_SERVER_IP:
            
            model_uri = f"models:/{model_name}/production"

            print("... Loading prediction model from production stage ...")

            model = mlflow.pyfunc.load_model(model_uri=model_uri)

            versions = mlflow.MlflowClient(MLFLOW_TRACKING_URI).get_latest_versions(
                    name =  model_name,
                    stages = ["Production"]
                )
            
            version = versions[0].version
            run_id = versions[0].run_id
            print(f"Version: {version} Run_id: {run_id}")

        else:
            print("... Using mock model for the test purposes... ")
            model = MockModel()
                
        return model


    def prediction(self, record):
        model = self.load_model()

        record_df = pd.DataFrame([record])
        price_prediction = model.predict(record_df)

        return float(price_prediction[0])


    def save_to_db(self, record, prediction):
        
        rec = record.copy()
        rec['prediction'] = prediction
        self.collection.insert_one(rec)


    def send_to_evidently_service(self, record, prediction):
        
        rec = record.copy()
        rec['prediction'] = prediction

        requests.post(
            url=f"{self.EVIDENTLY_SERVICE_ADDRESS}/iterate/car-prices",
            headers={"Content-Type": "application/json"},
            data = json.dumps([rec])
            )
    
    def endpoint(self, record):
        
        print("\nEstimating of the car with following characteristics\n", json.dumps(record, indent=4))
        
        price_prediction = self.prediction(record)
        print(price_prediction)
        result = {
            "price_estimation": price_prediction
        }
        
        self.save_to_db(record, price_prediction)
        self.send_to_evidently_service(record, price_prediction)
        
        return result


def init(
        PUBLIC_SERVER_IP: str,
        EVIDENTLY_SERVICE_ADDRESS: str,
        MONGODB_ADDRESS: str
    ):

    predict_service = ModelService(
        PUBLIC_SERVER_IP = PUBLIC_SERVER_IP,
        EVIDENTLY_SERVICE_ADDRESS = EVIDENTLY_SERVICE_ADDRESS,
        MONGODB_ADDRESS = MONGODB_ADDRESS
    )

    return predict_service