import os
import pandas as pd
import boto3

import mlflow
from mlflow.tracking import MlflowClient

from datetime import datetime
from dateutil.relativedelta import relativedelta

PUBLIC_SERVER_IP = os.getenv("PUBLIC_SERVER_IP", "51.250.28.168")
BUCKET = 'kkr-mlops-zoomcamp'


def read_file(key, bucket=BUCKET):

    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        region_name='ru-central1',
        # aws_access_key_id = "id",
        # aws_secret_access_key = "key")
    )
    obj = s3.get_object(Bucket=bucket, Key=key)

    data = pd.read_csv(obj['Body'], sep=",", na_values='NaN')

    return data


def load_data(current_date = "2015-6-17", periods = 1):
    
    dt_current = datetime.strptime(current_date, "%Y-%m-%d")
    
    if periods == 1:
        date_file = dt_current + relativedelta(months = - 1)
        print(f"Getting TEST data for {date_file.year}-{date_file.month} period")
        test_data = read_file(key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv")

        return test_data

    else:
        train_data = pd.DataFrame()
        for i in range(periods+1, 1, -1):
            date_file = dt_current + relativedelta(months = - i)
            try:
                data = read_file(key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv")
                print(f"Getting TRAIN data for {date_file.year}-{date_file.month} period")
            except:
                print(f"Cannot find file car-prices-{date_file.year}-{date_file.month}.csv",
                    "using blank")
                data = None
                
            train_data = pd.concat([train_data, data])
        
        return train_data


def na_filter(data):
    work_data = data.copy()
    non_type = work_data[data['make'].isna() | data['model'].isna() | data['trim'].isna()].index
    work_data.drop(non_type, axis=0, inplace=True)

    y = work_data.pop('sellingprice')

    return work_data, y


def load_model():
        MLFLOW_TRACKING_URI = f"http://{PUBLIC_SERVER_IP}:5001"
        model_name = "Auction-car-prices-prediction"

        print(f"... Connecting to MLFlow Server on {MLFLOW_TRACKING_URI} ...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if PUBLIC_SERVER_IP:
            
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
            pass
                
        return model


def prediction(record):
    model = load_model()

    record_df = pd.DataFrame([record])
    price_prediction = model.predict(record_df)

    return float(price_prediction[0])