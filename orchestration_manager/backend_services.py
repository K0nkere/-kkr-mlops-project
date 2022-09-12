import os
from datetime import datetime

import pickle
import boto3
import mlflow
import pandas as pd
from dateutil.relativedelta import relativedelta
from mlflow.tracking import MlflowClient

PUBLIC_SERVER_IP = os.getenv("PUBLIC_SERVER_IP")
BUCKET = os.getenv("BUCKET", 'kkr-mlops-zoomcamp')

def return_pre_parent():
    """
    Return path to pre-root folder
    """

    path_to_script = os.path.split(__file__)[0]
    pre_parent = os.path.split(path_to_script)[0]

    return pre_parent

def read_file(key, bucket=BUCKET):
    """
    Read data in csv from Bucket or from previously loaded local storage
    """

    try:
        print(f"... Connecting to {BUCKET} bucket ...")
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

    except:
        print(f"... Failed to connect to {BUCKET} bucket, getting data from local storage ...")
        file_path = return_pre_parent()
        data = pd.read_csv(f"{file_path}/{key}", sep=",", na_values='NaN')

    return data


def load_data(current_date = "2015-6-17", periods = 1):
    """
    Loads data for specified period:
        0 == current_date month for send data
        1 == current_date previously month for using as test
        n > 1 == n month befor current_date for train+valid 
    """

    dt_current = datetime.strptime(current_date, "%Y-%m-%d")

    if periods == 1:
        date_file = dt_current + relativedelta(months = - 1)
        print(f"Getting TEST data for {date_file.year}-{date_file.month} period")
        test_data = read_file(key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv")

        return test_data

    elif periods == 0:
        date_file = dt_current
        print(f"Getting TEST data for {date_file.year}-{date_file.month} period")
        current_data = read_file(key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv")

        return current_data

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
    """
    Removes rows with NA in 'make' or 'model' or 'trim' columns
    """

    work_data = data.copy()
    non_type = work_data[data['make'].isna() | data['model'].isna() | data['trim'].isna()].index
    work_data.drop(non_type, axis=0, inplace=True)

    y = work_data.pop('sellingprice')

    return work_data, y


def load_model():
    """
    Connects to bucket and load model from production stage. If it fails - gets the previously loaded model from local
    """

    MLFLOW_TRACKING_URI = f"http://{PUBLIC_SERVER_IP}:5001"
    model_name = "Auction-car-prices-prediction"

    print(f"... Connecting to MLFlow Server on {MLFLOW_TRACKING_URI} ...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:

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

    except:
        print("... Can`t get model from the bucket. Using locally pre-saved model ...")
        file_path = return_pre_parent()
        with open(f"{file_path}/3-deployment/model/model.pkl", "rb") as f_in:
            model = pickle.load(f_in)

    return model


def prediction(record, model = None):
    """
    Gets record from send-service, returns prediction based on loaded model
    """

    if not model:
        model = load_model()

    record_df = pd.DataFrame([record])
    price_prediction = model.predict(record_df)

    return float(price_prediction[0])
