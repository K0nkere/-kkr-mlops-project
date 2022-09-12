"""
Logic for Evidently report creation
"""

import os
import json
from datetime import datetime

import pandas as pd
import requests
import backend_services
from prefect import flow, task
from pymongo import MongoClient
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from dateutil.relativedelta import relativedelta
from evidently.model_profile import Profile
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
from evidently.model_profile.sections import (
    DataDriftProfileSection,
    RegressionPerformanceProfileSection,
)

signal_url = "http://127.0.0.1:9898/manager"

# @task
# def upload_target(filename):
#     client = MongoClient("mongodb://localhost:27018/")
#     collection = client.get_database("prediction_service").get_collection("data")
#     with open(filename) as f_target:
#         for line in f_target.readlines():
#             row = line.split(",")
#             # print({"id": row[0]}, {"$set": {"target": float(row[1])}})
#             collection.update_one({"id": row[0]}, {"$set": {"target": float(row[1]), "prediction": float(row[2])}})
#     client.close()


@task
def load_reference_data(current_date, periods=0):
    """
    Uploads model, creates prediction and adds them to original dataframe
    """

    model = backend_services.load_model()

    reference_data, target = backend_services.na_filter(
        backend_services.load_data(current_date, periods)
    )

    reference_data['prediction'] = model.predict(reference_data)
    reference_data['target'] = target.values

    return reference_data


def return_pre_parent():
    """
    Return path to pre-root folder
    """

    path_to_script = os.path.split(__file__)[0]
    pre_parent = os.path.split(path_to_script)[0]

    return pre_parent


@task
def fetch_data(filename):
    """
    Reads previously saved by send_data script data to dataframe
    """
    # client = MongoClient("mongodb://localhost:27018/")
    # data = client.get_database("prediction_service").get_collection("data-tst").find()
    # df = pd.DataFrame(list(data))

    file_path = return_pre_parent()
    file_path = f"{return_pre_parent()}/{filename}"
    try:
        data = pd.read_csv(file_path, sep=",", header=None)
        data.columns = ['id', 'target', 'prediction']

    except:
        print("... Cant find the target.csv file Upload the data first ...")
        return pd.DataFrame()

    return data


@task
def run_evidently(ref_data, data):
    """
    Creates evidently batch report
    """

    # ref_data.drop('ehail_fee', axis=1, inplace=True)
    # data.drop('ehail_fee', axis=1, inplace=True)  # drop empty column (until Evidently will work with it properly)
    profile = Profile(
        sections=[
            DataDriftProfileSection(),
            RegressionPerformanceProfileSection(),
        ]
    )
    mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=[],  # ['condition', 'odometer', 'mmr'],
        categorical_features=[],  # ['year', 'make', 'model', 'trim', 'body', 'transmission', 'color', 'interior'],
        datetime_features=['saledate'],
    )
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(
        tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)]
    )
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    """
    Saves evidently report to mongoDB
    """

    client = MongoClient("mongodb://localhost:27018/")
    client.get_database("prediction_service").get_collection(
        "report"
    ).insert_one(result[0])


@task
def save_html_report(result, current_date):
    """
    Saves evidently report locally in html format
    """

    file_path = return_pre_parent()
    result[1].save(f"{file_path}/reports/evidently_report_{current_date}.html")


@flow
def batch_analyze(current_date='2015-5-17', target_file='targets/target.csv'):
    """
    Connects to manager-service, sends request for report creation, upon report creation returns Drift status and period for retraining
    """

    print("... Sending a request for report creation ...")
    check = {"create_report": True}

    responce = requests.post(url=signal_url, json=check, timeout=60)

    responce = responce.json()
    current_date = responce["current_date"]

    if responce["create_report"]:

        target_file = responce["target_file"]

        print(f'... Creating report for {current_date} ...')

        ref_data = load_reference_data(current_date, periods=0)
        data = fetch_data(target_file)

        if len(data) == 0:
            return print("... Waiting for the latest data ...")

        ref_data.fillna(-1, inplace=True)
        data.fillna(-1, inplace=True)

        report = run_evidently(ref_data, data)

        save_report(report)
        save_html_report(report, current_date)

        retrain = report[0]['data_drift']['data']['metrics']['prediction'][
            'drift_detected'
        ]

        if retrain:
            retrain_dt = datetime.strptime(
                current_date, "%Y-%m-%d"
            ) + relativedelta(months=1)
            retrain_date = f"{retrain_dt.year}-{retrain_dt.month}-01"
            signal = {
                "service": "report",
                "drift": retrain,
                "retrain_period": retrain_date,
            }

        else:
            signal = {
                "service": "report",
                "drift": retrain,
                "retrain_period": False,
            }

        print(
            f'... Drift status on {current_date} for prediction model: {signal["drift"]} ...'
        )
        print(json.dumps(signal, indent=2))

        requests.post(url=signal_url, json=signal, timeout=60)

    else:
        print(
            f'... Create report = {responce["create_report"]} for {current_date} Waiting for the latest data ...'
        )
