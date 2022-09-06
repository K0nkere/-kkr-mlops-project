import json
import os
import pickle
import pandas as pd

import batch_service

from prefect import flow, task
from pymongo import MongoClient
import pyarrow.parquet as pq

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab,RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection


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

    model = batch_service.load_model()
   
    reference_data, target = batch_service.na_filter(batch_service.load_data(current_date, periods))
   
    reference_data['prediction'] = model.predict(reference_data)
    reference_data['target'] = target.values
    
    return reference_data


@task
def fetch_data(filename):
    # client = MongoClient("mongodb://localhost:27018/")
    # data = client.get_database("prediction_service").get_collection("data-tst").find()
    # df = pd.DataFrame(list(data))

    data = pd.read_csv(filename, sep = ",", header = None )
    data.columns = ['id', 'target', 'prediction']
    return data


@task
def run_evidently(ref_data, data):
    # ref_data.drop('ehail_fee', axis=1, inplace=True)
    # data.drop('ehail_fee', axis=1, inplace=True)  # drop empty column (until Evidently will work with it properly)
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
    mapping = ColumnMapping(
            prediction="prediction",
            numerical_features=[], #['condition', 'odometer', 'mmr'],
            categorical_features=[], #['year', 'make', 'model', 'trim', 'body', 'transmission', 'color', 'interior'],
            datetime_features=['saledate']
        )
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    client = MongoClient("mongodb://localhost:27018/")
    client.get_database("prediction_service").get_collection("report").insert_one(result[0])


@task
def save_html_report(result, current_date):
    result[1].save(f"evidently_report_{current_date}.html")


@flow
def batch_analyze(current_date='2015-6-17', target_file = '../target.csv'):
    # upload_target("../target.csv")
    ref_data = load_reference_data(current_date, periods=0)
    data = fetch_data(target_file)

    ref_data.fillna(-1,inplace=True)
    data.fillna(-1,inplace=True)

    report = run_evidently(ref_data, data)
    
    save_report(report)
    save_html_report(report, current_date)
    
    return report

# report = batch_analyze()[0]