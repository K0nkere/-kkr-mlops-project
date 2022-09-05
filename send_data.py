from textwrap import indent
import requests
import numpy as np
import pandas as pd
import boto3
import json
from uuid import uuid4
from time import sleep
from datetime import datetime
from dateutil.relativedelta import relativedelta

import prefect
from prefect import task, flow

BUCKET = "kkr-mlops-zoomcamp"
url = "http://127.0.0.1:9696/prediction"
signal_url = "http://127.0.0.1:9898/manager"

def read_file(key, bucket=BUCKET):

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
        data = pd.read_csv(f"./{key}", sep=",", na_values='NaN')

    return data


@task
def load_data(current_date = "2015-6-17", periods = 1):
    
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


@task
def na_filter(data):
    work_data = data.copy()
    non_type = work_data[data['make'].isna() | data['model'].isna() | data['trim'].isna()].index
    work_data.drop(non_type, axis=0, inplace=True)

    y = work_data.pop('sellingprice')

    return work_data, y


@flow
def sending_stream(current_date, periods):

    test_data, selling_price = na_filter(load_data(current_date, periods))
    test_data.index = range(len(test_data))
    test_data = test_data.replace(np.nan, None).to_dict(orient='index')

    # class DateTimeEncoder(json.JSONEncoder):
    #     def default(self, o):
    #         if isinstance(o, datetime):
    #             return o.isoformat()
    #         return json.JSONEncoder.default(self, o)

    with open('./target.csv', 'w') as f_in:        
        for index in test_data:

            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                # data=json.dumps(test_data[index], cls=DateTimeEncoder)
                data=json.dumps(test_data[index])
                )

            record = response.json()
            record['id'] = str(uuid4())
            print(f"Record #{index}", json.dumps(record))
            f_in.write(f"Record #{index} {record['id']}, {selling_price.iloc[index]}, {record['price_estimation']}\n")       
            sleep(0.1)
            if index >=200:
                break

    f_in.close()

    signal = {"current_date": current_date, "finished": True}

    response = requests.post(
                    signal_url,
                    json = signal
                    # headers={"Content-Type": "application/json"},
                    # data=json.dumps(test_data[index], cls=DateTimeEncoder)
                    # data=json.dumps(signal)
                    )

sending_stream(current_date = "2015-7-5", periods=0)

