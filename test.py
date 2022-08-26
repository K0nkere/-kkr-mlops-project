import boto3
import pandas as pd
import requests

test = {
    'year': 2012,
    'make': 'Mercedes-Benz',
    'model': 'E-Class',
    'trim': 'E350 Luxury',
    'body': 'Sedan',
    'transmission': 'automatic',
    'vin': 'wddhf5kb5ca633577',
    'state': 'nv',
    'condition': 4.6,
    'odometer': 34915.0,
    'color': 'black',
    'interior': 'off-white',
    'seller': 'mercedes-benz financial services',
    'mmr': 24800,
    'saledate': '2015-07-01 08:45:00'
    }

BUCKET = "kkr-mlops-zoomcamp"
url = "http://127.0.0.1:9696/prediction"

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

    data = pd.read_csv(obj['Body'], sep=",")

    return data


response = requests.post(url, json = test)
print(response.json())