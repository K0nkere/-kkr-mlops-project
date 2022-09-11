# import boto3
# import pandas as pd
import requests
from deepdiff import DeepDiff

# def read_file(key, bucket=BUCKET):

#     BUCKET = "kkr-mlops-zoomcamp"
#     session = boto3.session.Session()
#     s3 = session.client(
#         service_name='s3',
#         endpoint_url='https://storage.yandexcloud.net',
#         region_name='ru-central1',
#         # aws_access_key_id = "id",
#         # aws_secret_access_key = "key")
#     )
#     obj = s3.get_object(Bucket=bucket, Key=key)

#     data = pd.read_csv(obj['Body'], sep=",")

#     return data

def test_predict():
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

    url = "http://127.0.0.1:9696/prediction"
    
    response = requests.post(url, json = test)
    actual_response = response.json()

    expected_repsonse = {
        "price_estimation": 24572.63
    }
    
    diff = DeepDiff(actual_response, expected_repsonse, significant_digits=1)
    
    assert "values_changed" not in diff
    assert "types_changed" not in diff



