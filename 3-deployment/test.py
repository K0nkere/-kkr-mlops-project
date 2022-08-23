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

url = "http://127.0.0.1:9696/prediction"
# url = "http://10.129.0.20:9696/prediction"

response = requests.post(url, json = test)
print(response.json())