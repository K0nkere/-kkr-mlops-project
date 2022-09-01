import model_service
# import requests

# print ("hello this is a tests")

def test_prediction():
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
    # response = request.post(
    #     url = "http://127.0.0.1:9696",
    #     json = test
    # )

    model_price = model_service.ModelService(None, None, None).prediction(test)

    expected_price = 25000.0
    # {
    #     "price_estimation": 25000.0
    # }
    
    assert model_price == expected_price

#     return actual_response

# print(test_start())