"""
Predictoin test with Mock model
"""
import model_service

# import requests


def test_prediction():
    """
    Test prediction with Mock model
    """

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
        'saledate': '2015-07-01 08:45:00',
    }

    model_price = model_service.ModelService(None, None, None).prediction(test)

    expected_price = 25000.0

    assert model_price == expected_price
    