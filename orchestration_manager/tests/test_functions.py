import numpy as np
import pandas as pd
from datetime import datetime
from deepdiff import DeepDiff

import backend_services

class MockModel:
    def __init__(self, value):
        self.value = value

    def predict(self, data):
        n = len(data)

        return [self.value]*n

    def transform(self, data):
        return data
    
def test_read_file():

    date_file = datetime(year=2015, month=1, day=1)
    key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv"

    actual_data = backend_services.read_file(key)
    expected_data = pd.DataFrame()

    assert type(actual_data) ==type(expected_data)


def test_load_data():
    current_date = "2015-7-1"
    dt_current = datetime.strptime(current_date, "%Y-%m-%d")

    data_0 = backend_services.load_data(current_date, periods=0)
    
    data_1 = backend_services.load_data(current_date)
    data_1 = data_1.saledate.max()
    dt_data_1 = datetime.strptime(data_1, "%Y-%m-%d %H:%M:%S")

    data_periods = backend_services.load_data(current_date, periods=4)
    full_size = len(data_periods)
    unique_index = data_periods.index.nunique()

    assert data_0.saledate.max()[:7] == data_0.saledate.min()[:7]
    assert dt_data_1 < dt_current
    assert unique_index < full_size


def test_na_filter():
    
    test_data = [
        (2000,'toyota',"sienna","xle","SUV","automatic","4t3zf13c3yu301239","ca",2.3,198168.0,"white","beige","onemain remarketing services",2125,1000,"2015-01-14 04:30:00"),
        (2011,None, None, None,"SUV","automatic","5n3aa08a57n801070","ca",2.0,86721.0,"white","black","twin motor",14350,13000,"2015-01-06 12:30:00"),
        (2014,"BMW",None,None,"Convertible","automatic","wbayp9c57ed169262","ca",3.8,10736.0,"black","black","the hertz corporation",67000,65000,"2015-01-06 12:30:00"),
        (2015,None, None,"Luxury","Sedan","automatic","knalw4d42f6018633","ca",4.4,9684.0,"white","gray","kia motors america, inc",41000,41000,"2015-01-20 04:30:00"),
        (2013,"Hyundai","Elantra","GLS","Sedan","automatic",None,"ca",1.0,24120.0,"brown","beige","hyundai motor finance",11950,5200,"2015-01-13 12:00:00"),
        (2011,"Hyundai","Equus","Signature","Sedan","automatic","kmhgh4jf8bu042912","ca",None,36047.0,"black","black","hyundai motor finance",25500,26000,"2015-01-20 04:30:00")
    ]

    expected_data = [
        (2000,'toyota',"sienna","xle","SUV","automatic","4t3zf13c3yu301239","ca",2.3,198168.0,"white","beige","onemain remarketing services",2125,1000,"2015-01-14 04:30:00"),
        (2013,"Hyundai","Elantra","GLS","Sedan","automatic",None,"ca",1.0,24120.0,"brown","beige","hyundai motor finance",11950,5200,"2015-01-13 12:00:00"),
        (2011,"Hyundai","Equus","Signature","Sedan","automatic","kmhgh4jf8bu042912","ca",None,36047.0,"black","black","hyundai motor finance",25500,26000,"2015-01-20 04:30:00")
    ]

    columns = ["year","make","model","trim","body","transmission","vin","state","condition","odometer","color","interior","seller","mmr","sellingprice","saledate"]

    test_dataframe = pd.DataFrame(test_data, columns=columns)

    expected_dataframe = pd.DataFrame(expected_data, columns=columns)
    expected_dataframe.drop(['sellingprice'], inplace=True, axis=1)

    actual_dataframe, _ = backend_services.na_filter(test_dataframe)

    diff = DeepDiff(
            expected_dataframe.replace({np.nan: None}).to_dict(orient="records"),
            actual_dataframe.replace({np.nan: None}).to_dict(orient="records")
        )
    assert 'values_changed' not in diff
    assert 'types_changed' not in diff

def test_prediction():
    mock_model = MockModel(1000.0)
    
    test_record = (2013,"Hyundai","Elantra","GLS","Sedan","automatic",None,"ca",1.0,24120.0,"brown","beige","hyundai motor finance",11950,5200,"2015-01-13 12:00:00")
    columns = ["year","make","model","trim","body","transmission","vin","state","condition","odometer","color","interior","seller","mmr","sellingprice","saledate"]

    test_dict = dict(zip(columns, test_record))

    expected_prediciton = 1000.0
    actual_prediction = backend_services.prediction(test_dict, model = mock_model)

    assert expected_prediciton == actual_prediction