import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta

def main():
    """
    Takes original dataframe and splits it by months
    """

    data = pd.read_csv('./datasets/car_prices.csv',
                        sep=',',
                        error_bad_lines=False,
                        parse_dates=['saledate'])
                        
    data['saledate'] = data['saledate'].apply(lambda t: datetime.strftime(t, "%Y-%m-%d %H:%M"))
    data['saledate'] = data['saledate'].apply(lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M"))

    period = datetime(year=2014, month=12, day=1).date()
    td_max = data.saledate.max().date()

    while period <= td_max:
        mask = (data['saledate'] < np.datetime64(period  + relativedelta(months=1))) \
                &(data['saledate'] >= np.datetime64(period))
        
        filename = f"car-prices-{period.year}-{period.month}"
        print(filename,
            period + relativedelta(months=1),
            np.sum(mask))
        data[mask].to_csv(f"./datasets/{filename}.csv", sep=",", index=False)
        period = period + relativedelta(months=1)

if __name__=="__main__":
    main()