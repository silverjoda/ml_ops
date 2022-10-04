import httplib2

# These aren't needed, just for this example

from bs4 import BeautifulSoup
SCRIPTING_KEY = "jtpsb0vor47z3mv"
SERIES_CODE_LOAD = 'B0610'
import time
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.ar_model import AutoReg


def read_elexon(date_from, date_to, series_code):
    # Time delta of 1 day. We'll use this to iterate over the dates
    delta = datetime.timedelta(days=1)

    # Pandas dataframe
    df = pd.DataFrame()

    # Iterate over range of dates and get the data
    while (date_from <= date_to):
        # Build the URL
        url = f'https://api.bmreports.com/BMRS/{series_code}/v1?APIKey={SCRIPTING_KEY}&SettlementDate={date_from}&Period=*&ServiceType=xml'
        http_obj = httplib2.Http()

        # Get the data
        resp, content = http_obj.request(uri=url,
                                        method='GET',
                                        headers={'Content-Type': 'application/xml; charset=UTF-8'}, )

        # Dummy data
        # with open('testdata.xml', 'r') as f:
        #    content = f.read()

        # Make XML into a BeautifulSoup object
        content = BeautifulSoup(content, 'xml')

        # Parse the desired quantity out of the XML
        quantities = [int(d.text) for d in content.find_all('quantity')]
        settlement_period = [int(d.text) for d in content.find_all('settlementPeriod')]

        # Sort the data according to settlement period just to make sure that we have the data in the right order.
        items_sorted = [(y, x) for y, x in sorted(zip(settlement_period, quantities), key=lambda pair: pair[0])]
        settlement_period_sorted, quantities_sorted = list(zip(*items_sorted))

        # Add data to pandas dataframe
        d = {"settlement_date": [date_from] * len(quantities), "settlement_period" : settlement_period_sorted, "quantity": quantities_sorted}
        df = pd.concat([df, pd.DataFrame(data=d)], ignore_index=True)

        # Increase by 1 day
        date_from += delta

        # Sleep for 1 second to avoid overloading the API
        time.sleep(1)

    # Save the pandas dataframe
    df.to_xml(f'data/series_{series_code}.xml', index=False)

def analyze_dataset(filename):
    # Read data
    df = pd.read_xml(filename)

    # Change date str to datetime object
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])

    # Plot the data
    quantity_arr = np.array(df['quantity'])
    #plt.plot(quantity_arr)
    #plt.show()

    # Addfuller stationarity test
    #res = adfuller(df['quantity'])
    #print(res)

    # Basic info
    #print(df.head(n=10))
    #print(df.info())
    #print(df.describe())

    # How many days with an atypical amount of settlement periods?
    sd_counts = df.groupby('settlement_date')['settlement_period'].count()
    sd_counts_irreg = sd_counts[sd_counts != 48]

    # Days with an atypical amount of settlement periods
    #print(len(sd_counts_irreg))

    # Show data from days where amount of settlement periods is atypical
    # for day, cnt in sd_counts_irreg.items():
    #     if cnt < 48:
    #         print(df[df['settlement_date'] == day])
    #         print(cnt)
    #         #plt.plot(df[df['settlement_date'] == day]['quantity'])
    #         #plt.show()

    # Minor issue 1): Settlement values from 19 of the days are more or less than the standard 48 settlement periods.
    # The values of a specific settlement can be missing or can be duplicate, both in any position.
    # Solution: fill in missing values and average duplicate so they do not affect seasonality.
    # Minor issue 2): There are noisy dips in the data, we can filter those out by replacing the value with average of neighboring values.

    # Group all duplicate settlement days and periods to their mean
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False).mean()

    # Get new counts
    sd_counts = df.groupby('settlement_date')['settlement_period'].count()
    sd_counts_irreg = sd_counts[sd_counts != 48]

    # Make new empty dataframe that we will fill with the missing values
    df_mv = pd.DataFrame(columns=df.columns)

    # Go over each sd from 1 to 48 and fill missing entries with nan quantities
    for day, cnt in sd_counts_irreg.items():
        df_day = df[df['settlement_date'] == day]

        # Find out missing periods
        sp_missing = np.setdiff1d(np.arange(1,49), df_day["settlement_period"].values)

        # Make new dataframe
        data = {"settlement_date" : [day] * len(sp_missing), "settlement_period" : sp_missing, "quantity" : None}
        df_missing = pd.DataFrame(data=data)

        df_mv = pd.concat([df_mv, df_missing])

    # Add the dataframe with missing values
    df = pd.concat([df, df_mv], ignore_index=True)
    df = df.sort_values(by=["settlement_date", "settlement_period"], ignore_index=True)

    # Get low range outlier values (TODO: Change to more sophisticated method. Use quantile(0.001) BUT PER WEEK, not global!!)
    #outlier_thresh = df["quantity"].quantile(0.001)
    outlier_thresh = 8200

    # Change some very low value outliers to nan
    df.loc[df["quantity"] < outlier_thresh, "quantity"] = None

    # Intepolate added nan values
    df["quantity"] = df["quantity"].interpolate(limit_direction="both")

    # Show the days where there are huge dips in the data. On what hours do the dips occur? Are those days linked with some special events?
    # After analysis: It looks like it's just noise

    #stl = STL(df.loc[(df["settlement_date"] >= datetime.datetime(2022, 8, 1)) & (df["settlement_date"] <= datetime.datetime(2022, 9, 1))]["quantity"], period=48, robust=True)
    #res = stl.fit()
    #fig = res.plot()
    #plt.show()
    #exit()

    # f = plt.figure(figsize=(6, 6))
    # gs = f.add_gridspec(1,7)
    #
    # for i in range(3,10):
    #     start_date = datetime.datetime(2022, i, 1)
    #     end_date = datetime.datetime(2022, i+1, 1)
    #     stl = STL(df.loc[(df["settlement_date"] >= start_date) &
    #                      (df["settlement_date"] <= end_date)]["quantity"], period=48, robust=True)
    #     res = stl.fit()
    #     day_seqs = np.array(res.seasonal.values).reshape(-1, 48)
    #     day_seqs_list = [list(ds) for ds in day_seqs]
    #
    #     day_seqs_df = pd.DataFrame(data={"settlement_period" : list(range(1, 49)) * int(len(res.seasonal.values) / 48), "quantity" : np.array(res.seasonal.values)})
    #
    #     with sns.axes_style("darkgrid"):
    #         ax = f.add_subplot(gs[0, i-3])
    #         sns.lineplot(data=day_seqs_df, x="settlement_period", y="quantity")
    #
    #     plt.ylim(-12000, 7500)
    #
    # plt.show()

    # Split data into train (march-august) and test (september)
    # print(pd.to_datetime(df["settlement_date"], infer_datetime_format=True).info())
    # exit()

    # Normalize the data

    # Analyze data in the frame of a week (month):
    # Decompose the data into trend, seasonality and noise
    # Perform fourier analysis with the iterative noise rejection method
    # Try SARIMAX
    # Try Ready sota method such as Prophet

    # Analyze all the data
    # Try LSTM or similar NN on whole dataset with several days of history and 12h horizon
    # Try incorporate exogenous variables such as information from other available series in the API

    start_date = datetime.datetime(2022, 3, 1)
    end_date = datetime.datetime(2022, 3, 2)
    val_start_date = datetime.datetime(2022, 3, 3)
    val_end_date = datetime.datetime(2022, 3, 5)
    trn_df = df.loc[(df["settlement_date"] >= start_date) &
                    (df["settlement_date"] <= end_date)]
    val_df = df.loc[(df["settlement_date"] >= val_start_date) &
                    (df["settlement_date"] <= val_end_date)]

    # TODO: Add exog variables to arima
    stlf = STLForecast(trn_df["quantity"], ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=48, seasonal=3, trend_deg=1)
    stlf_res = stlf.fit()

    print(stlf_res.summary())

    # TODO: Add evaluation

    forecast = stlf_res.forecast(48)
    plt.plot(trn_df["quantity"], 'b')
    plt.plot(val_df["quantity"], 'g')
    plt.plot(forecast, 'r--')
    plt.show()

def date_test():
    # consider the start date as 2021-march 1 st
    start_date = datetime.date(2022, 3, 1)

    # consider the end date as 2021-march 1 st
    end_date = datetime.date(2022, 9, 30)

    # delta time
    delta = datetime.timedelta(days=1)

    # iterate over range of dates
    while (start_date <= end_date):
        print(str(start_date))
        start_date += delta

def testpd():
    ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
                         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
                'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
                'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
                'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
    df = pd.DataFrame(ipl_data)
    print(df)
    print(df.groupby('Team')['Points'].count())

def main():
    #read_elexon(datetime.date(2022, 3, 1), datetime.date(2022, 9, 30), SERIES_CODE_LOAD)
    analyze_dataset("data/series_B0610.xml")
    #testpd()



if __name__ == "__main__":
    main()

