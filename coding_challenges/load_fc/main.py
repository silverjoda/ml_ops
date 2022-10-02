import httplib2

# These aren't needed, just for this example

from bs4 import BeautifulSoup
SCRIPTING_KEY = "jtpsb0vor47z3mv"
SERIES_CODE_LOAD = 'B0610'
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        d = {"settlement_date": [str(date_from)] * len(quantities), "settlement_period" : settlement_period_sorted, "quantity": quantities_sorted}
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

    # Plot the data
    quantity_arr = np.array(df['quantity'])
    #plt.plot(quantity_arr)
    #plt.show()

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

    # Fill in missing values
    for day, cnt in sd_counts_irreg.items():
        df_day = df[df['settlement_date'] == day]

        #print(df_day["settlement_period"].values)
        #exit()

        # Find all duplicates and replace with average

        # Go over each sd from 1 to 48 and fill missing values

    # Show the days where there are huge dips in the data. On what hours do the dips occur? Are those days linked with some special events?

    # Normalize the data

    # Analyze data in the frame of a week (month):
    # Decompose the data into trend, seasonality and noise
    # Perform fourier analysis with the iterative noise rejection method
    # Try SARIMAX
    # Try Ready sota method such as Prophet

    # Analyze all the data
    # Try LSTM or similar NN on whole dataset with several days of history and 12h horizon
    # Try incorporate exogenous variables such as information from other available series in the API


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

