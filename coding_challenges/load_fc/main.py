import httplib2

# These aren't needed, just for this example

from bs4 import BeautifulSoup
SCRIPTING_KEY = "jtpsb0vor47z3mv"
SERIES_CODE_LOAD = 'B0610'
import datetime
import pandas as pd
import numpy as np


def read_elexon(date_from, date_to, series_code):
    # Time delta of 1 day. We'll use this to iterate over the dates
    delta = datetime.timedelta(days=1)

    # Pandas dataframe
    df = pd.DataFrame()

    # Iterate over range of dates and get the data
    while (date_from <= date_to):
        # Build the URL
        url = f'https://api.bmreports.com/BMRS/{series_code}/v1?APIKey={SCRIPTING_KEY}&SettlementDate=2015-03-01&Period=*&ServiceType=xml'
        http_obj = httplib2.Http()

        # Get the data
        # resp, content = http_obj.request(uri=url,
        #                                 method='GET',
        #                                 headers={'Content-Type': 'application/xml; charset=UTF-8'}, )

        # Dummy data
        with open('testdata.xml', 'r') as f:
           content = f.read()

        # Make XML into a BeautifulSoup object
        content = BeautifulSoup(content, 'xml')

        # Parse the desired quantity out of the XML
        quantities = [int(d.text) for d in content.find_all('quantity')]
        settlement_period = [int(d.text) for d in content.find_all('settlementPeriod')]

        assert len(quantities) == len(settlement_period) == 48

        # Sort the data according to settlement period just to make sure that we have the data in the right order.
        items_sorted = [(y, x) for y, x in sorted(zip(settlement_period, quantities), key=lambda pair: pair[0])]
        settlement_period_sorted, quantities_sorted = list(zip(*items_sorted))

        # Add data to pandas dataframe
        d = {"settlement_date": [str(date_from)] * len(quantities), "settlement_period" : settlement_period_sorted, "quantity": quantities_sorted}
        df = pd.concat([df, pd.DataFrame(data=d)], ignore_index=True)

        # Increase by 1 day
        date_from += delta

    # Save the pandas dataframe
    df.to_xml(f'data/series_{series_code}.xml', index=False)

def analyze_dataset(filename):
    df = pd.read_xml(filename)
    print(df.head(n=10))
    print(df.info())

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

def main():
    #read_elexon(datetime.date(2021, 3, 1), datetime.date(2021, 9, 30), SERIES_CODE_LOAD)
    analyze_dataset("data/series_B0610.xml")

if __name__ == "__main__":
    main()

