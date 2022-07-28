import os

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def calc_P_co(data_arr_hours, horizon):
    """
    Returns the probability that for the current price C_p, and historical data "data", that the mean consumption was less than expected.
    :param C_p: Float, current price
    :param data: Array, past prices
    :return: Float probability in range [0,1]
    """

    X = []
    y = []
    # Make data_arr_hours into dataset
    for i in range(horizon, len(data_arr_hours) - 1):
        X.append(data_arr_hours[i-horizon:i])
        y.append(int(data_arr_hours[i+1] > 0)) # Make labels 0 or 1

    pipe = make_pipeline(StandardScaler(), linear_model.LogisticRegression())
    pipe.fit(X, y)

    # Now classify the last horizon features
    pred = np.exp(pipe.predict_log_proba(X[-1:])[0,0])

    print(f"classifier predicted the probability of positive consumption to be: {pred}")

    return pred

def decide_whether_to_buy(C_p, P_co, V_co, thresh):
    """
    Decide whether it makes sense to buy, given the current price C_p and probability P_co that consumption will be less than mean.
    :param C_p: Float, current price
    :param P_co: Float, probability [0,1]
    :param thresh: Float, value of gain below which we don't consider the transaction to be interesting
    :return: Boolean: Buying decision
    """

    return V_co * P_co - C_p - thresh > 0

def get_data():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "data/data.csv")
    data = pd.read_csv(filename)
    return data

def analyze_data(data):
    print("Data head")
    print(data.head())

    print("Data tail")
    print(data.tail())

    print("Data descr")
    print(data.describe())

def preprocess_data_into_hours(df):
    # Group by minutes (not working)
    #df.groupby(pd.Grouper(key='utc', freq='Min')).count()

    # Assume the data is correct, just take chunks of 60 into numpy array
    data_arr = df["val"].to_numpy()

    # Pad the data with 7 more minutes to get a full hour
    data_arr = np.concatenate((data_arr, np.zeros(7)))

    print()
    print(f"Hours of data: {data_arr.shape[0] / 60}")

    # convert into hours
    data_hours = []
    n_minutes_in_hour = 60
    for i in range(0, len(data_arr) - 1, n_minutes_in_hour):
        data_hours.append(data_arr[i:i+n_minutes_in_hour].sum())

    return data_hours


# Define parameters:
C_p = 45. # Current market price
data = [] # Historical data
V_co = 90. # Compensation value declared by operator
thresh = 0. # Threshold below which we don't consider the transaction to be interesting
horizon = 10

df = get_data()
analyze_data(df)

data_arr_hours = preprocess_data_into_hours(df)

# Get probability of settlement, given data
P_co = calc_P_co(data_arr_hours, horizon)

# Calculate using derived decision rule in task2.txt whether we should buy or do nothing.
should_buy = decide_whether_to_buy(C_p, P_co, V_co, thresh)

print(f"We should buy the 1MW: {should_buy}")






