import numpy as np
import matplotlib.pyplot as plt


def calc_P_co(C_p, data):
    """
    Returns the probability that for the current price C_p, and historical data "data", that the mean consumption was less than expected.
    :param C_p: Float, current price
    :param data: Array, past prices
    :return: Float probability in range [0,1]
    """


    pass

def decide_whether_to_buy(C_p, P_co, V_co, thresh):
    """
    Decide whether it makes sense to buy, given the current price C_p and probability P_co that consumption will be less than mean.
    :param C_p: Float, current price
    :param P_co: Float, probability [0,1]
    :param thresh: Float, value of gain below which we don't consider the transaction to be interesting
    :return: Boolean: Buying decision
    """

    return V_co * P_co - C_p - thresh > 0

def plot_expected_gains_based_on_just_current_price(data, V_co):
    """
    Plot predicted probability of compensation (sell high) and the expected gains, given JUST the current price (without historical data).
    Perhaps there is a trend here, but unlikely.
    :param data: Data array used for the plots
    :param V_co: Value at which we will sell high
    :return:
    """

    price_arr = None
    pred_arr = None

    # Make pricing bins
    price_max = 100
    n_bins = 10
    price_bins = np.linspace(0, price_max, n_bins)
    price_bins_dict = {}
    pred_bins_dict = {}

    # Go over price bins and add all current prices and predictions that correspond to that bin
    for i in range(n_bins - 1):
        bot_lim = price_bins[i]
        top_lim = price_bins[i+1]
        price_bins_dict[i] = []
        pred_bins_dict[i] = []
        for price, pred in zip(price_arr, pred_arr):
            if bot_lim <= price < top_lim:
                price_bins_dict[i].append(price)
                pred_bins_dict[i].append(pred)

    pred_bin_list = []
    expected_gain_list = []
    # Go over both dicts and for each current price bin calculate the probability of compensation
    for i in range(n_bins - 1):
        prob_pred = price_bins_dict[i].sum() / float(len(price_bins_dict[i]))
        pred_bin_list.append(prob_pred)
        expected_gain_list.append(V_co * prob_pred - (pred_bin_list[i:i+1].sum() * 0.5))
        # TODO:  We could also use frequencies to quantify the certainty of prediction

    plt.plot(price_bins, pred_bin_list, 'g-', label='prob of selling high')
    plt.plot(price_bins, expected_gain_list, 'r-', label='expected gains')
    plt.title("Probability of compensation and expected gains given just current price")
    plt.legend()
    plt.show()

# Define parameters:
C_p = 45. # Current market price
data = [] # Historical data
V_co = 90. # Compensation value declared by operator
thresh = 0. # Threshold below which we don't consider the transaction to be interesting

P_co = calc_P_co(C_p, data)
should_buy = decide_whether_to_buy(C_p, P_co, V_co, thresh)

