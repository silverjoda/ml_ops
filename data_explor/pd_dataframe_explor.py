import pandas as pd
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)

x_df = pd.DataFrame(X)