import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # Load dataset
    ds = sklearn.datasets.load_diabetes()

    # Split dataset
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(ds.data, ds.target, random_state=3, train_size=0.9)

    # Define and fit automl classifier
    dt = LinearRegression()
    dt.fit(X_train, y_train)

    # Evaluate result
    y_hat = dt.predict(X_train)
    #print(f"Xtrain: {X_train}, Predictions: {y_hat}")
    print("R2 score on trn", sklearn.metrics.r2_score(y_train, y_hat))
    print("MAE on trn", sklearn.metrics.mean_absolute_error(y_train, y_hat))

    y_hat_tst = dt.predict(X_test)
    #print(f"Xtest: {X_test}, Predictions: {y_hat_tst}")
    print("R2 score on trn", sklearn.metrics.r2_score(y_test, y_hat_tst))
    print("MAE score on trn", sklearn.metrics.mean_absolute_error(y_test, y_hat_tst))