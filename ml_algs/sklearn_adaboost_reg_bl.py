import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # Load dataset
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)

    # Split dataset
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Define and fit automl classifier
    clf = AdaBoostRegressor()
    clf.fit(X_train, y_train)

    # Evaluate result
    y_hat_tst = clf.predict(X_test)
    print("MSE err", sklearn.metrics.mean_absolute_error(y_test, y_hat_tst))