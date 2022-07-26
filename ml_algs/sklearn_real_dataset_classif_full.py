import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.ensemble import AdaBoostClassifier

# Full classification




def evaluate_algo(X, y, clf):
    pass

def train_algo(X, y):
    pass

def preprocess_dataset(X, y):
    pass

def explore_dataset(X, y):
    print(f"N_values in dataset: {len(X)}")
    assert len(X) == len(y)

if __name__ == "__main__":
    # Load real dataset
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    # Explore dataset
    explore_dataset(X, y)

    # Split dataset
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Define and fit automl classifier
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    # Evaluate result
    y_hat_tst = clf.predict(X_test)
    print("Accuracy_score", sklearn.metrics.accuracy_score(y_test, y_hat_tst))