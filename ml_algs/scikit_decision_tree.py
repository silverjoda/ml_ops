import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    # Load dataset
    X = np.array([[1,1,4,5], [2,1,4,5], [3,1,4,5], [2,2,4,5], [4,1,4,5], [4,2,4,5], [4,3,4,5], [4,4,4,5], [3,2,4,5], [3,3,4,5]])
    y = np.array([0] * 4  + [1] * 6)

    # Split dataset
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=3, train_size=0.9)

    # Define and fit automl classifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Evaluate result
    y_hat = dt.predict(X_train)
    print(f"Xtrain: {X_train}, Predictions: {y_hat}")
    print("Accuracy score", sklearn.metrics.accuracy_score(y_train, y_hat))

    y_hat = dt.predict(X_test)
    print(f"Xtest: {X_test}, Predictions: {y_hat}")
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

    fig = dt.fit(X_train, y_train)
    sklearn.tree.plot_tree(dt)
    plt.show()
