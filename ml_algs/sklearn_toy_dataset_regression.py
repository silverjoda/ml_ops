import pandas as pd
import sklearn
import sklearn.datasets
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load dataset
    ds = sklearn.datasets.load_diabetes()
    #print(ds.DESCR)

    # Make into pandas dataframe
    ds_feat_df = pd.DataFrame(ds.data, columns=ds.feature_names)
    ds_tar_df = pd.DataFrame(ds.target, columns=["target"])
    ds_df = pd.concat((ds_feat_df, ds_tar_df))

    print("Description and head")
    print(ds_df.describe())
    print(ds_df.head())

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

    # Make PCA from features
    pca = PCA(n_components=2, whiten=True)
    pca.fit(ds.data, y=ds.target)
    ds_lowdim = pca.transform(ds.data)

    fig = plt.figure()
    plt.title("Low dimensional plot of the Diabetes dataset")
    plt.scatter(ds_lowdim[:, 0], ds_lowdim[:, 1])
    plt.show()

