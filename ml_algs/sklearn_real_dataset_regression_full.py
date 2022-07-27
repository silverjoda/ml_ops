import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.pipeline import make_pipeline
import matplotlib
#matplotlib.use("TkAgg")
#matplotlib.interactive(True)
import seaborn as sn
np.set_printoptions(precision=3, suppress=True)


# Full classification

def evaluate_algo(X, y, pipe):
    y_hat_tst = pipe.predict(X)
    # Doesn't say anything, need to
    print("MSE", sklearn.metrics.mean_squared_error(y, y_hat_tst))

def train_algo(X, y):
    # Define and fit GradientBoost classifier
    pipe = make_pipeline(StandardScaler(), sklearn.ensemble.GradientBoostingRegressor())
    pipe.fit(X, y)

    return pipe

def preprocess_dataset(X, y):
    ds_df = pd.DataFrame(X)
    ds_df["targets"] = y

    # Remove nans
    ds_df.dropna(inplace=True)

    def remove_outliers_by_std(ds_df):
        # Remove outliers by rejecting values outside n_std deviations
        for col in ds_df.columns:
            mean = ds_df[col].mean()
            std = ds_df[col].std()

            n_std = 3
            ds_df = ds_df[(ds_df[col] - mean) < std * n_std]
        return ds_df

    def remove_outliers_by_zscores(ds_df):
        z_scores = zscore(ds_df)
        abs_z_scores = np.abs(z_scores)
        n_z = 3
        filtered_entries = (abs_z_scores < n_z).all(axis=1)
        return ds_df[filtered_entries]

    ds_df = remove_outliers_by_std(ds_df)
    #ds_df_z = remove_outliers_by_zscores(ds_df)

    # Normalize features (whiten)
    X = ds_df[filter(lambda x: x != "targets", ds_df.columns)].to_numpy()
    y = ds_df["targets"].to_numpy()[:, np.newaxis]

    return X, y

def explore_dataset(X, y):
    assert len(X) == len(y)
    print(f"N_values in dataset: {len(X)}")
    print(f"N_features in dataset: {len(X[0])}")

    # Make X into pandas frame and check it out
    X_df = pd.DataFrame(X)
    print("Description")
    print(X_df.describe())
    print("Head")
    print(X_df.head(n = 5))

    total_nan_rows = X_df.isnull().any(axis=1).sum()
    print(f"Total number of rows with nan value: {total_nan_rows}")

    # Covariance matrix
    cor_mat_np = np.corrcoef(X.T)
    #sn.heatmap(cor_mat_np, annot=True, fmt='g')
    #plt.title("Correlation matrix on the data")
    #plt.show()

    print("Correlation matrix")
    print(cor_mat_np)
    print()

    # Do SVD to analyze eigenvalues
    _, S, _ = np.linalg.svd(X[:1000].T)
    print("Eigenvalues of the individual features")
    print(S)
    print()

    # TODO: Does eigenvalue magnitude represent information content?
    print("Eigenvalues of the individual features as cumulative percentages")
    print(100 * S.cumsum() / S.sum())
    print()

    # Plot a 3D PCA
    #plot_3D_PCA(X, y)

    # Targets:
    tar_df = pd.DataFrame(y, columns=["targets"])
    print("Targets description")
    print(tar_df.describe())
    print()

    print("Targets head")
    print(tar_df.head())
    print()

    # Show histogram of targets
    #plt.hist(y, bins=10)
    #plt.show()

def plot_3D_PCA(X, y):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    ex_variance = np.var(X_pca, axis=0)
    ex_variance_ratio = ex_variance / np.sum(ex_variance)
    print(f"ex_variance_ratio: {ex_variance_ratio}")

    Xax = X_pca[:, 0]
    Yax = X_pca[:, 1]
    Zax = X_pca[:, 2]

    y_mean, y_min, y_max = np.mean(y), np.min(y), np.max(y)
    cmap = lambda x : 1 * (x - y_mean) / (y_max - y_min)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')

    ax.scatter(Xax, Yax, Zax, c=list(map(cmap, y)), s=40)

    # for loop ends
    ax.set_xlabel("First Principal Component", fontsize=14)
    ax.set_ylabel("Second Principal Component", fontsize=14)
    ax.set_zlabel("Third Principal Component", fontsize=14)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Load real dataset
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    # Explore dataset
    #explore_dataset(X, y)

    # Dataset Preprocessing
    X_pp, y_pp = preprocess_dataset(X, y)

    # Split dataset
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X_pp, y_pp, random_state=1)

    # Training:
    pipe = train_algo(X_train, y_train)

    # Evaluate result
    evaluate_algo(X_test, y_test, pipe)
