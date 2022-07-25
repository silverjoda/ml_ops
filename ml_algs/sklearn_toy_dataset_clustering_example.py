import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.datasets

# Dataset
ds = sklearn.datasets.load_iris()
ds_df = pd.DataFrame(ds, columns=ds.feature_names)

print(ds.data.shape, len(np.unique(ds.target)))

# Clustering
clf = sklearn.cluster.KMeans(n_clusters=3)
clf.fit(ds.data)

rnd_indeces = np.random.randint(0, len(ds.data), 10)
print(clf.predict(ds.data[rnd_indeces]))
print(ds.target[rnd_indeces])

