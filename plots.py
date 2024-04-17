import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def diversity(array:list)->float:

    data_matrix = [blob.anatomy() for blob in array]
    pca = PCA(n_components=4) # max 6 components
    trans_data = pca.fit_transform(data_matrix)

    db = DBSCAN(eps=0.1, min_samples=40,).fit(trans_data)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)

    H_ = 0
    Total = len(labels) - n_noise_
    # print(Total)
    for i in range(n_clusters_):
        freq = np.count_nonzero(labels==i)
        H_ -= freq/Total * np.log2( freq/Total )
    
    return H_
