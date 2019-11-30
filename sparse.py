import networkx as nx
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def sparse_partitioning(G, k, unique_nodes):

    print(f"Creating laplacian")
    laplacian = nx.laplacian_matrix(G)

    print(f"Calculating eigenvalues and vectors")
    vals, vecs = eigs(laplacian.asfptype(), k=k, sigma=0)

    print(f"K-means clustering")
    labels = KMeans(init='k-means++', n_clusters=k).fit_predict(vecs)

    print(f"Testing conductance")
    total_conductance = 0
    for i in range(k):
        idx = np.where(labels == i)[0]
        conductance = nx.algorithms.cuts.conductance(G, idx)
        total_conductance += conductance
        print("Conductance of cluster", i, ":", conductance)
    print(total_conductance)

    print(f"Writing values to df")
    df = pd.DataFrame({'vertexID': unique_nodes, 'clusterID': labels})

    return df
