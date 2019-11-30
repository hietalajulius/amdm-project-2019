
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


def sparse_partitioning(G, k, unique_nodes, eigen_k):

    #print(f"Creating laplacian")
    laplacian = nx.laplacian_matrix(G)

    #print(f"Calculating eigenvalues and vectors")
    vals, vecs = eigsh(laplacian.asfptype(), k=eigen_k, sigma=0)
    # plot_eigenvalues(vals, vecs)

    """
    print(f"normalise eigenvectors")
    #     4. form matrix U that R^(nxk) with columns u1, ... uk of L0
    #     5. normalize U so that rows have norm 1
    #     6. consider the i-th row of U as point yi € R^k,  i = 1, ... n,
    i = np.where(vals < 10e-6)[0]
    U = np.array(vecs[:, i[1]])
    U_norm = normalize(U, axis=1, norm='l1').reshape(-1, 1)

    print(f"K-means clustering")
    labels = KMeans(init='k-means++', n_clusters=k).fit_predict(U_norm)
    """
    labels = KMeans(init='k-means++', n_clusters=k).fit_predict(vecs)

    #print(f"Testing conductance")
    total_conductance = 0
    for i in range(k):
        idx = np.where(labels == i)[0]
        conductance = nx.algorithms.cuts.cut_size(G, idx) / len(idx)
        total_conductance += conductance
        print("Conductance of cluster", i, ":", conductance)
    print(f"total_conductance with k {k} and eigen k {eigen_k} is {round(total_conductance, 3)}")

    #print(f"Writing values to df")
    df = pd.DataFrame({'vertexID': unique_nodes, 'clusterID': labels})

    return df, total_conductance

def plot_eigenvalues(e, v):
    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(e)
    ax1.title.set_text('eigenvalues')
    i = np.where(e < 10e-6)[0]
    ax2 = plt.subplot(122)
    plt.plot(v[:, i[0]])
    fig.tight_layout()
    plt.show()