
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


def sparse_partitioning(G, k, unique_nodes, eigen_k, load_vectors=False, graph_name=None, mode=None):

    #print(f"Creating laplacian")
    if not load_vectors:
        laplacian = nx.laplacian_matrix(G)
        #print(f"Calculating eigenvalues and vectors")
        vals, vecs = eigsh(laplacian.asfptype(), k=eigen_k, sigma=0)
        # plot_eigenvalues(vals, vecs)
    else:
        if mode == 'laplacian':
            vecs = np.load('./eigenvectors/laplacian/k_100_' + graph_name + '_laplacian.npy')
        elif mode == 'generalized':
            vecs = np.load('./eigenvectors/generalized_eigenproblem/' + graph_name + '_generalized_eigenproblem.npy')
        elif mode == 'normalized':
            eigs = np.load('./eigenvectors/normalized_laplacian/' + graph_name + '_normalized_laplacian.npy')
            vecs = eigs / np.linalg.norm(eigs, ord=2, axis=1, keepdims=True)
        else:
            print(f"Partitioning mode not found {mode}")
            return
        # print(f"eigenvec shape is {vecs.shape}")
        vecs = np.real(vecs)
        vecs = vecs[:, :eigen_k]
    print(f"eigenvec shape is {vecs.shape}")

    labels = KMeans(init='k-means++', n_clusters=k).fit_predict(vecs)

    #print(f"Testing conductance")
    total_conductance = 0
    for i in range(k):
        idx = np.where(labels == i)[0]
        conductance = nx.algorithms.cuts.cut_size(G, idx) / len(idx)
        total_conductance += conductance
        print(f"Conductance of cluster {i}: {round(conductance, 6)}")
    print(f"total_conductance with k {k} and eigen k {eigen_k} is {round(total_conductance, 4)}")

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