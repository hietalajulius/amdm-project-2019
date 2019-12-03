
import matplotlib.pyplot as plt
import networkx as nx

from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


def sparse_partitioning(G, k, unique_nodes, eigen_k, load_vectors=True, graph_name=None, mode=None, vecs_full=None):
    from scipy.sparse.linalg import eigs
    if vecs_full is None:
        print(f"Creating laplacian")
        if not load_vectors:
            if mode == 'normalized':
                laplacian = nx.normalized_laplacian_matrix(G)
            else:
                laplacian = nx.laplacian_matrix(G)
            print(f"Calculating eigenvalues and vectors")
            vals, vecs = eigs(laplacian.asfptype(), k=100, sigma=0, OPpart='r')
            filename = "k_" + str(k) + "_" + graph_name + "_" + mode + ".npy"
            np.save(filename, vecs)
            # vals, vecs = eigsh(laplacian.asfptype(), k=100, sigma=0, which='LM')

            vecs_full = vecs
            vecs = vecs_full[:, :eigen_k]
            # plot_eigenvalues(vals, vecs)
        else:
            if mode == 'laplacian':
                vecs = np.load('./eigenvectors/laplacian/k_100_' + graph_name + '_laplacian.npy')
            elif mode == 'generalized':
                vecs = np.load('./eigenvectors/generalized_eigenproblem/k_100_' + graph_name + '_generalized_eigenproblem.npy')
            elif mode == 'normalized':
                eigs = np.load('./eigenvectors/normalized_laplacian/k_100_' + graph_name + '_normalized_laplacian.npy')
                vecs = eigs / np.linalg.norm(eigs, ord=2, axis=1, keepdims=True)
            else:
                print(f"Partitioning mode not found {mode}")
                return
            # print(f"eigenvec shape is {vecs.shape}")
            vecs_full = np.real(vecs)
            vecs = vecs_full[:, :eigen_k]
    else:
        vecs = vecs_full[:, :eigen_k]
    # print(f"eigenvec shape is {vecs.shape}")

    # print(f"Partitioning with kmeans")
    labels = KMeans(init='k-means++', n_clusters=k, n_init=20, n_jobs=4).fit_predict(vecs)

    #print(f"Testing conductance")
    total_conductance = 0
    for i in range(k):
        idx = np.where(labels == i)[0]
        conductance = nx.algorithms.cuts.cut_size(G, idx) / len(idx)
        total_conductance += conductance
        # print(f"Conductance of cluster {i}: {round(conductance, 6)}")
    print(f"total_conductance with k {k} and eigen k {eigen_k} is {round(total_conductance, 4)}")

    #print(f"Writing values to df")
    # print(f"Size of unique_nodes is {len(np.unique(unique_nodes))} vs clusterID {len(labels)}")
    df = pd.DataFrame({'vertexID': unique_nodes, 'clusterID': labels})

    return df, total_conductance, vecs_full

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