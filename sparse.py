
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import diags


def sparse_partitioning(G, k, unique_nodes, eigen_k, load_vectors=True, graph_name=None, mode=None, vecs_full=None):

    if vecs_full is None:
        if not load_vectors:
            if mode == 'normalized':
                laplacian = nx.normalized_laplacian_matrix(G)
                vals, vecs = sparse.linalg.eigs(laplacian.asfptype(), k=100, sigma=0, OPpart='r')
                vecs = np.real(vecs)
                vecs = normalize(vecs, axis=1, norm='l1')

            elif mode == 'generalized':
                laplacian = nx.laplacian_matrix(G)
                degree = diags(np.array(G.degree())[:, 1]).asfptype()
                vals, vecs = sparse.linalg.eigs(laplacian.asfptype(), k=100, M=degree, sigma=0, OPpart='r')

            else:
                laplacian = nx.laplacian_matrix(G)
                vals, vecs = sparse.linalg.eigs(laplacian.asfptype(), k=100, sigma=0, OPpart='r')

        else:
            if mode == 'laplacian':
                vecs = np.load('./eigenvectors/laplacian/k_100_' + graph_name + '_laplacian.npy')
            elif mode == 'generalized':
                vecs = np.load('./eigenvectors/generalized_eigenproblem/k_100_' + graph_name + '_generalized_eigenproblem.npy')
            elif mode == 'normalized':
                eigs = np.load('./eigenvectors/normalized_laplacian/k_100_' + graph_name + '_normalized_laplacian.npy')
                eigs = np.real(eigs)
                vecs = normalize(eigs, axis=1, norm='l1')

            else:
                print(f"Partitioning mode not found {mode}")
                return

        vecs_full = np.real(vecs)
        vecs = vecs_full[:, :eigen_k]
    else:
        vecs = vecs_full[:, :eigen_k]

    labels = KMeans(init='k-means++', n_clusters=k, n_init=20, n_jobs=2).fit_predict(vecs)

    total_cut = 0
    for i in range(k):
        idx = np.where(labels == i)[0]
        cut = nx.algorithms.cuts.cut_size(G, idx) / len(idx)
        total_cut += cut

    df = pd.DataFrame({'vertexID': unique_nodes, 'clusterID': labels})

    return df, total_cut, vecs_full

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