import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def normalized_spectral_clustering(unique_nodes, list_of_edges, k):
    """
    algorithm 3: normalized spectral clustering
    input graph adjacency matrix A, number k
    1. form diagonal matrix D
    2. form normalized Laplacian L0 = I 􀀀 D􀀀1=2AD􀀀1=2
    3. compute the first k eigenvectors u1; : : : ; uk of L0
    4. form matrix U 2 Rnk with columns u1; : : : ; uk
    5. normalize U so that rows have norm 1
    6. consider the i-th row of U as point yi 2 Rk ; i = 1; : : : ; n,
    7. cluster the points fyigi=1;:::;n into clusters C1; : : : ;Ck
    e.g., with k-means clustering
    output clusters A1; : : : ;Ak with Ai = fj j yj 2 Cig
    """

    """
    list_of_edges in this format
    X = np.array([
    [1, 3], [2, 1], [1, 1],
    [3, 2], [7, 8], [9, 8],
    [9, 9], [8, 7], [13, 14],
    [14, 14], [15, 16], [14, 15]
])
    """

    n = len(unique_nodes)
    # similarity matrix
    sim_matrix = np.zeros((n, n))

    W = pairwise_distances(list_of_edges, metric="euclidean")
    vectorizer = np.vectorize(lambda x: 1 if x < 5 else 0)
    W = np.vectorize(vectorizer)(W)
    print(W)

    # degree matrix
    D = np.diag(np.sum(np.array(W.todense()), axis=1))
    print('degree matrix:')
    print(D)

    # laplacian matrix
    L = D - W
    print('laplacian matrix:')
    print(L)

    e, v = np.linalg.eig(L)
    # eigenvalues
    print('eigenvalues:')
    print(e)
    # eigenvectors
    print('eigenvectors:')
    print(v)

    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(e)
    ax1.title.set_text('eigenvalues')
    i = np.where(e < 10e-6)[0]
    ax2 = plt.subplot(122)
    plt.plot(v[:, i[0]])
    fig.tight_layout()
    plt.show()

    U = np.array(v[:, i[1]])
    km = KMeans(init='k-means++', n_clusters=k)
    km.fit(U)
    clusters = km.labels_

    df = pd.DataFrame({'vertexID':unique_nodes, 'clusterID': clusters})

    return df
