from scipy.linalg import eigh
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
    # print(f"list_of_edges")
    # print(list_of_edges)
    n = len(unique_nodes)

    # input graph adjacency matrix A, number k
    A = adjacency_matrix(list_of_edges, unique_nodes)
    # print("A.shape", A.shape)

    # 1. form diagonal matrix D
    D = diagonal_degree_matrix(A)
    # print(f"D is {D}")

    #     2. form normalized Laplacian L0 = I - D^(-1/2)AD^(-1/2)
    L0 = np.identity(n) - np.dot(D, A).dot(D)
    # print(f"L0 is {L0}")

    #     3. compute the first k eigenvectors u1; : : : ; uk of L0
    e, v = eigh(L0, eigvals=(0, k-1))
    # eigenvalues
    # print('eigenvalues:')
    # print(e)
    # eigenvectors
    # print('eigenvectors:')
    # print(v)
    plot_eigenvalues(e, v)

    #     4. form matrix U that R^(nxk) with columns u1, ... uk of L0
    #     5. normalize U so that rows have norm 1
    #     6. consider the i-th row of U as point yi € R^k,  i = 1, ... n,
    i = np.where(e < 10e-6)[0]
    U = np.array(v[:, i[1]])
    # print(f"U shape is {U.shape}")

    #     7. cluster the points {yi}_ i=1,...,n into clusters C1,...,Ck
    km = KMeans(init='k-means++', n_clusters=k)
    km.fit(U.reshape(-1, 1))
    clusters = km.labels_

    df = pd.DataFrame({'vertexID':unique_nodes, 'clusterID': clusters})

    return df

def adjacency_matrix(list_of_edges, unique_nodes):
    """

    :param list_of_edges:
    :param unique_nodes:
    :return:
    """
    n = len(unique_nodes)
    # adj_matrix matrix
    adj_matrix = np.zeros((n, n))
    for v1, v2 in list_of_edges:
        # print(f"edge is {v1} - {v2}")
        i1 = np.where(unique_nodes == v1)[0][0]
        i2 = np.where(unique_nodes == v2)[0][0]
        adj_matrix[i1, i2] = 1
        adj_matrix[i2, i1] = 1
        # print(f"adj matrix found value in index1 {i1} index2 {i2}")

    # print(f"sim_matrix.shape {adj_matrix.shape}, sum is {np.sum(adj_matrix)}")
    return adj_matrix

def diagonal_degree_matrix(adj):
    """

    :param adj:
    :return:
    """
    diag = np.zeros([adj.shape[0], adj.shape[0]]) # basically dimensions of your graph
    rows, cols = adj.nonzero()
    for row, col in zip(rows, cols):
        diag[row, row] += 1
    # print(f"diagonal {diag}")

    # Calculate D^(-1/2)
    for row, col in zip(rows, cols):
        diag[row, row] = 1 / diag[row, row]**0.5

    return diag

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
