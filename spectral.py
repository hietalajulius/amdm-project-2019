from scipy.linalg import eigh
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import eye
from scipy.sparse import identity
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
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

    print(f"Creating adjacency matrix")
    # input graph adjacency matrix A, number k
    A = adjacency_matrix(list_of_edges, unique_nodes)
    # print("A.shape", A.shape)

    print(f"1. form diagonal matrix D")
    # 1. form diagonal matrix D
    n = len(unique_nodes)
    D = diagonal_degree_matrix(A, n)
    # print(f"D is {D}")

    print(f"2. form normalized Laplacian L0")
    #     2. form normalized Laplacian L0 = I - D^(-1/2)AD^(-1/2)
    # L0 = np.identity(n) - (D.dot(A)).dot(D)
    L0 = identity(n) - (D.dot(A)).dot(D)
    # L0 = eye(n).toarray() - (D.dot(A)).dot(D)
    # print(f"L0 is {L0}")

    print(f"3. compute the first k={k} eigenvectors u1,...,uk of L0")
    #     3. compute the first k eigenvectors u1,...,uk of L0
    # L0_arr = L0.toarray()
    # e, v = eigh(L0_arr, eigvals=(0, k-1))
    e, v = eigsh(L0, k=k)

    # eigenvalues
    #print('eigenvalues:')
    #print(e)
    # eigenvectors
    #print('eigenvectors:')
    #print(v)
    plot_eigenvalues(e, v)

    print(f"4-6. k-means clustering")
    #     4. form matrix U that R^(nxk) with columns u1, ... uk of L0
    #     5. normalize U so that rows have norm 1
    #     6. consider the i-th row of U as point yi € R^k,  i = 1, ... n,
    i = np.where(e < 10e-6)[0]
    #print(i)
    U = np.array(v[:, i[1]])
    U_norm = normalize(U, axis=1, norm='l1')
    # print(f"U shape is {U.shape}")

    #     7. cluster the points {yi}_ i=1,...,n into clusters C1,...,Ck
    km = KMeans(init='k-means++', n_clusters=k)
    km.fit(U_norm.reshape(-1, 1))
    clusters = km.labels_

    print(f"Data to dataframe")
    df = pd.DataFrame({'vertexID':unique_nodes, 'clusterID': clusters})
    # print(df)
    print(f"Spectral clustering done")
    return df

def adjacency_matrix(list_of_edges, unique_nodes):
    """

    :param list_of_edges:
    :param unique_nodes:
    :return:
    """
    n = len(unique_nodes)
    # adj_matrix matrix
    # adj_matrix = np.zeros((n, n), dtype=np.float32)
    # adj_matrix = csc_matrix((n, n), dtype=np.float32)
    adj_matrix = lil_matrix((n, n), dtype=np.float32)
    for v1, v2 in list_of_edges:
        # print(f"edge is {v1} - {v2}")
        i1 = np.where(unique_nodes == v1)[0][0]
        i2 = np.where(unique_nodes == v2)[0][0]
        adj_matrix[i1, i2] = 1
        adj_matrix[i2, i1] = 1
        # print(f"adj matrix found value in index1 {i1} index2 {i2}")

    # print(f"sim_matrix.shape {adj_matrix.shape}, sum is {np.sum(adj_matrix)}")
    # adj_mat_sparse = csc_matrix(adj_matrix)

    return adj_matrix.tocsc()

def diagonal_degree_matrix(adj, n):
    """

    :param adj:
    :return:
    """
    # diag = np.zeros([adj.shape[0], adj.shape[0]], dtype=np.float32) # basically dimensions of your graph
    # diag = csc_matrix((adj.shape[0], adj.shape[0]), dtype=np.float32).toarray()
    diag = lil_matrix((n, n), dtype=np.float32)

    rows, cols = adj.nonzero()
    for row, col in zip(rows, cols):
        diag[row, row] += 1
    # print(f"diagonal {diag}")

    # Calculate D^(-1/2)
    for row, col in zip(rows, cols):
        diag[row, row] = 1 / diag[row, row]**0.5

    # diag_sparse = csc_matrix(diag)

    return diag.tocsc()

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
