from graph import Graph
import networkx as nx
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import scipy

def get_eigens(matrix, k, graph_name, graph_type, folder):
    vals, vecs = eigs(matrix.asfptype(), k=k, sigma=0, OPpart='r')
    filename = folder+"k_"+str(k)+"_"+graph_name + "_" + graph_type + ".npy"
    np.save(filename,vecs)
    return vecs

def get_generalized_eigens(matrix, degree, k, graph_name, graph_type, folder):
    vals, vecs = eigs(matrix.asfptype(), k=k, M=degree, sigma=0, OPpart='r')
    filename = folder+"k_"+str(k)+"_"+graph_name + "_" + graph_type + ".npy"
    np.save(filename,vecs)
    return vecs

def save_eigens(fnames_competition, k_s):
    for i, fname in enumerate(fnames_competition):
        graph = Graph(fname=fname)
        laplacian = nx.laplacian_matrix(graph.G)
        get_eigens(laplacian.asfptype(), k_s[i], fname, "laplacian", './eigenvectors/laplacian/')
        
def save_normalized_eigens(fnames_competition, k_s):
    for i, fname in enumerate(fnames_competition):
        graph = Graph(fname=fname)
        laplacian = nx.normalized_laplacian_matrix(graph.G)
        get_eigens(laplacian.asfptype(), k_s[i], fname, "normalized_laplacian", './eigenvectors/normalized_laplacian/')
        
def save_generalized_eigens(fnames_competition, k_s):
    for i, fname in enumerate(fnames_competition):
        graph = Graph(fname=fname)
        laplacian = nx.laplacian_matrix(graph.G)
        degree = scipy.sparse.diags(np.array(graph.G.degree())[:,1])
        get_generalized_eigens(laplacian.asfptype(), degree.asfptype(), k_s[i], fname, "generalized_eigenproblem", './eigenvectors/generalized_eigenproblem/')

def save_eigenvalues_for_all_graphs():
    fnames_competition = ['ca-GrQc',
                      'Oregon-1',
                      'soc-Epinions1',
                      'web-NotreDame',
                      'roadNet-CA'
                      ]
    k_s = [100,100,100,100,100]

    save_eigens(fnames_competition, k_s)
    save_normalized_eigens(fnames_competition, k_s)
    save_generalized_eigens(fnames_competition, k_s)