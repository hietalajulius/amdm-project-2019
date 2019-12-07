import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sparse import sparse_partitioning

class Graph:
    def __init__(self, fname=""):
        """

        :param fname:

        """
        self.fname = fname  # graph name

        self.algorithm = None  # give name of partition algorithm
        self.df_data = None  # input data as dataframe
        self.df_output = None  # datafame that contains list of vertices and their partition index k
        self.graph_edges = None
        self.list_of_edges = None
        self.list_of_nodes = None
        self.G = None  # networkx graph

        self.objective = None

        self.graphID = None
        self.numOfVertices = None
        self.numOfEdges = None
        self.k = None  # number of partitions

        self._read_file()
        self._init_graph()

    def get_networkx_graph(self):
        return self.G

    def _add_nodes(self):
        """

        :return:
        """

        nodes1 = self.df_data.loc[:, 'node1'].unique()
        nodes2 = self.df_data.loc[:, 'node2'].unique()
        unique_nodes = np.unique(np.concatenate((nodes1, nodes2), 0))
        self.list_of_nodes = unique_nodes
        self.G.add_nodes_from(unique_nodes)

        self.df_output = pd.DataFrame(data=unique_nodes, columns=['vertexID'])

    def _add_edges(self):
        """

        :return:
        """
        self.list_of_edges = self.df_data.loc[:, ['node1', 'node2']].values

        graph_edges = [tuple(x) for x in self.df_data.to_records(index=False)]
        self.graph_edges = graph_edges
        self.G.add_edges_from(graph_edges)

    def _init_graph(self):
        """

        :return:
        """
        self.G = nx.Graph()
        self._add_nodes()
        self._add_edges()

    def _calculate_edges_between(self, v, not_v):

        count = 0
        for v1, v2 in self.list_of_edges:
            if (v1 in v) and (v2 in not_v):
                count += 1
            elif (v1 in not_v) and (v2 in v):
                count += 1
        return count

    def _divide_vertices(self, i):

        vertexID = self.df_output['vertexID'].values
        clusterID = self.df_output['clusterID'].values
        mask = clusterID == i

        v, not_v = vertexID[mask], vertexID[~mask]

        return v, not_v

    def _read_file(self):
        """
        Read first line [# graphID numOfVertices numOfEdges k]
        Then read all edges to self.df_data
        :return:
        """

        fpath = os.getcwd()
        file_dir = os.path.join(os.path.join(fpath, 'graphs_processed'), f"{self.fname}.txt")


        with open(file_dir, "r", newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)

        _, self.graphID, self.numOfVertices, self.numOfEdges, self.k = row1[0].split(" ")
        self.graphID, self.numOfVertices, self.numOfEdges, self.k = self.graphID, \
                                                                    int(self.numOfVertices), \
                                                                    int(self.numOfEdges), \
                                                                    int(self.k)

        header_names = ['node1', 'node2']
        df = pd.read_csv(file_dir, sep=' ', skiprows=1,  header=None, names=header_names)
        self.df_data = df

    def draw_map(self):
        """
        Draws graph with nodes and edges
        https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html
        :return:
        """

        nx.draw(self.G, node_size=10)
        #plt.savefig(f"graph_{self.fname}.png")
        plt.suptitle(f"{self.fname}")
        plt.show()

    def draw_partitioned_map(self):
        """
        Draws graph with partitioned nodes and edges
        https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html
        :return:
        """

        pos = nx.spring_layout(self.G)
        cmap = plt.cm.rainbow(np.linspace(0, 1, self.k))

        for i in range(self.k):
            v, _ = self._divide_vertices(i)
            nodelist = v.tolist()
            node_color = cmap[i]
            color_2d = np.zeros((len(nodelist), 4)) + node_color.reshape(-1, 1).T

            nx.draw_networkx_nodes(self.G, pos,
                                   nodelist=nodelist,
                                   node_color=color_2d,
                                   node_size=10,
                                   alpha=0.8)

        nx.draw_networkx_edges(self.G, pos,
                               edgelist=self.graph_edges,
                               width=1, alpha=0.5, edge_color='grey')

        #plt.savefig(f"partitioned_graph_{self.fname}.png")
        plt.suptitle(f"{self.fname}")
        plt.show() 

    def partition_graph(self, precomputed_eigenvectors=True):
        """
        Write partition values to self.df_output
        self.df_output['vertexID'] = list_of_vertices
        self.df_output['clusterID'] = cluster_id_for_vertex

        :param algorithm: name of algorithm as string
        :return:
        """

        k0_list = np.arange(self.k, self.k + 50, 1)
        phi_min = 1000000

        modes = ['laplacian', 'generalized', 'normalized']
        colors = ['b', 'r', 'g']
        for i, spectral_mode in enumerate(modes):
            phi_list = []
            k_list = []
            vecs_full = None

            for k0 in k0_list:
                df, phi, vecs_full = sparse_partitioning(G=self.G,
                                                            k=self.k,
                                                            unique_nodes=self.list_of_nodes,
                                                            eigen_k=k0,
                                                            load_vectors=precomputed_eigenvectors,
                                                            graph_name=self.fname,
                                                            mode=spectral_mode,
                                                            vecs_full=vecs_full)
                k_list.append(k0)
                phi_list.append(phi)
                # Saving values if new record and save output
                if phi < phi_min:
                    phi_min = phi
                    self.df_output = df
                    self.write_output()

            if spectral_mode == 'laplacian':
                spectral_name = 'the algorithm utilizing the unnormalised Laplacian'
            elif spectral_mode == 'generalized':
                spectral_name = 'the algorithm utilizing the generalised eigenproblem'
            elif spectral_mode == 'normalized':
                spectral_name = 'the algorithm utilizing the normalised Laplacian'

            k0 = k_list[np.argmin(np.array(phi_list))]
            min_phi = round(phi_list[np.argmin(np.array(phi_list))], 5)
            print(f"With {spectral_name}, the smallest ratio cut of {min_phi} was found "
                    f"with {k0} eigenvector components.")

            phi_log = np.log10(np.array(phi_list))
            if np.average(phi_list) < 3:
                plt.plot(k_list, phi_list, label=spectral_name, color=colors[i])
        plt.xlabel(f"Number of eigenvector components")
        plt.ylabel(f"Graph ratio cut")
        plt.legend()
        plt.gcf()
        plt.savefig(f"eigen_components_{self.fname}")
        plt.show()

    def write_output(self):
        output_name = f"{self.fname}.output"
        fpath = os.getcwd()
        file_dir = os.path.join(os.path.join(fpath, 'results'), f"{output_name}")

        self.df_output.to_csv(file_dir,
                              sep=' ',
                              header=False,
                              index=False)

        with open(file_dir, 'r') as original: data = original.read()
        with open(file_dir, 'w') as modified: modified.write(f"# {self.graphID} {self.numOfVertices} {self.numOfEdges} {self.k}\n" + data)
