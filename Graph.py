import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class Graph:
    def __init__(self, fname="", fpath=""):
        """

        :param fname:
        :param fpath:
        """
        self.fname = fname  # filename
        self.fpath = fpath  # filepath or cwd

        self.algorithm = None  # give name of partition algorithm
        self.df_data = None  # input data as dataframe
        self.df_output = None  # datafame that contains list of vertices and their partition index k
        self.list_of_edges = None
        self.list_of_nodes = None
        self.G = None  # networkx graph
        self.k = None # number of partitions
        self.objective = None

        self._read_file()
        self._init_graph()

    def _add_nodes(self):
        """

        :return:
        """

        nodes1 = self.df_data.loc[:, 'node1'].unique()
        nodes2 = self.df_data.loc[:, 'node2'].unique()
        unique_nodes = np.unique(np.concatenate((nodes1, nodes2), 0))
        # a list of nodes:
        self.list_of_nodes = unique_nodes
        self.G.add_nodes_from(unique_nodes)

        print(unique_nodes)
        self.df_output = pd.DataFrame(data=unique_nodes,
                                     columns=['vertexID'])
        print(f"Added nodes to graph")

    def _add_edges(self):
        """

        :return:
        """

        self.list_of_edges = [tuple(x) for x in self.df_data.to_records(index=False)]
        # adding a list of edges:
        self.G.add_edges_from(self.list_of_edges)
        print(f"Added edges to graph")

    def _init_graph(self):
        """

        :return:
        """

        # https://www.python-course.eu/networkx.php
        self.G = nx.Graph()
        self._add_nodes()
        self._add_edges()

        print(self.G.nodes())
        print(self.G.edges())

    def _read_file(self):
        """
        Read first line
        :return:
        """

        if self.fpath == "":
            self.fpath = os.getcwd()
        file_dir = os.path.join(os.path.join(self.fpath, 'graphs_processed'), f"{self.fname}.txt")
        print(f"Reading edge data from {file_dir}")

        # TODO
        #  Get [graphID numOfVertices numOfEdges k] from first comment line
        # self.graphID, self.numOfVertices, self.numOfEdges, self.k = read_first_line()

        header_names = ['node1', 'node2']
        df = pd.read_csv(file_dir, sep=' ', skiprows=1,  header=None, names=header_names)
        print(df.head())
        self.df_data = df

    def _divide_vertices(self, i):

        vertexID = self.df_output['vertexID'].values
        clusterID = self.df_output['clusterID'].values
        mask = clusterID == i
        print(mask)

        V, not_V = vertexID[mask], vertexID[~mask]

        return V, not_V

    def _calculate_edges_between(self, V, not_V):

        count = 0
        for v1, v2 in self.list_of_edges:
            print(f"edge is {v1} - {v2}")
            if (v1 in V) and (v2 in not_V):
                count += 1
            elif (v1 in not_V) and (v2 in V):
                count += 1
        return count

    def calculate_objective(self):
        """
        calculate the objective function
        :return:
        """
        # TODO
        theta = 0
        print(self.k)
        for i in range(self.k):
            V, not_V = self._divide_vertices(i)
            num_edges = self._calculate_edges_between(V, not_V)
            number_of_nodes = len(V)
            theta += num_edges / number_of_nodes  # Float or int division?

        print(f"Objective is {theta}")
        return theta

    def draw_map(self):
        """
        https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html
        # nodes
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[0, 1, 2, 3],
                               node_color='r',
                               node_size=500,
                               alpha=0.8)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[4, 5, 6, 7],
                               node_color='b',
                               node_size=500,
                               alpha=0.8)
        :return:
        """

        nx.draw(self.G, node_size=10)
        plt.savefig(f"graph_{self.fname}.png")  # save as png
        plt.suptitle(f"{self.fname}")
        plt.show()  # display

    def draw_partitioned_map(self):

        # TODO
        NotImplementedError

    def partition_graph(self,
                        algorithm,
                        k=2):
        """
        self.df_output['vertexID'] = list_of_vertices
        self.df_output['clusterID'] = cluster_id_for_vertex

        :param algorithm: name of algorithm as string
        :param k:
        :return:
        """

        self.k = k
        # TODO
        if algorithm == 'spectral':
            NotImplementedError
            self.spectral_partition()
        elif algorithm == 'test':

            self.df_output['clusterID'] = 0
            self.df_output.loc[100:, 'clusterID'] = 1

        else:
            NotImplementedError

    def write_output(self):

        # TODO
        # write: the first line specifies the problem parameters (# graphID numOfVertices
        # numOfEdges k)

        # write other data
        self.df_output.to_csv(path=self.fpath,
                              sep=' ',
                              header=True)

