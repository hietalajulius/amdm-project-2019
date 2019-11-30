import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sparse import sparse_partitioning
from spectral import normalized_spectral_clustering

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
        # a list of nodes:
        self.list_of_nodes = unique_nodes
        self.G.add_nodes_from(unique_nodes)

        # print(unique_nodes)
        self.df_output = pd.DataFrame(data=unique_nodes, columns=['vertexID'])
        print(f"Added nodes to graph")

    def _add_edges(self):
        """

        :return:
        """
        # print(self.df_data)
        self.list_of_edges = self.df_data.loc[:, ['node1', 'node2']].values

        graph_edges = [tuple(x) for x in self.df_data.to_records(index=False)]
        self.graph_edges = graph_edges
        # adding a list of edges:
        self.G.add_edges_from(graph_edges)

        # self.list_of_edges = graph_edges
        print(f"Added edges to graph")

    def _init_graph(self):
        """

        :return:
        """

        # https://www.python-course.eu/networkx.php
        self.G = nx.Graph()
        self._add_nodes()
        self._add_edges()

        # print(self.G.nodes())
        # print(self.G.edges())

    def _calculate_edges_between(self, v, not_v):

        count = 0
        for v1, v2 in self.list_of_edges:
            # print(f"edge is {v1} - {v2}")
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

        if self.fpath == "":
            self.fpath = os.getcwd()
        file_dir = os.path.join(os.path.join(self.fpath, 'graphs_processed'), f"{self.fname}.txt")
        print(f"Reading edge data from {file_dir}")


        with open(file_dir, "r", newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)

        #  Get [graphID numOfVertices numOfEdges k] from first comment line
        _, self.graphID, self.numOfVertices, self.numOfEdges, self.k = row1[0].split(" ")
        self.graphID, self.numOfVertices, self.numOfEdges, self.k = self.graphID, \
                                                                    int(self.numOfVertices), \
                                                                    int(self.numOfEdges), \
                                                                    int(self.k)

        header_names = ['node1', 'node2']
        df = pd.read_csv(file_dir, sep=' ', skiprows=1,  header=None, names=header_names)
        print(df.head())
        self.df_data = df

    def calculate_objective(self):
        """
        calculate the objective function, tarkista onko oikein?
        Pitäiskö tehä tämä partitionin sisään ei class funktioks?
        :return:
        """
        # TODO
        print(f"Calculating objective function")
        theta = 0
        for i in range(self.k):
            V, not_V = self._divide_vertices(i)
            num_edges = self._calculate_edges_between(V, not_V)
            number_of_nodes = len(V)

            obj = round(num_edges / number_of_nodes, 1)
            print(f"Objective is {obj}. Number of partition edges is {num_edges} "
                  f"and number of nodes in partition {i} is {number_of_nodes}")
            theta += num_edges / number_of_nodes  # Float or int division?

        print(f"Total objective value is {theta}")
        return theta

    def draw_map(self):
        """
        Draws graph with nodes and edges
        https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html
        :return:
        """

        nx.draw(self.G, node_size=10)
        plt.savefig(f"graph_{self.fname}.png")  # save as png
        plt.suptitle(f"{self.fname}")
        plt.show()  # display

    def draw_partitioned_map(self):
        """
        Draws graph with partitioned nodes and edges
        https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html
        :return:
        """
        print(f"Drawing map with partitioned vertices")

        pos = nx.spring_layout(self.G)  # k=0.2
        # print(f"self.k is {self.k}")
        cmap = plt.cm.rainbow(np.linspace(0, 1, self.k))
        # print(f"cmap is {cmap}")
        for i in range(self.k):
            v, _ = self._divide_vertices(i)
            nodelist = v.tolist()
            # for node in v:
            #    nodelist.append(node)
            print(f"nodelist size for cluster {i} is {len(nodelist)}")
            node_color = cmap[i]
            # print(f"node_color {node_color}")
            color_2d = np.zeros((len(nodelist), 4)) + node_color.reshape(-1, 1).T
            # print(f"color_2d shape is {color_2d.shape}")

            nx.draw_networkx_nodes(self.G, pos,
                                   nodelist=nodelist,
                                   node_color=color_2d,
                                   node_size=10,
                                   alpha=0.8)

        nx.draw_networkx_edges(self.G, pos,
                               edgelist=self.graph_edges,
                               width=1, alpha=0.5, edge_color='grey')

        plt.savefig(f"partitioned_graph_{self.fname}.png")  # save as png
        plt.suptitle(f"{self.fname}")
        plt.show()  # display

    def partition_graph(self,
                        algorithm):
        """
        Write partition values to self.df_output
        self.df_output['vertexID'] = list_of_vertices
        self.df_output['clusterID'] = cluster_id_for_vertex

        :param algorithm: name of algorithm as string
        :return:
        """

        print(f"Partitioning graph to {self.k} clusters with {algorithm}")
        # TODO
        if algorithm == 'spectral':
            df = normalized_spectral_clustering(self.list_of_nodes,
                                                self.list_of_edges,
                                                self.k)

        elif algorithm == 'sparse':
            df = sparse_partitioning(self.G, self.k, self.list_of_nodes)
        else:
            print(f"Check algorithm spelling, not found.")


        self.df_output = df

    def write_output(self):

        output_name = f"{self.fname}.output"
        if self.fpath == "":
            self.fpath = os.getcwd()
        file_dir = os.path.join(os.path.join(self.fpath, 'results'), f"{output_name}")
        print(f"Writing results to {file_dir}")

        # write node and cluster data
        self.df_output.to_csv(file_dir,
                              sep=' ',
                              header=False,
                              index=False)

        # TODO
        # write: the first line specifies the problem parameters (# graphID numOfVertices numOfEdges k)
        with open(file_dir, 'r') as original: data = original.read()
        with open(file_dir, 'w') as modified: modified.write(f"# {self.graphID} {self.numOfVertices} {self.numOfEdges} {self.k}\n" + data)
