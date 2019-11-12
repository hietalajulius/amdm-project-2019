import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class Graph:
    def __init__(self, fname="", fpath="", algorithm=""):
        self.data = []
        self.fname = fname
        self.fpath = fpath
        self.algorithm = algorithm

        self.df_data = None
        self.G = None

        self._read_file()
        self._init_graph()

    def _read_file(self):

        if self.fpath == "":
            self.fpath = os.getcwd()
        file_dir = os.path.join(os.path.join(self.fpath, 'graphs_processed'), self.fname)
        print(f"Reading graph from {file_dir}")

        header_names = ['node1', 'node2']
        df = pd.read_csv(file_dir, sep=' ', skiprows=1,  header=None, names=header_names)
        print(df.head())
        self.df_data = df

    def _add_nodes(self):

        nodes1 = self.df_data.loc['node1'].unique()
        nodes2 = self.df_data.loc['node2'].unique()
        unique_nodes = np.unique(np.concatenate((nodes1, nodes2), 0))
        # a list of nodes:
        self.G.add_nodes_from(unique_nodes)

    def _add_edges(self):
        self.list_of_edges = [(0, 4016), (1, 2977), (1, 258), (1, 421), (1, 1201)]

        # adding a list of edges:
        self.G.add_edges_from(self.list_of_edges)

    def _init_graph(self):

        # https://www.python-course.eu/networkx.php
        self.G = nx.Graph()
        self._add_nodes()
        self._add_edges()

        print(self.G.nodes())
        print(self.G.edges())

    def draw_map(self):

        nx.draw(self.G)
        plt.savefig(f"graph_{self.fname}.png")  # save as png
        plt.show()  # display
