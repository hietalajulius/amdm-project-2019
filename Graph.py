import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd



# https://www.python-course.eu/networkx.php

def read_file(fname):
    fdir = os.getcwd()
    fpath = os.path.join(fdir, fname)

    df = pd.read.csv(fpath, sep=' ', skip_blank_lines=True)

    return df

def init_graph():
    G = nx.Graph()

    # a list of nodes:
    G.add_nodes_from(["b", "c"])

    # adding a list of edges:
    G.add_edges_from([("a", "c"), ("c", "d"), ("a", 1), (1, "d"), ("a", 2)])

    print(G.nodes())
    print(G.edges())

def draw_map(G):

    nx.draw(G)
    plt.savefig("simple_path.png")  # save as png
    plt.show()  # display
