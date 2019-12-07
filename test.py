# Comment
from random import sample
from graph import Graph

# can run with larger grid
fnames_competition_small = ['ca-GrQc',
                           'Oregon-1',
                           'soc-Epinions1',
                           'web-NotreDame',
                           ]

# run with smaller grid
fnames_roadnet = ['roadNet-CA']  # test with smallest graph

for fname in fnames_roadnet:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()
    graph.partition_graph(algorithm="sparse_k_test")

