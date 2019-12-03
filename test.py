# Comment
from random import sample
from graph import Graph

# files for competition
fnames_competition_test = ['ca-GrQc',
                           'Oregon-1',
                           'soc-Epinions1',
                           'web-NotreDame',
                           ]

# files ok for memory
fnames_small = ['web-NotreDame',
                'Oregon-1',
                'ca-GrQc',
                'soc-Epinions1',
                'roadNet-CA'
                ]

fnames_test = ['ca-GrQc']  # test with smallest graph

for fname in fnames_competition_test:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()
    graph.partition_graph(algorithm="sparse_k_test")

