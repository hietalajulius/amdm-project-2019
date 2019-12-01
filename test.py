# Comment
from random import sample
from graph import Graph

# files for competition
fnames_competition_test = ['roadNet-CA',
                           'soc-Epinions1'
                           ]

# files ok for memory
fnames_small = ['web-NotreDame',
                'Oregon-1',
                'ca-GrQc',
                'roadNet-CA',
                'soc-Epinions1'
                ]

fnames_test = ['ca-GrQc']  # test with smallest graph

for fname in fnames_competition_test:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()
    graph.partition_graph(algorithm="sparse_k_test")


"""
for fname in fnames_competition_large:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()
    graph.partition_graph(algorithm="sparse")
    # theta = graph.calculate_objective()
    # graph.draw_partitioned_map()
    graph.write_output()
"""


