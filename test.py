# Comment
from random import sample
from graph import Graph

# files for competition
fnames_competition_small = ['ca-GrQc',
                            'soc-Epinions1',
                            'Oregon-1',
                            ]

fnames_competition_large = [
                            'web-NotreDame',
                            'roadNet-CA']

# files ok for memory
fnames_small = ['ca-AstroPh',
                  'ca-CondMat',
                  'ca-GrQc',
                  'ca-HepPh',
                  'ca-HepTh',
                  'Oregon-1']

fnames_test = ['ca-GrQc']  # test with smallest graph

for fname in fnames_competition_small:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()
    graph.partition_graph(algorithm="sparse_k_test")

for fname in fnames_competition_large:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()
    graph.partition_graph(algorithm="sparse")
    # theta = graph.calculate_objective()
    # graph.draw_partitioned_map()
    graph.write_output()



