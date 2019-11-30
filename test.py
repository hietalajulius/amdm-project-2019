# Comment
from random import sample
from graph import Graph

# files for competition
fnames_competition = ['ca-GrQc',
                      'soc-Epinions1',
                      'roadNet-CA',
                      'web-NotreDame',
                      'Oregon-1']

# files ok for memory
fnames_small = ['ca-AstroPh',
                  'ca-CondMat',
                  'ca-GrQc',
                  'ca-HepPh',
                  'ca-HepTh',
                  'Oregon-1']

fnames_test = ['roadNet-CA']  # test with smallest graph

for fname in fnames_test:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()

    graph.partition_graph(algorithm="sparse")

    # theta = graph.calculate_objective()

    # graph.draw_partitioned_map()

    graph.write_output()
