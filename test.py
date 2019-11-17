# Comment
from random import sample
from graph import Graph

# files with memory error
fnames = ['soc-Epinions1',
          'roadNet-CA',
          'web-NotreDame']

# files ok for memory
fnames = ['ca-AstroPh',
          'ca-CondMat',
          'ca-GrQc',
          'ca-HepPh',
          'ca-HepTh',
          'Oregon-1']

# fname = ['ca-GrQc']  # tet with smallest graph

for fname in fnames:
    print(f"Creating graph from {fname}")
    graph = Graph(fname=fname,
                  fpath="")
    # graph.draw_map()

    graph.partition_graph(algorithm="spectral")

    theta = graph.calculate_objective()

    # graph.draw_partitioned_map()

    graph.write_output()
