# Comment
from random import sample
from graph import Graph

fnames = ['ca-AstroPh',
          'ca-CondMat',
          'ca-GrQc',
          'ca-HepPh',
          'ca-HepTh',
          'Oregon-1',
          'roadNet-CA',
          'soc-Epinions1',
          'web-NotreDame']

fname = sample(fnames, 1)[0]  # draw random graph
fname = fnames[2]

print(f"Creating graph from {fname}")
graph = Graph(fname=fname,
              fpath="")

# graph.draw_map()

graph.partition_graph(algorithm="spectral")

theta = graph.calculate_objective()

graph.draw_partitioned_map()

# graph.write_output()
