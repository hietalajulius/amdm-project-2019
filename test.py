# Comment
from graph import Graph

graph = Graph(fname="ca-GrQc.txt",
              fpath="",
              algorithm="",
              k=2)

graph.draw_map()

graph.partition_graph()

graph.draw_partition_map()

graph.write_output()