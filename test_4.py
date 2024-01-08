from utils import NetMCData, NetworkType
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

cwd = Path.cwd()

data = NetMCData.import_data(cwd.joinpath("test_input"), prefix="test")
graph = data.get_graph(NetworkType.A)


def find_closest_node(coords: np.array, point: np.array) -> tuple[np.array, float]:
    tree = KDTree(coords)
    distance, index = tree.query(point)
    closest_node = coords[index]
    return closest_node, distance


point = np.array([0, 0])
closest_node, distance = find_closest_node(data.crds_a, point)

print(f"The closest node to {point} is {
      closest_node} at a distance of {distance}.")

pos = nx.get_node_attributes(graph, "pos")
nx.draw(graph, pos, node_size=10, node_color="lightblue", with_labels=False)
plt.show()
