# Code to produce coordinates of a lattice of hexagonal nodes and the connections between them
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path
import collections


HEXAGON_SIDE_LENGTH = 1
HEXAGON_SMALL_HEGIHT = HEXAGON_SIDE_LENGTH / 2

import numpy as np

def sort_dictionary(dictionary: dict, func: callable) -> dict:
    return {key: value for key, value in sorted(dictionary.items(), key=func)}


def get_dims(coords: np.array) -> np.array:
    return np.array([[min(coords[:, 0]), max(coords[:, 0])], [min(coords[:, 1]), max(coords[:, 1])]])


def find_closest_node(coords: np.array, point: np.array) -> np.array:
    tree = KDTree(coords)
    distance, index = tree.query(point)
    closest_node = coords[index]
    return closest_node, distance


def connect_nearest_neighbours(coords: np.array, graph: nx.Graph, target_distance: float,
                               num_neighbours: int, colour: str = "black") -> nx.Graph:
    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=num_neighbours+1)
    for node_index, neighbors in enumerate(indices):
        for neighbor, distance in zip(neighbors, distances[node_index]):
            if node_index != neighbor and np.isclose(distance, target_distance, rtol=1e-2):
                graph.add_edge(node_index, neighbor, color=colour)
    return graph


def check_coordination(graph: nx.graph, target_coordination: int) -> bool:
    num_not_target = 0
    valid = True
    for node in graph.nodes:
        if len(list(graph.neighbors(node))) != target_coordination:
            valid = False
            num_not_target += 1
    print(f"{num_not_target} nodes have coordination different from {target_coordination}.")
    return valid


def delete_uncommon_neighbours(graph: nx.graph, edge_nodes: list[int]) -> list[int]:
    # Count the number of neighbours for each node
    neighbor_counts = [len(list(graph.neighbors(node))) for node in edge_nodes]

    # Find the most common number of neighbours
    counter = collections.Counter(neighbor_counts)
    most_common_neighbors = counter.most_common(1)[0][0]

    # Delete nodes that do not have the most common number of neighbours
    for node in edge_nodes:
        if len(list(graph.neighbors(node))) != most_common_neighbors:
            edge_nodes.remove(node)
    return edge_nodes

def identify_edge_nodes(coords: np.ndarray, axis: int, compare_to: float) -> list[int]:
    edge_nodes = {}
    for node, coord in enumerate(coords):
        if np.isclose(coord[axis], compare_to, rtol=1e-2):
            edge_nodes[node] = coord
    edge_nodes = list(sort_dictionary(edge_nodes, lambda item: item[1][(axis + 1) % 2]).keys())
    return edge_nodes

def connect_pbc(graph: nx.Graph, coords: np.array, colour: str = "green") -> nx.Graph:
    bottom_nodes = identify_edge_nodes(coords, 1, coord_dims[1][0])
    top_nodes = identify_edge_nodes(coords, 1, coord_dims[1][1])
    left_nodes = identify_edge_nodes(coords, 0, coord_dims[0][0])
    right_nodes = identify_edge_nodes(coords, 0, coord_dims[0][1])

    for bottom_node, top_node in zip(bottom_nodes, top_nodes):
        if len(list(graph.neighbors(bottom_node))) != len(list(graph.neighbors(top_node))):
            raise ValueError(f"Bottom node {bottom_node} has {len(list(graph.neighbors(bottom_node)))} neighbors, but top node {top_node} has {len(list(graph.neighbors(top_node)))} neighbors.")
    bottom_top_coordination = len(list(graph.neighbors(bottom_nodes[0])))
    print(f"Bottom and top nodes have coordination of {bottom_top_coordination}.")

    for i in range(len(bottom_nodes)):
        graph.add_edge(bottom_nodes[i], top_nodes[i], color=colour)
        if bottom_top_coordination == 1:
            graph.add_edge(bottom_nodes[(i+1)%len(bottom_nodes)], top_nodes[i], color=colour)

    left_nodes = delete_uncommon_neighbours(graph, left_nodes)
    right_nodes = delete_uncommon_neighbours(graph, right_nodes)

    for i in range(len(left_nodes)):
        graph.add_edge(left_nodes[i], right_nodes[i], color=colour)
    return graph

def generate_hexagonal_lattice(dimensions: np.array, bond_length: float) -> np.ndarray:
    horizontal_spacing = bond_length * np.sqrt(3)
    vertical_spacing = bond_length / 2
    box_size = np.array([dimensions[0][1] - dimensions[0][0], dimensions[1][1] - dimensions[1][0]])
    num_nodes_x = int(box_size[0] / horizontal_spacing) + 1
    num_nodes_y = int(box_size[1] / vertical_spacing) + 1
    if num_nodes_x % 2 != 0:
        num_nodes_x += 1
    if num_nodes_y * 2 / 3 % 2 != 0:
        num_nodes_y += 1
    coordinates = []
    for i in range(num_nodes_x):
        for j in range(num_nodes_y):
            # if (j - 1) % 3 != 0: # This line for Olly's style
            if j % 3 != 0:
                x = i * horizontal_spacing + dimensions[0][0] + (j % 2) * horizontal_spacing / 2
                y = j * vertical_spacing + dimensions[1][0]
                coordinates.append((x, y))
    return np.array(coordinates)

cwd = Path.cwd()
box_size = np.array([[0, 5], [0, 5]])
hexagon_side_length = 1

coords = generate_hexagonal_lattice(box_size, hexagon_side_length)
# coords = np.genfromtxt(cwd.joinpath("test_input", "test_A_crds.dat"))
coord_dims = get_dims(coords)

graph = nx.Graph()
for node, coord in enumerate(coords):
    graph.add_node(node, pos=coord)

graph = connect_nearest_neighbours(coords, graph, hexagon_side_length, 3)
graph = connect_pbc(graph, coords)
check_coordination(graph, 3)
point = np.array([box_size[0][0], box_size[1][0]])
closest_node, distance = find_closest_node(coords, point)

print(f"The closest node to {point} is {closest_node} at a distance of {np.round(distance, 5)}.")

# Plot the graph
pos = nx.get_node_attributes(graph, "pos")
edge_colours = [graph[node_1][node_2]["color"] for node_1, node_2 in graph.edges()]
label_pos = {node: (x + 0.1, y + 0.1) for node, (x, y) in pos.items()}
nx.draw(graph, pos, node_size=15, node_color="blue", edge_color=edge_colours)
nx.draw_networkx_labels(graph, label_pos)
plt.show()