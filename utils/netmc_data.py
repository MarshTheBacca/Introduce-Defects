from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

NETWORK_TYPE_MAP = {"base": "A", "ring": "B"}


def shift_nodes(array: np.ndarray, deleted_node: int) -> np.ndarray:
    for i, node in enumerate(array):
        if node > deleted_node:
            array[i] -= 1
    return array


def get_nodes(path: Path, network_type: str) -> list[NetMCNode]:
    if network_type not in ("base", "ring"):
        raise ValueError("Invalid network type {network_type}")
    coords = np.genfromtxt(path)
    nodes = [NetMCNode(coord, id, network_type) for id, coord in enumerate(coords)]
    return nodes


def fill_neighbours(path: Path, selected_nodes: list[NetMCNode], bonded_nodes: list[NetMCNode]) -> None:
    with open(path, "r") as net_file:
        for selected_node, net in enumerate(net_file):
            connected_nodes = net.strip().split()
            for connected_node in connected_nodes:
                node_1 = selected_nodes[selected_node]
                node_2 = bonded_nodes[int(connected_node)]
                if node_2.type == node_1.type and node_2 not in node_1.neighbours:
                    node_1.add_neighbour(node_2)
                elif node_2.type != node_1.type and node_2 not in node_1.ring_neighbours:
                    node_1.add_ring_neighbour(node_2)


def get_geom_code(path: Path) -> str:
    with open(path, "r") as aux_file:
        aux_file.readline()
        aux_file.readline()
        geom_code = aux_file.readline().strip()
    return geom_code


def settify(iterable: list) -> list:
    return_list = []
    for item in iterable:
        if item not in return_list:
            return_list.append(item)
    return return_list


def find_common_elements(lists: list[list]) -> list:
    first_list = lists[0]
    common_elements = [element for element in first_list if all(element in lst for lst in lists[1:])]
    return common_elements


@dataclass
class NetMCData:
    base_network: NetMCNetwork
    ring_network: NetMCNetwork

    @staticmethod
    def from_files(path: Path, prefix: str):
        base_nodes = get_nodes(path.joinpath(f"{prefix}_A_crds.dat"), "base")
        ring_nodes = get_nodes(path.joinpath(f"{prefix}_B_crds.dat"), "ring")
        fill_neighbours(path.joinpath(f"{prefix}_A_net.dat"), base_nodes, base_nodes)
        fill_neighbours(path.joinpath(f"{prefix}_B_net.dat"), ring_nodes, ring_nodes)
        fill_neighbours(path.joinpath(f"{prefix}_A_dual.dat"), base_nodes, ring_nodes)
        fill_neighbours(path.joinpath(f"{prefix}_B_dual.dat"), ring_nodes, base_nodes)
        base_geom_code = get_geom_code(path.joinpath(f"{prefix}_A_aux.dat"))
        ring_geom_code = get_geom_code(path.joinpath(f"{prefix}_B_aux.dat"))
        base_network = NetMCNetwork(base_nodes, "base", base_geom_code)
        ring_network = NetMCNetwork(ring_nodes, "ring", ring_geom_code)
        return NetMCData(base_network, ring_network)

    def export(self, path: Path, prefix: str) -> None:
        self.base_network.export(path, prefix)
        self.ring_network.export(path, prefix)

    def check(self) -> None:
        self.base_network.check()
        self.ring_network.check()

    def __eq__(self, other) -> bool:
        if isinstance(other, NetMCData):
            return (self.base_network == other.base_network and
                    self.ring_network == other.ring_network)
        return False

    def zero_coords(self) -> None:
        translation_vector = -self.dimensions[:, 0]
        self.base_network.translate(translation_vector)
        self.ring_network.translate(translation_vector)

    def centre_coords(self) -> None:
        translation_vector = -np.mean(self.dimensions, axis=1)
        self.base_network.translate(translation_vector)
        self.ring_network.translate(translation_vector)

    def delete_node(self, node: NetMCNode) -> None:
        for neighbour in node.neighbours:
            neighbour.delete_neighbour(node)
        for ring_neighbour in node.ring_neighbours:
            ring_neighbour.delete_ring_neighbour(node)
        self.base_network.delete_node(node)
        self.ring_network.delete_node(node)

    def add_node(self, node: NetMCNode) -> None:
        if node.type == "base":
            self.base_network.add_node(node)
        else:
            self.ring_network.add_node(node)
        for neighbour in node.neighbours:
            neighbour.add_neighbour(node)
        for ring_neighbour in node.ring_neighbours:
            ring_neighbour.add_ring_neighbour(node)

    def merge_nodes(self, nodes_to_merge: list[NetMCNode]) -> None:
        node_types = [node_to_merge.type for node_to_merge in nodes_to_merge]
        if len(set(node_types)) != 1:
            raise ValueError(f"Cannot merge nodes {nodes_to_merge} because they have different types")
        target_network = self.base_network if node_types[0] == "base" else self.ring_network

        # I have to use settify instead of set() because the nodes are not hashable
        all_neighbours = settify([neighbour for node_to_merge in nodes_to_merge for neighbour in node_to_merge.neighbours if neighbour not in nodes_to_merge])
        all_ring_neighbours = settify([neighbour for node_to_merge in nodes_to_merge for neighbour in node_to_merge.ring_neighbours])
        new_coord = np.mean([ring_neighbour.coord for ring_neighbour in all_ring_neighbours], axis=0)
        new_node = NetMCNode(new_coord, target_network.num_nodes, node_types[0], all_neighbours, all_ring_neighbours)
        target_network.add_node(new_node)
        for node in all_neighbours:
            node.add_neighbour(new_node)
        for node in all_ring_neighbours:
            node.add_ring_neighbour(new_node)
        for node_to_merge in nodes_to_merge:
            self.delete_node(node_to_merge)

    def add_bond(self, node_1: NetMCNode, node_2: NetMCNode) -> None:
        if node_1.type == "base" and node_1 not in self.base_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_1.id} is not in the base network")
        if node_2.type == "base" and node_2 not in self.base_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_2.id} is not in the base network")
        if node_1.type == "ring" and node_1 not in self.ring_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_1.id} is not in the ring network")
        if node_2.type == "ring" and node_2 not in self.ring_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_2.id} is not in the ring network")
        node_1.add_neighbour(node_2)
        node_2.add_neighbour(node_1)

    def split_ring(self, node_1: NetMCNode, node_2: NetMCNode) -> None:
        if (node_1 or node_2) not in self.base_network.nodes:
            raise ValueError(f"Cannot split ring between nodes {node_1.id} and {node_2.id} because one or both nodes are not in the base network")
        if node_1.type != "base" or node_2.type != "base":
            raise ValueError(f"Cannot split ring between nodes {node_1.id} and {node_2.id} because one or both nodes are not base nodes")
        common_rings = find_common_elements([node_1.ring_neighbours, node_2.ring_neighbours])
        if len(common_rings) != 1:
            raise ValueError(f"Cannot split ring between nodes {node_1.id} and {node_2.id} because they do not share a ring")
        common_ring = common_rings[0]
        self.add_bond(node_1, node_2)
        ring_walk = common_ring.get_ring_walk()
        index_1 = ring_walk.index(node_1)
        index_2 = ring_walk.index(node_2)
        if index_1 < index_2:
            new_ring_node_1_ring_nodes = ring_walk[index_1:index_2 + 1]
            new_ring_node_2_ring_nodes = ring_walk[index_2:] + ring_walk[:index_1 + 1]
        else:
            new_ring_node_1_ring_nodes = ring_walk[index_2:index_1 + 1]
            new_ring_node_2_ring_nodes = ring_walk[index_1:] + ring_walk[:index_2 + 1]
        new_ring_node_1_nodes = []
        for ring_node in new_ring_node_1_ring_nodes:
            for neighbour in ring_node.ring_neighbours:
                if neighbour not in new_ring_node_1_nodes:
                    new_ring_node_1_nodes.append(neighbour)
        new_ring_node_2_nodes = []
        for ring_node in new_ring_node_2_ring_nodes:
            for neighbour in ring_node.ring_neighbours:
                if neighbour not in new_ring_node_2_nodes:
                    new_ring_node_2_nodes.append(neighbour)
        new_ring_node_1 = NetMCNode(np.mean([ring_node.coord for ring_node in new_ring_node_1_ring_nodes], axis=0),
                                    self.ring_network.num_nodes, "ring", new_ring_node_1_nodes, new_ring_node_1_ring_nodes)
        self.add_node(new_ring_node_1)
        new_ring_node_2 = NetMCNode(np.mean([ring_node.coord for ring_node in new_ring_node_2_ring_nodes], axis=0),
                                    self.ring_network.num_nodes, "ring", new_ring_node_2_nodes, new_ring_node_2_ring_nodes)
        self.add_node(new_ring_node_2)
        self.add_bond(new_ring_node_1, new_ring_node_2)
        self.delete_node(common_ring)

    @property
    def graph(self) -> nx.Graph:
        # Assumes base-ring connections are the same as ring-base connections
        graph = nx.Graph()
        for node in self.base_network.nodes:
            graph.add_node(node.id, pos=(node.x, node.y), source='base_network')
        for bond in self.base_network.bonds:
            if bond.length < 2 * self.base_network.avg_bond_length:
                graph.add_edge(bond.node_1.id, bond.node_2.id, source='base_network')
        for bond in self.base_network.ring_bonds:
            if bond.length < 2 * self.base_network.avg_ring_bond_length:
                graph.add_edge(bond.node_1.id, self.base_network.num_nodes + bond.node_2.id, source='base_ring_bonds')
        for node in self.ring_network.nodes:
            graph.add_node(self.base_network.num_nodes + node.id, pos=(node.x, node.y), source='ring_network')
        for bond in self.ring_network.bonds:
            if bond.length < 2 * self.ring_network.avg_bond_length:
                graph.add_edge(self.base_network.num_nodes + bond.node_1.id, self.base_network.num_nodes + bond.node_2.id, source='ring_network')
        return graph

    def draw_graph(self) -> None:
        node_colors = ['red' if data['source'] == 'base_network' else 'blue' for _, data in self.graph.nodes(data=True)]
        edge_colors = []
        for _, _, data in self.graph.edges(data=True):
            if data['source'] == 'base_network':
                edge_colors.append('black')
            elif data['source'] == 'ring_network':
                edge_colors.append('green')
            else:
                edge_colors.append('blue')
        node_sizes = [30 if data['source'] == 'base_network' else 10 for _, data in self.graph.nodes(data=True)]
        edge_widths = [2 if data['source'] == 'base_network' else 1 for _, _, data in self.graph.edges(data=True)]
        pos = nx.get_node_attributes(self.graph, "pos")
        nx.draw(self.graph, pos, node_color=node_colors, edge_color=edge_colors, node_size=node_sizes, width=edge_widths)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([[self.xlo, self.xhi], [self.ylo, self.yhi]])

    @property
    def xlo(self) -> float:
        return min(self.base_network.xlo, self.ring_network.xlo)

    @property
    def xhi(self) -> float:
        return max(self.base_network.xhi, self.ring_network.xhi)

    @property
    def ylo(self) -> float:
        return min(self.base_network.ylo, self.ring_network.ylo)

    @property
    def yhi(self) -> float:
        return max(self.base_network.yhi, self.ring_network.yhi)


@dataclass
class NetMCNetwork:
    nodes: list[NetMCNode] = field(default_factory=lambda: [])
    type: str = "base"
    geom_code: str = "2DE"

    def __post_init__(self):
        self.avg_bond_length = np.mean([bond.length for bond in self.bonds])
        self.avg_ring_bond_length = np.mean([bond.length for bond in self.ring_bonds])

    def delete_node(self, node: NetMCNode) -> None:
        if node.type == self.type:
            self.nodes.remove(node)
            for i, node in enumerate(self.nodes):
                node.id = i

    # in a separate method because this is a time-consuming operation
    def get_avg_bond_length(self) -> float:
        return np.mean([bond.length for bond in self.bonds])

    def get_avg_ring_bond_length(self) -> float:
        return np.mean([bond.length for bond in self.ring_bonds])

    def check(self) -> None:
        if self.type not in ("base", "ring"):
            raise ValueError(f"Network has invalid type {self.type}")
        for node in self.nodes:
            node.check()
        for bond in self.bonds:
            bond.check()
        for ring_bond in self.ring_bonds:
            ring_bond.check()

    def translate(self, vector: np.ndarray) -> None:
        for node in self.nodes:
            node.translate(vector)

    def get_nearest_node(self, point: np.array) -> tuple(NetMCNode, float):
        distance, index = self.kdtree.query(point)
        return self.nodes[index], distance

    def add_node(self, node) -> None:
        self.nodes.append(node)

    @property
    def kdtree(self):
        return KDTree(np.array([[node.x, node.y] for node in self.nodes]))

    @property
    def bonds(self) -> list[NetMCBond]:
        bonds = []
        for node in self.nodes:
            for neighbour in node.neighbours:
                bonds.append(NetMCBond(node, neighbour))
        return bonds

    @property
    def ring_bonds(self) -> list[NetMCBond]:
        bonds = []
        for node in self.nodes:
            for neighbour in node.ring_neighbours:
                bonds.append(NetMCBond(node, neighbour))
        return bonds

    @property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node.id, pos=(node.x, node.y))
        for bond in self.bonds:
            if bond.length < 2 * self.avg_bond_length:
                graph.add_edge(bond.node_1.id, bond.node_2.id)
        return graph

    def export(self, path: Path, prefix: str) -> None:
        self.export_aux(path.joinpath(f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_aux.dat"))
        self.export_coords(path.joinpath(f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_crds.dat"))
        self.export_base_bonds(path.joinpath(f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_net.dat"))
        self.export_ring_bonds(path.joinpath(f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_dual.dat"))

    def export_aux(self, path: Path) -> None:
        with open(path, "w") as aux_file:
            aux_file.write(f"{self.num_nodes}\n")
            aux_file.write(f"{self.max_connections:<10}{self.max_ring_connections:<10}\n")
            aux_file.write(f"{self.geom_code}\n")
            aux_file.write(f"{self.xhi:<20.6f}{self.yhi:<20.6f}\n")
            aux_file.write(f"{self.xlo:<20.6f}{self.ylo:<20.6f}\n")

    def export_coords(self, path: Path) -> None:
        coords = np.array([[node.x, node.y] for node in self.nodes])
        np.savetxt(path, coords, fmt="%-19.6f")

    def export_bonds(self, path: Path, neighbour_attribute: str) -> None:
        with open(path, "w") as file:
            for node in self.nodes:
                for neighbour in getattr(node, neighbour_attribute):
                    file.write(f"{neighbour.id:<10}")
                file.write("\n")

    def export_base_bonds(self, path: Path) -> None:
        self.export_bonds(path, "neighbours")

    def export_ring_bonds(self, path: Path) -> None:
        self.export_bonds(path, "ring_neighbours")

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_bonds(self):
        return len(self.bonds)

    @property
    def num_ring_bonds(self):
        return len(self.ring_bonds)

    @property
    def dimensions(self):
        return np.array([[self.xlo, self.xhi], [self.ylo, self.yhi]])

    @property
    def max_connections(self):
        return max([node.num_neighbours for node in self.nodes])

    @property
    def max_ring_connections(self):
        return max([node.num_ring_neighbours for node in self.nodes])

    @property
    def xlo(self):
        return min([node.x for node in self.nodes])

    @property
    def xhi(self):
        return max([node.x for node in self.nodes])

    @property
    def ylo(self):
        return min([node.y for node in self.nodes])

    @property
    def yhi(self):
        return max([node.y for node in self.nodes])

    def __eq__(self, other):
        if isinstance(other, NetMCNetwork):
            return (self.nodes == other.nodes and
                    self.type == other.type and
                    self.geom_code == other.geom_code)
        return False


@dataclass
class NetMCNode:
    coord: np.array
    id: int
    type: str
    neighbours: list[NetMCNode] = field(default_factory=lambda: [])
    ring_neighbours: list[NetMCNode] = field(default_factory=lambda: [])

    def check(self) -> None:
        if self.type not in ("base", "ring"):
            raise ValueError(f"Node {self.id} has invalid type {self.type}")
        for neighbour in self.neighbours:
            if neighbour == self:
                raise ValueError(f"Node {self.id} has itself as neighbour")
            if self not in neighbour.neighbours:
                raise ValueError(f"Node {self.id} has neighbour {neighbour.id}, but neighbour does not have node as neighbour")
            if self.type != neighbour.type:
                raise ValueError(f"Node {self.id} has neighbour {neighbour.id}, but neighbour has different type")
        for ring_neighbour in self.ring_neighbours:
            if self not in ring_neighbour.ring_neighbours:
                raise ValueError(f"Node {self.id} has ring neighbour {ring_neighbour.id}, but ring neighbour does not have node as ring neighbour")
            if self.type == ring_neighbour.type:
                raise ValueError(f"Node {self.id} has ring neighbour {ring_neighbour.id}, but ring neighbour has same type")

    def get_ring_walk(self) -> list[NetMCNode]:
        """
        Returns a list of nodes such that the order is how they are connected in the ring.
        """
        walk = [self.ring_neighbours[0]]
        while len(walk) < len(self.ring_neighbours):
            current_node = walk[-1]
            for neighbour in current_node.neighbours:
                if neighbour in self.ring_neighbours and neighbour not in walk:
                    walk.append(neighbour)
                    break
        return walk

    def add_neighbour(self, neighbour: 'NetMCNode') -> None:
        self.neighbours.append(neighbour)

    def delete_neighbour(self, neighbour: 'NetMCNode') -> None:
        self.neighbours.remove(neighbour)

    def add_ring_neighbour(self, neighbour: 'NetMCNode') -> None:
        self.ring_neighbours.append(neighbour)

    def delete_ring_neighbour(self, neighbour: 'NetMCNode') -> None:
        self.ring_neighbours.remove(neighbour)

    def translate(self, vector: np.array) -> None:
        self.coord += vector

    @property
    def num_neighbours(self) -> int:
        return len(self.neighbours)

    @property
    def num_ring_neighbours(self) -> int:
        return len(self.ring_neighbours)

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    def __eq__(self, other) -> bool:
        if isinstance(other, NetMCNode):
            return (self.id == other.id and
                    self.type == other.type and
                    np.array_equal(self.coord, other.coord) and
                    self.neighbours == other.neighbours and
                    self.ring_neighbours == other.ring_neighbours)
        return False

    def __repr__(self) -> str:
        return f"Node {self.id} ({self.type}) at {self.coord}. neighbours: {[neighbour.id for neighbour in self.neighbours]}." \
            f"ring neighbours: {[ring_neighbour.id for ring_neighbour in self.ring_neighbours]}"


@dataclass
class NetMCBond:
    node_1: NetMCNode
    node_2: NetMCNode

    @property
    def length(self):
        return np.linalg.norm(self.node_1.coord - self.node_2.coord)

    @property
    def type(self):
        return f"{self.node_1.type}-{self.node_2.type}"

    def check(self) -> None:
        if self.node_1 == self.node_2:
            raise ValueError(f"Bond {self} has same node as both nodes")

    def __eq__(self, other):
        if isinstance(other, NetMCBond):
            return ((self.node_1 == other.node_1 and self.node_2 == other.node_2) or
                    (self.node_1 == other.node_2 and self.node_2 == other.node_1))
        return False
