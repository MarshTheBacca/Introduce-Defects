from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from .bss_bond import BSSBond
from .bss_node import BSSNode
from .other_utils import calculate_angle, pbc_vector

NETWORK_TYPE_MAP = {"base": "base_network", "ring": "dual_network"}


@dataclass
class BSSNetwork:
    nodes: list[BSSNode] = field(default_factory=lambda: [])
    type: str = "base"
    dimensions: np.ndarray = field(default_factory=lambda: np.array([[0, 0], [1, 1]]))

    def delete_node(self, node: BSSNode) -> None:
        if node.type == self.type:
            self.nodes.remove(node)
            for i, node in enumerate(self.nodes):
                node.id = i

    def add_node(self, node) -> None:
        if node.type == self.type:
            self.nodes.append(node)
            node.id = self.num_nodes - 1

    def check(self) -> bool:
        valid = True
        if self.type not in ("base", "ring"):
            print(f"Network has invalid type {self.type}")
            valid = False
        for node in self.nodes:
            if not node.check(self.dimensions):
                valid = False
        for bond in self.bonds:
            if not bond.check():
                valid = False
        for ring_bond in self.ring_bonds:
            if not ring_bond.check():
                valid = False
        return valid

    def bond_close_nodes(self, target_distance: float, coordination: int) -> None:
        coords = np.array([[node.x, node.y] for node in self.nodes])
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=coordination + 1)
        for node_index, neighbours in enumerate(indices):
            for neighbour, distance in zip(neighbours, distances[node_index]):
                if node_index != neighbour and np.isclose(distance, target_distance, rtol=1e-2):
                    self.nodes[node_index].add_neighbour(self.nodes[neighbour], self.dimensions)
                    self.nodes[neighbour].add_neighbour(self.nodes[node_index], self.dimensions)

    def get_edge_lengths(self) -> np.ndarray:
        """
        Gets the lengths of all the edges in the network.

        Returns:
            np.ndarray: An array of the lengths of the edges in the network.
        """
        lengths = []
        for node in self.nodes:
            for neighbour in node.neighbours:
                lengths.append(np.linalg.norm(pbc_vector(node.coord, neighbour.coord, self.dimensions)))
        return np.array(lengths)

    def translate(self, vector: np.ndarray) -> None:
        """
        Translates all nodes by the given vector, and wraps them around the periodic boundary conditions.

        Args:
            vector: The vector by which to translate the nodes.
        """
        for node in self.nodes:
            node.translate(vector)
        self.wrap_nodes()

    def wrap_nodes(self) -> None:
        width = self.dimensions[1][0] - self.dimensions[0][0]
        height = self.dimensions[1][1] - self.dimensions[0][1]
        for node in self.nodes:
            node.coord[0] = self.dimensions[0][0] + (node.coord[0] - self.dimensions[0][0]) % width
            node.coord[1] = self.dimensions[0][1] + (node.coord[1] - self.dimensions[0][1]) % height

    def scale(self, scale_factor: float) -> None:
        for node in self.nodes:
            node.scale(scale_factor)
        self.dimensions *= scale_factor

    def get_nearest_node(self, point: np.ndarray) -> tuple[BSSNode, float]:
        """
        Gets the nearest node to a given point and the distance to that node.
        """
        distance, index = self.kdtree.query(point)
        return self.nodes[index], distance

    def get_average_bond_length(self) -> float:
        return np.mean([bond.pbc_length(self.dimensions) for bond in self.bonds])

    @ property
    def kdtree(self):
        return KDTree(np.array([[node.x, node.y] for node in self.nodes]))

    @ property
    def bonds(self) -> list[BSSBond]:
        """
        Computationally expensive, do not use regularly.
        """
        bonds = []
        for node in self.nodes:
            for neighbour in node.neighbours:
                bond = BSSBond(node, neighbour)
                if bond not in bonds:
                    bonds.append(bond)
        return bonds

    @ property
    def ring_bonds(self) -> list[BSSBond]:
        bonds = []
        for node in self.nodes:
            for neighbour in node.ring_neighbours:
                bonds.append(BSSBond(node, neighbour))
        return bonds

    @ property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node.id, pos=(node.x, node.y))
        for bond in self.bonds:
            if bond.length < 2 * self.avg_bond_length:
                graph.add_edge(bond.node_1.id, bond.node_2.id)
        return graph

    def get_angles(self) -> Generator[tuple[BSSNode, BSSNode, BSSNode], None, None]:
        for node in self.nodes:
            for angle in node.get_angles():
                yield angle

    def evaluate_strain(self, ideal_bond_length: float, ideal_bond_angle: float) -> float:
        """
        Evaluates the strain of a network using a very basic model. E = sum((l - l_0)^2) + sum((theta - theta_0)^2)
        Bond angles are in degrees.
        """
        bond_length_strain = np.sum([(bond.length - ideal_bond_length) ** 2 for bond in self.bonds])
        angle_strain = 0
        for angle in self.get_angles():
            angle_deviation = calculate_angle(angle[0].coord, angle[1].coord, angle[2].coord) - ideal_bond_angle
            angle_strain += angle_deviation ** 2
        return bond_length_strain + angle_strain

    def export(self, dimensions: np.ndarray, path: Path) -> None:
        self.export_info(path.joinpath(f"{NETWORK_TYPE_MAP[self.type]}_info.txt"), dimensions)
        self.export_coords(path.joinpath(f"{NETWORK_TYPE_MAP[self.type]}_coords.txt"))
        self.export_base_bonds(path.joinpath(f"{NETWORK_TYPE_MAP[self.type]}_connections.txt"))
        self.export_ring_bonds(path.joinpath(f"{NETWORK_TYPE_MAP[self.type]}_dual_connections.txt"))

    def export_info(self, path: Path, dimensions: np.ndarray) -> None:
        with open(path, "w") as info_file:
            info_file.write(f"Number of nodes: {self.num_nodes}\n")
            info_file.write(f"xhi: {dimensions[1][0]}\n")
            info_file.write(f"yhi: {dimensions[1][1]}\n")

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

    def get_assortativity_distribution(self) -> np.ndarray:
        """
        Gets a list of the number of neighbours of each node and the number of neighbours of each neighbour.
        """
        assortativity_distribution = []
        for node in self.nodes:
            for neighbour in node.neighbours:
                assortativity_distribution.append([len(node.neighbours), len(neighbour.neighbours)])
        return np.array(assortativity_distribution)

    def get_pearsons(self) -> float:
        """
        Gets the pearsons correlation coefficient of the network
        """
        assortativity_distribution = self.get_assortativity_distribution()
        pearsons_coeff = np.corrcoef(assortativity_distribution.T)[0, 1]
        if pearsons_coeff < -1 or pearsons_coeff > 1:
            raise ValueError(f"Out of range pearsons correlation coefficient: {pearsons_coeff}")
        return pearsons_coeff

    @ property
    def num_nodes(self):
        return len(self.nodes)

    @ property
    def num_bonds(self):
        return len(self.bonds)

    @ property
    def num_ring_bonds(self):
        return len(self.ring_bonds)

    @ property
    def max_connections(self):
        return max([node.num_neighbours for node in self.nodes])

    @ property
    def max_ring_connections(self):
        return max([node.num_ring_neighbours for node in self.nodes])

    @ property
    def node_xlo(self):
        return min([node.x for node in self.nodes])

    @ property
    def node_xhi(self):
        return max([node.x for node in self.nodes])

    @ property
    def node_ylo(self):
        return min([node.y for node in self.nodes])

    @ property
    def node_yhi(self):
        return max([node.y for node in self.nodes])

    def __eq__(self, other):
        if isinstance(other, BSSNetwork):
            return (self.nodes == other.nodes and
                    self.type == other.type and
                    self.dimensions == other.dimensions)
        return False

    def __repr__(self) -> str:
        string = f"BSSNetwork of type {self.type} with {self.num_nodes} nodes: \n"
        for node in self.nodes:
            string += f"{node}\n"
        return string
