from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

NETWORK_TYPE_MAP = {"base": "A", "ring": "B"}


class InvalidNetworkException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidUndercoordinatedNodesException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CouldNotBondUndercoordinatedNodesException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def calculate_angle(coord_1: np.ndarray, coord_2: np.ndarray, coord_3: np.ndarray) -> float:
    """
    Returns the angle between three points in degrees.
    """
    vector1 = np.array(coord_1) - np.array(coord_2)
    vector2 = np.array(coord_3) - np.array(coord_2)
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle = np.degrees(angle)
    # Return the acute angle
    return min(angle, 180 - angle)


def get_nodes(path: Path, network_type: str) -> list[NetMCNode]:
    """
    Read a file containing node coordinates and return a list of NetMCNode objects.
    """
    if network_type not in ("base", "ring"):
        raise ValueError("Invalid network type {network_type}")
    coords = np.genfromtxt(path)
    nodes = [NetMCNode(coord=coord, neighbours=[], ring_neighbours=[], id=id, type=network_type) for id, coord in enumerate(coords)]
    return nodes


def fill_neighbours(path: Path, selected_nodes: list[NetMCNode], bonded_nodes: list[NetMCNode]) -> None:
    """
    Read a file containing node connections and add the connections to the selected nodes.
    """
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


def get_aux_data(path: Path) -> tuple[np.ndarray, str]:
    with open(path, "r") as aux_file:
        aux_file.readline()
        aux_file.readline()
        geom_code = aux_file.readline().strip()
        xhi_yhi = aux_file.readline().strip().split()
        xlo_ylo = aux_file.readline().strip().split()
        dimensions = np.array([[float(xlo_ylo[0]), float(xlo_ylo[1])],
                               [float(xhi_yhi[0]), float(xhi_yhi[1])]])
    return dimensions, geom_code


def settify(iterable: list) -> list:
    """
    Used to remove duplicates from a list while preserving order. This is for lists with mutable types, ie, are not hashable.
    """
    return_list = []
    for item in iterable:
        if item not in return_list:
            return_list.append(item)
    return return_list


def find_common_elements(lists: list[list]) -> list:
    """
    Return a list of elements that are common to all lists.
    """
    first_list = lists[0]
    common_elements = [element for element in first_list if all(
        element in lst for lst in lists[1:])]
    return common_elements


def rounded_sqrt(number: float) -> int:
    """
    Return the square root of a number rounded to the nearest integer.
    """
    if number < 0:
        raise ValueError("Cannot take the square root of a negative number.")
    return int(np.sqrt(number) + 0.5)


def pbc_vector(vector1: np.ndarray, vector2: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    Calculate the vector difference between two vectors, taking into account periodic boundary conditions.
    Remember dimensions = [[xlo, ylo], [xhi, yhi]]
    """
    if len(vector1) != len(vector2) or len(vector1) != len(dimensions):
        raise ValueError("Vectors must have the same number of dimensions.")
    difference_vector = np.subtract(vector2, vector1)
    dimension_ranges = dimensions[1] - dimensions[0]
    half_dimension_ranges = dimension_ranges / 2
    difference_vector = (difference_vector + half_dimension_ranges) % dimension_ranges - half_dimension_ranges
    return difference_vector


def is_pbc_bond(node_1: NetMCNode, node_2: NetMCNode, dimensions: np.array) -> bool:
    """
    Identifies bonds that cross the periodic boundary. So if the length of the bond is 
    more than 10% longer than the distance between the two nodes with periodic boundary conditions,
    then it is considered a periodic bond.
    """
    if np.linalg.norm(node_1.coord - node_2.coord) > np.linalg.norm(pbc_vector(node_1.coord, node_2.coord, dimensions)) * 1.1:
        return True
    return False


@dataclass
class NetMCNetwork:
    nodes: list[NetMCNode] = field(default_factory=lambda: [])
    type: str = "base"
    geom_code: str = "2DE"

    def __post_init__(self):
        self.avg_bond_length = np.mean([bond.length for bond in self.bonds])
        self._ = self.get_avg_ring_bond_length()
        self.avg_ring_bond_length = self._.copy()

    def __repr__(self) -> str:
        string = f"NetMCNetwork of type {self.type} with {self.num_nodes} nodes: \n"
        for node in self.nodes:
            string += f"{node}\n"
        return string

    def delete_node(self, node: NetMCNode) -> None:
        if node.type == self.type:
            self.nodes.remove(node)
            for i, node in enumerate(self.nodes):
                node.id = i

    def add_node(self, node) -> None:
        if node.type == self.type:
            self.nodes.append(node)
            node.id = self.num_nodes - 1

    # in a separate method because this is a time-consuming operation

    def get_avg_bond_length(self) -> float:
        bond_lengths = [bond.length for bond in self.bonds]
        # filter out any periodic bonds
        threshold = 4 * np.mean(bond_lengths)
        return np.mean([length for length in bond_lengths if length < threshold])

    def get_avg_ring_bond_length(self) -> float:
        return np.mean([bond.length for bond in self.ring_bonds])

    def check(self) -> bool:
        valid = True
        if self.type not in ("base", "ring"):
            print(f"Network has invalid type {self.type}")
            valid = False
        for node in self.nodes:
            valid = node.check()
        for bond in self.bonds:
            valid = bond.check()
        for ring_bond in self.ring_bonds:
            valid = ring_bond.check()
        return valid

    def bond_close_nodes(self, target_distance: float, coordination: int) -> None:
        coords = np.array([[node.x, node.y] for node in self.nodes])
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=coordination + 1)
        for node_index, neighbours in enumerate(indices):
            for neighbour, distance in zip(neighbours, distances[node_index]):
                if node_index != neighbour and np.isclose(distance, target_distance, rtol=1e-2):
                    self.nodes[node_index].add_neighbour(self.nodes[neighbour])
                    self.nodes[neighbour].add_neighbour(self.nodes[node_index])

    def translate(self, vector: np.ndarray) -> None:
        for node in self.nodes:
            node.translate(vector)

    def get_nearest_node(self, point: np.array) -> tuple[NetMCNode, float]:
        distance, index = self.kdtree.query(point)
        return self.nodes[index], distance

    @property
    def kdtree(self):
        return KDTree(np.array([[node.x, node.y] for node in self.nodes]))

    @property
    def bonds(self) -> list[NetMCBond]:
        """
        Computationally expensive, do not use regularly.
        """
        bonds = []
        for node in self.nodes:
            for neighbour in node.neighbours:
                bond = NetMCBond(node, neighbour)
                if bond not in bonds:
                    bonds.append(bond)
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

    def get_angles(self) -> Iterator[tuple[NetMCNode, NetMCNode, NetMCNode]]:
        for node in self.nodes:
            for angle in node.get_angles():
                yield angle

    def evaluate_strain(self, ideal_bond_length: float, ideal_bond_angle: float) -> float:
        """
        Evaluates the strain of a network using a very basic model. E = sum((l - l_0)^2) + sum((theta - theta_0)^2)
        Bond angles are in degrees.
        """
        bond_length_strain = np.sum(
            [(bond.length - ideal_bond_length) ** 2 for bond in self.bonds])
        angle_strain = 0
        for angle in self.get_angles():
            angle_deviation = calculate_angle(
                angle[0].coord, angle[1].coord, angle[2].coord) - ideal_bond_angle
            angle_strain += angle_deviation ** 2
        return bond_length_strain + angle_strain

    def export(self, dimensions: np.ndarray, path: Path, prefix: str) -> None:
        self.export_aux(path.joinpath(
            f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_aux.dat"), dimensions)
        self.export_coords(path.joinpath(
            f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_crds.dat"))
        self.export_base_bonds(path.joinpath(
            f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_net.dat"))
        self.export_ring_bonds(path.joinpath(
            f"{prefix}_{NETWORK_TYPE_MAP[self.type]}_dual.dat"))

    def export_aux(self, path: Path, dimensions: np.ndarray) -> None:
        with open(path, "w") as aux_file:
            aux_file.write(f"{self.num_nodes}\n")
            aux_file.write(f"{self.max_connections:<10}{self.max_ring_connections:<10}\n")
            aux_file.write(f"{self.geom_code}\n")
            aux_file.write(f"{dimensions[1][0]:<20.6f}{dimensions[1][1]:<20.6f}\n")
            aux_file.write(f"{dimensions[0][0]:<20.6f}{dimensions[0][1]:<20.6f}\n")

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
    def max_connections(self):
        return max([node.num_neighbours for node in self.nodes])

    @property
    def max_ring_connections(self):
        return max([node.num_ring_neighbours for node in self.nodes])

    @property
    def node_xlo(self):
        return min([node.x for node in self.nodes])

    @property
    def node_xhi(self):
        return max([node.x for node in self.nodes])

    @property
    def node_ylo(self):
        return min([node.y for node in self.nodes])

    @property
    def node_yhi(self):
        return max([node.y for node in self.nodes])

    def __eq__(self, other):
        if isinstance(other, NetMCNetwork):
            return (self.nodes == other.nodes and
                    self.type == other.type and
                    self.geom_code == other.geom_code)
        return False


@dataclass
class NetMCData:
    base_network: NetMCNetwork = field(
        default_factory=lambda: NetMCNetwork([], "base"))
    ring_network: NetMCNetwork = field(
        default_factory=lambda: NetMCNetwork([], "ring"))
    dimensions: np.ndarray = field(
        default_factory=lambda: np.array([[0, 0], [1, 1]]))

    @staticmethod
    def from_files(path: Path, prefix: str):
        base_nodes = get_nodes(path.joinpath(f"{prefix}_A_crds.dat"), "base")
        ring_nodes = get_nodes(path.joinpath(f"{prefix}_B_crds.dat"), "ring")
        fill_neighbours(path.joinpath(f"{prefix}_A_net.dat"), base_nodes, base_nodes)
        fill_neighbours(path.joinpath(f"{prefix}_B_net.dat"), ring_nodes, ring_nodes)
        fill_neighbours(path.joinpath(f"{prefix}_A_dual.dat"), base_nodes, ring_nodes)
        fill_neighbours(path.joinpath(f"{prefix}_B_dual.dat"), ring_nodes, base_nodes)
        dimensions, base_geom_code = get_aux_data(path.joinpath(f"{prefix}_A_aux.dat"))
        _, ring_geom_code = get_aux_data(path.joinpath(f"{prefix}_B_aux.dat"))
        base_network = NetMCNetwork(base_nodes, "base", base_geom_code)
        ring_network = NetMCNetwork(ring_nodes, "ring", ring_geom_code)
        return NetMCData(base_network, ring_network, dimensions)

    @staticmethod
    def gen_triangle_lattice(num_rings: int) -> NetMCData:
        netmc_data = NetMCData()
        length = rounded_sqrt(num_rings)
        num_ring_nodes = length * length
        num_base_nodes = 2 * num_ring_nodes
        dy = np.sqrt(3) / 2  # Distance between rows

        # Add ring nodes
        for y in range(length):
            for x in range(length):
                ring_node_coord = np.array([1 / 2 * (y % 2) + x, (1 / 2 + y) * dy])
                netmc_data.add_node(NetMCNode(ring_node_coord, "ring", [], []))

        # Add base nodes in two loops to get left-to-right, bottom-to-top ordering of IDs
        for y in range(length):
            # Add lower base nodes
            for x in range(length):
                node = netmc_data.ring_network.nodes[y * length + x]
                base_node_coord = node.coord + np.array([1 / 2, -np.sqrt(3) / 6])
                netmc_data.add_node(NetMCNode(base_node_coord, "base", [], []))
            # Add upper base nodes
            for x in range(length):
                node = netmc_data.ring_network.nodes[y * length + x]
                base_node_coord = node.coord + np.array([1 / 2, np.sqrt(3) / 6])
                netmc_data.add_node(NetMCNode(base_node_coord, "base", [], []))

        # Add connections
        for id in range(num_ring_nodes):
            y, x = divmod(id, length)
            # Add neighbours to left and right
            ring_ring_neighbours = [(y * length + (id + offset) % length) % num_ring_nodes for offset in [-1, 1]]
            # Add dual neighbours to left and right (slightly below ring node)
            ring_base_neighbours = [(2 * y * length + (id + offset) % length) % num_base_nodes for offset in [-1, 0]]
            # Add dual neighbours to left and right (slightly above ring node)
            ring_base_neighbours += [((2 * y + 1) * length + (id + offset) % length) % num_base_nodes for offset in [-1, 0]]

            if y % 2 == 0:
                # Add neighbours above to the left and right
                ring_ring_neighbours += [((y + 1) * length + (id + offset) % length) % num_ring_nodes for offset in [-1, 0]]
                # Add neighbours below to the left and right
                ring_ring_neighbours += [((y - 1) * length + (id + offset) % length) % num_ring_nodes for offset in [-1, 0]]
                # Add dual neighbour above
                ring_base_neighbours += [((2 * y + 2) * length + (id - 1) % length) % num_base_nodes]
                # Add dual neighbour below
                ring_base_neighbours += [((2 * y - 1) * length + (id - 1) % length) % num_base_nodes]
            else:
                # Add neighbours above to the left and right
                ring_ring_neighbours += [((y + 1) * length + (id + offset) % length) % num_ring_nodes for offset in [0, 1]]
                # Add neighbours below to the left and right
                ring_ring_neighbours += [((y - 1) * length + (id + offset) % length) % num_ring_nodes for offset in [0, 1]]
                # Add dual neighbour above
                ring_base_neighbours += [((2 * y + 2) * length + (id) % length) % num_base_nodes]
                # Add dual neighbour below
                ring_base_neighbours += [((2 * y - 1) * length + (id) % length) % num_base_nodes]
            for neighbour in ring_ring_neighbours:
                netmc_data.ring_network.nodes[id].add_neighbour(netmc_data.ring_network.nodes[neighbour])
            for neighbour in ring_base_neighbours:
                netmc_data.ring_network.nodes[id].add_ring_neighbour(netmc_data.base_network.nodes[neighbour])
                netmc_data.base_network.nodes[neighbour].add_ring_neighbour(netmc_data.ring_network.nodes[id])
        for id in range(num_base_nodes):
            y, x = divmod(id, length)
            # Add neighbours to left and right
            if y % 4 == 0:
                base_base_neighbours = [((y - 1) * length + (id + offset) % length) % num_base_nodes for offset in [-1, 0]]
                base_base_neighbours += [((y + 1) * length + (id) % length) % num_base_nodes]
            elif y % 4 == 1:
                base_base_neighbours = [((y - 1) * length + (id) % length) % num_base_nodes]
                base_base_neighbours += [((y + 1) * length + (id + offset) % length) % num_base_nodes for offset in [-1, 0]]
            elif y % 4 == 2:
                base_base_neighbours = [((y - 1) * length + (id + offset) % length) % num_base_nodes for offset in [0, 1]]
                base_base_neighbours += [((y + 1) * length + (id) % length) % num_base_nodes]
            else:
                base_base_neighbours = [((y - 1) * length + (id) % length) % num_base_nodes]
                base_base_neighbours += [((y + 1) * length + (id + offset) % length) % num_base_nodes for offset in [0, 1]]
            for neighbour in base_base_neighbours:
                netmc_data.base_network.nodes[id].add_neighbour(netmc_data.base_network.nodes[neighbour])

        netmc_data.set_dimensions(np.array([[0, 0], [length, length * np.sqrt(3) / 2]]))
        return netmc_data

    def set_dimensions(self, dimensions: np.ndarray) -> None:
        self.dimensions = dimensions

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
        self.base_network.add_node(node)
        self.ring_network.add_node(node)
        for neighbour in node.neighbours:
            neighbour.add_neighbour(node)
        for ring_neighbour in node.ring_neighbours:
            ring_neighbour.add_ring_neighbour(node)

    def delete_node_and_merge_rings(self, node: NetMCNode) -> None:
        rings_to_merge = node.ring_neighbours.copy()
        # I have to use settify instead of set() because the nodes are not hashable
        all_neighbours = settify(
            [neighbour for node_to_merge in rings_to_merge for neighbour in node_to_merge.neighbours if neighbour not in rings_to_merge])
        all_ring_neighbours = settify(
            [neighbour for node_to_merge in rings_to_merge for neighbour in node_to_merge.ring_neighbours])
        new_coord = np.mean(
            [ring_neighbour.coord for ring_neighbour in all_ring_neighbours], axis=0)
        new_node = NetMCNode(
            new_coord, rings_to_merge[0].type, all_neighbours, all_ring_neighbours)
        self.add_node(new_node)
        for node_to_merge in rings_to_merge:
            self.delete_node(node_to_merge)
        self.delete_node(node)

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

    def get_undrecoordinated_nodes(self, network: NetMCNetwork, target_coordination: int) -> list[NetMCNode]:
        undercoordinated_nodes = []
        for node in network.nodes:
            if node.num_neighbours < target_coordination:
                undercoordinated_nodes.append(node)
        return undercoordinated_nodes

    @staticmethod
    def check_undercoordinated(undercoordinated_nodes: list[NetMCNode], ring_walk: list[NetMCNode]) -> None:
        if len(undercoordinated_nodes) % 2 != 0:
            raise InvalidUndercoordinatedNodesException("Number of undercoordinated nodes is odd, so cannot bond them.")
        # Check there are no three consecutive undercoordinated nodes in the ring walk
        for node_1, node_2, node_3 in zip(ring_walk, ring_walk[1:] + ring_walk[:1], ring_walk[2:] + ring_walk[:2]):
            if node_1 in undercoordinated_nodes and node_2 in undercoordinated_nodes and node_3 in undercoordinated_nodes:
                raise InvalidUndercoordinatedNodesException("There are three consecutive undercoordinated nodes in the ring walk.")
        islands = []
        for i, node in enumerate(undercoordinated_nodes):
            next_node = undercoordinated_nodes[(
                i + 1) % len(undercoordinated_nodes)]
            if next_node in node.neighbours:
                islands.append([node, next_node])
        for i, island in enumerate(islands):
            font_of_island = island[1]
            back_of_next_island = islands[(i + 1) % len(islands)][0]
            undercoordinated_nodes_between_islands = [node for node in undercoordinated_nodes if
                                                      ring_walk.index(font_of_island) < ring_walk.index(node) < ring_walk.index(back_of_next_island)]
            if len(undercoordinated_nodes_between_islands) % 2 != 0:
                raise InvalidUndercoordinatedNodesException("There are an odd number of undercoordinated nodes between two adjacent undercoordinated nodes.")

    @staticmethod
    def arrange_undercoordinated(undercoordinated_nodes: list[NetMCNode]) -> tuple[list[NetMCNode], list[NetMCNode]]:
        common_rings = find_common_elements(
            [node.ring_neighbours for node in undercoordinated_nodes])
        if len(common_rings) != 1:
            raise InvalidUndercoordinatedNodesException("Undercoordinated nodes do not share a common ring, so cannot bond them.")
        common_ring = common_rings[0]
        ring_walk = common_ring.get_ring_walk()
        undercoordinated_nodes.sort(key=lambda node: ring_walk.index(node))
        return undercoordinated_nodes, ring_walk

    @staticmethod
    def bond_undercoordinated_nodes(netmc_data: NetMCData) -> NetMCData:
        potential_network_1 = copy.deepcopy(netmc_data)
        potential_network_2 = copy.deepcopy(netmc_data)
        potential_network_1.flip_flop(direction=1)
        potential_network_2.flip_flop(direction=-1)
        if potential_network_1.flopped and potential_network_2.flopped:
            raise CouldNotBondUndercoordinatedNodesException("Could not bond undercoordinated nodes.")
        elif potential_network_1.flopped:
            return potential_network_2
        elif potential_network_2.flopped:
            return potential_network_1
        else:
            return NetMCData.compare_networks(potential_network_1, potential_network_2)

    @staticmethod
    def compare_networks(network_1: NetMCData, network_2: NetMCData) -> NetMCData:
        network_1_strain = network_1.base_network.evaluate_strain(
            ideal_bond_length=1, ideal_bond_angle=120)
        network_2_strain = network_2.base_network.evaluate_strain(
            ideal_bond_length=1, ideal_bond_angle=120)
        print(f"Network 1 strain: {network_1_strain} Network 2 strain: {network_2_strain}")
        if network_1_strain < network_2_strain:
            return network_1
        if network_1_strain == network_2_strain:
            return network_1
        return network_2

    def flip_flop(self, direction: int) -> None:
        """
        Function is called flip flop because it's used to generate the two different possible networks when bonding undercoordinated nodes.
        The movement of every bond seems to 'flip-flop' between the two networks.
        """
        self.flopped = False  # Essentially means the network is invalid if True
        undercoordinated_nodes = self.get_undrecoordinated_nodes(self.base_network, 3)
        undercoordinated_nodes, ring_walk = self.arrange_undercoordinated(undercoordinated_nodes)
        self.check_undercoordinated(undercoordinated_nodes, ring_walk)
        if direction == -1:
            undercoordinated_nodes = undercoordinated_nodes[-1:] + undercoordinated_nodes[:-1]
        while len(undercoordinated_nodes) > 0:
            selected_node = undercoordinated_nodes.pop(0)
            next_node = undercoordinated_nodes.pop(0)
            if next_node in selected_node.neighbours:
                # Cannot bond undercoordinated nodes that are already bonded, this is a dead end
                self.flopped = True
                return
            new_ring_1, new_ring_2, ring_to_remove = self.get_resulting_rings(selected_node, next_node)
            self.add_bond(selected_node, next_node)
            self.split_ring(new_ring_1, new_ring_2, ring_to_remove)

    def get_resulting_rings(self, node_1: NetMCNode, node_2: NetMCNode) -> tuple[NetMCNode, NetMCNode, NetMCNode]:
        try:
            common_ring = find_common_elements(
                [node_1.ring_neighbours, node_2.ring_neighbours])[0]
        except IndexError:
            raise InvalidUndercoordinatedNodesException("Undercoordinated nodes do not share a common ring, so cannot bond them.")
        ring_walk = common_ring.get_ring_walk()
        index_1 = ring_walk.index(node_1)
        index_2 = ring_walk.index(node_2)
        # get base network nodes
        if index_1 < index_2:
            new_ring_node_1_ring_nodes = ring_walk[index_1:index_2 + 1]
            new_ring_node_2_ring_nodes = ring_walk[index_2:] + \
                ring_walk[:index_1 + 1]
        else:
            new_ring_node_1_ring_nodes = ring_walk[index_2:index_1 + 1]
            new_ring_node_2_ring_nodes = ring_walk[index_1:] + \
                ring_walk[:index_2 + 1]
        # get ring network nodes
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
                                    "ring", new_ring_node_1_nodes, new_ring_node_1_ring_nodes)
        new_ring_node_2 = NetMCNode(np.mean([ring_node.coord for ring_node in new_ring_node_2_ring_nodes], axis=0),
                                    "ring", new_ring_node_2_nodes, new_ring_node_2_ring_nodes)
        return new_ring_node_1, new_ring_node_2, common_ring

    def split_ring(self, new_ring_node_1: NetMCNode, new_ring_node_2: NetMCNode, ring_to_remove: NetMCNode) -> None:
        self.add_node(new_ring_node_1)
        self.add_node(new_ring_node_2)
        self.delete_node(ring_to_remove)
        self.add_bond(new_ring_node_1, new_ring_node_2)

    @property
    def graph(self) -> nx.Graph:
        # Assumes base-ring connections are the same as ring-base connections
        graph = nx.Graph()
        for node in self.base_network.nodes:
            graph.add_node(node.id, pos=(node.x, node.y),
                           source='base_network')
        for bond in self.base_network.bonds:
            if bond.length < 2 * self.base_network.avg_bond_length:
                graph.add_edge(bond.node_1.id, bond.node_2.id,
                               source='base_network')
        for bond in self.base_network.ring_bonds:
            if bond.length < 2 * self.base_network.avg_ring_bond_length:
                graph.add_edge(bond.node_1.id, self.base_network.num_nodes +
                               bond.node_2.id, source='base_ring_bonds')
        for node in self.ring_network.nodes:
            graph.add_node(self.base_network.num_nodes + node.id,
                           pos=(node.x, node.y), source='ring_network')
        for bond in self.ring_network.bonds:
            if bond.length < 2 * self.ring_network.avg_bond_length:
                graph.add_edge(self.base_network.num_nodes + bond.node_1.id,
                               self.base_network.num_nodes + bond.node_2.id, source='ring_network')
        return graph

    def draw_graph(self, base_nodes: bool = True, ring_nodes: bool = False,
                   base_bonds: bool = True, ring_bonds: bool = False, base_ring_bonds: bool = False,
                   base_labels: bool = False, ring_labels: bool = False, offset: float = 0.2) -> None:
        graph = nx.Graph()
        id_shift = 0
        if base_nodes:
            for node in self.base_network.nodes:
                graph.add_node(node.id, pos=(node.x, node.y),
                               source='base_network')
            id_shift += self.base_network.num_nodes
        if ring_nodes:
            for node in self.ring_network.nodes:
                graph.add_node(node.id + id_shift, pos=(node.x, node.y), source='ring_network')
        if base_bonds:
            for node in self.base_network.nodes:
                for neighbour in node.neighbours:
                    if neighbour.id > node.id:
                        if not is_pbc_bond(node, neighbour, self.dimensions):
                            graph.add_edge(node.id, neighbour.id, source='base_network')
        if ring_bonds:
            for node in self.ring_network.nodes:
                for neighbour in node.neighbours:
                    if neighbour.id > node.id:
                        if not is_pbc_bond(node, neighbour, self.dimensions):
                            graph.add_edge(node.id + id_shift, neighbour.id + id_shift, source='ring_network')
        if base_ring_bonds:
            for node in self.ring_network.nodes:
                for neighbour in node.ring_neighbours:
                    if not is_pbc_bond(node, neighbour, self.dimensions):
                        graph.add_edge(node.id + id_shift, neighbour.id, source='base_ring_bonds')

        node_colors = ['red' if data['source'] == 'base_network' else 'blue' for _, data in graph.nodes(data=True)]
        edge_colors = []
        for _, _, data in graph.edges(data=True):
            if data['source'] == 'base_network':
                edge_colors.append('black')
            elif data['source'] == 'ring_network':
                edge_colors.append('green')
            else:
                edge_colors.append('blue')
        node_sizes = [30 if data['source'] == 'base_network' else 10 for _, data in graph.nodes(data=True)]
        edge_widths = [2 if data['source'] == 'base_network' else 1 for _, _, data in graph.edges(data=True)]
        pos = nx.get_node_attributes(graph, "pos")
        pos_labels = {node: (x + offset, y + offset)
                      for node, (x, y) in pos.items()}
        nx.draw(graph, pos, node_color=node_colors,
                edge_color=edge_colors, node_size=node_sizes, width=edge_widths)
        if base_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={
                                    node.id: node.id for node in self.base_network.nodes}, font_size=7, font_color="gray")
        if ring_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={
                                    node.id + id_shift: node.id for node in self.ring_network.nodes}, font_size=7, font_color="purple")

    @property
    def xlo(self):
        return self.dimensions[0, 0]

    @property
    def xhi(self):
        return self.dimensions[1, 0]

    @property
    def ylo(self):
        return self.dimensions[0, 1]

    @property
    def yhi(self):
        return self.dimensions[1, 1]

    def __deepcopy__(self, memo) -> NetMCData:
        copied_base_nodes_dict = {id(node): NetMCNode(
            coord=node.coord, id=node.id, type=node.type) for node in self.base_network.nodes}
        copied_ring_nodes_dict = {id(node): NetMCNode(
            coord=node.coord, id=node.id, type=node.type) for node in self.ring_network.nodes}
        for node in self.base_network.nodes:
            copied_node = copied_base_nodes_dict[id(node)]
            copied_node.neighbours = [copied_base_nodes_dict[id(
                neighbour)] for neighbour in node.neighbours]
            copied_node.ring_neighbours = [copied_ring_nodes_dict[id(
                neighbour)] for neighbour in node.ring_neighbours]
        for node in self.ring_network.nodes:
            copied_node = copied_ring_nodes_dict[id(node)]
            copied_node.neighbours = [copied_ring_nodes_dict[id(
                neighbour)] for neighbour in node.neighbours]
            copied_node.ring_neighbours = [copied_base_nodes_dict[id(
                neighbour)] for neighbour in node.ring_neighbours]
        return NetMCData(NetMCNetwork(list(copied_base_nodes_dict.values()), "base", self.base_network.geom_code),
                         NetMCNetwork(list(copied_ring_nodes_dict.values()), "ring", self.ring_network.geom_code))


@dataclass
class NetMCNode:
    coord: np.array
    type: str
    neighbours: list[NetMCNode] = field(default_factory=lambda: [])
    ring_neighbours: list[NetMCNode] = field(default_factory=lambda: [])
    id: Optional[int] = None

    def check(self) -> bool:
        valid = True
        if self.type not in ("base", "ring"):
            print(f"Node {self.id} has invalid type {self.type}")
            valid = False
        for neighbour in self.neighbours:
            if neighbour == self:
                print(f"Node {self.id} ({self.type}) has itself as neighbour")
                valid = False
            if self not in neighbour.neighbours:
                print(f"Node {self.id} ({self.type}) has neighbour {neighbour.id}, but neighbour does not have node as neighbour")
                valid = False
            if self.type != neighbour.type:
                print(f"Node {self.id} ({self.type}) has neighbour {neighbour.id}, but neighbour has different type")
                valid = False
        for ring_neighbour in self.ring_neighbours:
            if self not in ring_neighbour.ring_neighbours:
                print(f"Node {self.id} ({self.type}) has ring neighbour {ring_neighbour.id}, but ring neighbour does not have node as ring neighbour")
                valid = False
            if self.type == ring_neighbour.type:
                print(f"Node {self.id} ({self.type}) has ring neighbour {ring_neighbour.id}, but ring neighbour has same type")
                valid = False
        return valid

    def get_ring_walk(self) -> list[NetMCNode]:
        """
        Returns a list of nodes such that the order is how they are connected in the ring.
        """
        walk = [self.ring_neighbours[0]]
        counter = 0
        while len(walk) < len(self.ring_neighbours):
            if counter > 999:
                raise ValueError(f"Could not find ring walk for node {self.id} ({self.type}) ring_neighbours: {[ring_neighbour.id for ring_neighbour in self.ring_neighbours]}")
            current_node = walk[-1]
            for neighbour in current_node.neighbours:
                if neighbour in self.ring_neighbours and neighbour not in walk:
                    walk.append(neighbour)
                    break
            counter += 1
        return walk

    def get_angles(self) -> Iterator[tuple[NetMCNode, NetMCNode, NetMCNode]]:
        for i in range(len(self.neighbours)):
            node_1 = self.neighbours[i]
            node_2 = self.neighbours[(i + 1) % len(self.neighbours)]
            yield (node_1, self, node_2)

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
        string = f"Node {self.id} {self.type} at {self.coord}. Neighbours: "
        for neighbour in self.neighbours:
            string += f"{neighbour.id}, "
        string += "Ring neighbours: "
        for ring_neighbour in self.ring_neighbours:
            string += f"{ring_neighbour.id}, "
        return string


@dataclass
class NetMCBond:
    node_1: NetMCNode
    node_2: NetMCNode

    @property
    def length(self) -> float:
        return np.linalg.norm(self.node_1.coord - self.node_2.coord)

    def pbc_length(self, dimensions: np.ndarray) -> float:
        return np.linalg.norm(pbc_vector(self.node_1.coord, self.node_2.coord, dimensions))

    @property
    def type(self) -> str:
        return f"{self.node_1.type}-{self.node_2.type}"

    def check(self) -> bool:
        if self.node_1 == self.node_2:
            print(f"Bond ({self.type} between node_1: {self.node_1.id} node_2: {self.node_2.id} bonds identical nodes")
            return False
        return True

    def __eq__(self, other) -> bool:
        if isinstance(other, NetMCBond):
            return ((self.node_1 == other.node_1 and self.node_2 == other.node_2) or
                    (self.node_1 == other.node_2 and self.node_2 == other.node_1))
        return False

    def __repr__(self) -> str:
        return f"Bond of type {self.type} between {self.node_1.id} and {self.node_2.id}"
