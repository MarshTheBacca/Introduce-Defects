from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backend_bases import KeyEvent, ResizeEvent
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde

from .bss_network import BSSNetwork
from .bss_node import BSSNode
from .other_utils import (find_common_elements, pbc_vector, rounded_even_sqrt,
                          settify)


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


def get_nodes(path: Path, network_type: str) -> list[BSSNode]:
    """
    Read a file containing node coordinates and return a list of BSSNode objects.
    """
    if network_type not in ("base", "ring"):
        raise ValueError("Invalid network type {network_type}")
    coords = np.genfromtxt(path)
    nodes = [BSSNode(coord=coord, neighbours=[], ring_neighbours=[], id=id, type=network_type) for id, coord in enumerate(coords)]
    return nodes


def fill_neighbours(path: Path, selected_nodes: list[BSSNode], bonded_nodes: list[BSSNode], dimensions: np.ndarray) -> None:
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
                    node_1.add_neighbour(node_2, dimensions)
                elif node_2.type != node_1.type and node_2 not in node_1.ring_neighbours:
                    node_1.add_ring_neighbour(node_2, dimensions)


def get_info_data(path: Path) -> np.ndarray:
    """
    Reads the dimensions from a BSS info file (num nodes is not necessary)
    """
    with open(path, "r") as info_file:
        info_file.readline()
        xhi = info_file.readline().strip().split()[1]
        yhi = info_file.readline().strip().split()[1]
    dimensions = np.array([[0, 0], [float(xhi), float(yhi)]])
    return dimensions


def get_fixed_rings(path: Path) -> set[int]:
    with open(path, "r") as fixed_rings_file:
        return {int(ring.strip()) for ring in fixed_rings_file.readlines() if ring.strip()}


def is_pbc_bond(node_1: BSSNode, node_2: BSSNode, dimensions: np.ndarray) -> bool:
    """
    Identifies bonds that cross the periodic boundary. So if the length of the bond is 
    more than 10% longer than the distance between the two nodes with periodic boundary conditions,
    then it is considered a periodic bond.
    """
    if np.linalg.norm(node_1.coord - node_2.coord) > np.linalg.norm(pbc_vector(node_1.coord, node_2.coord, dimensions)) * 1.1:
        return True
    return False


@dataclass
class BSSData:
    base_network: BSSNetwork = field(default_factory=lambda: BSSNetwork([], "base", np.array([[0, 0], [1, 1]])))
    ring_network: BSSNetwork = field(default_factory=lambda: BSSNetwork([], "ring", np.array([[0, 0], [1, 1]])))
    dimensions: np.ndarray = field(default_factory=lambda: np.array([[0, 0], [1, 1]]))
    fixed_rings: set[int] = field(default_factory=lambda: {})

    @staticmethod
    def from_files(path: Path):
        dimensions = get_info_data(path.joinpath("base_network_info.txt"))
        base_nodes = get_nodes(path.joinpath("base_network_coords.txt"), "base")
        ring_nodes = get_nodes(path.joinpath("dual_network_coords.txt"), "ring")
        fill_neighbours(path.joinpath("base_network_connections.txt"), base_nodes, base_nodes, dimensions)
        fill_neighbours(path.joinpath("dual_network_connections.txt"), ring_nodes, ring_nodes, dimensions)
        fill_neighbours(path.joinpath("base_network_dual_connections.txt"), base_nodes, ring_nodes, dimensions)
        fill_neighbours(path.joinpath("dual_network_dual_connections.txt"), ring_nodes, base_nodes, dimensions)
        fixed_rings = get_fixed_rings(path.joinpath("fixed_rings.txt"))
        base_network = BSSNetwork(base_nodes, "base", dimensions)
        ring_network = BSSNetwork(ring_nodes, "ring", dimensions)
        return BSSData(base_network, ring_network, dimensions, fixed_rings)

    @staticmethod
    def gen_hexagonal(num_rings: int) -> BSSData:
        bss_data = BSSData()
        length = rounded_even_sqrt(num_rings)
        dimensions = np.array([[0, 0], [np.sqrt(3) * length, 1.5 * length]])
        bss_data.set_dimensions(dimensions)

        num_ring_nodes = length * length
        num_base_nodes = 2 * num_ring_nodes
        dy = 1.5  # Distance between rows of rings

        # Add ring nodes
        for y in range(length):
            for x in range(length):
                ring_node_coord = np.array([np.sqrt(3) / 2 * (- (y % 2) + 2 * x + 1.5), y * dy + 0.75])
                bss_data.add_node(BSSNode(ring_node_coord, "ring", [], []))

        # Add base nodes in two loops to get left-to-right, bottom-to-top ordering of IDs
        for y in range(length):
            # Add lower base nodes
            for x in range(length):
                node = bss_data.ring_network.nodes[y * length + x]
                if y % 2 == 0:
                    base_node_coord = node.coord + np.array([-np.sqrt(3) / 2, -1 / 2])
                else:
                    base_node_coord = node.coord + np.array([np.sqrt(3) / 2, -1 / 2])
                bss_data.add_node(BSSNode(base_node_coord, "base", [], []))
            # Add upper base nodes
            for x in range(length):
                node = bss_data.ring_network.nodes[y * length + x]
                if y % 2 == 0:
                    base_node_coord = node.coord + np.array([-np.sqrt(3) / 2, 1 / 2])
                else:
                    base_node_coord = node.coord + np.array([np.sqrt(3) / 2, 1 / 2])
                bss_data.add_node(BSSNode(base_node_coord, "base", [], []))

        # Add connections
        for id in range(num_ring_nodes):
            y, x = divmod(id, length)
            # Add neighbours to left and right
            ring_ring_neighbours = [(y * length + (id + offset) % length) % num_ring_nodes for offset in [-1, 1]]
            if y % 2 == 0:
                # Add dual neighbours to left and right (slightly below ring node)
                ring_base_neighbours = [(2 * y * length + (id + offset) % length) % num_base_nodes for offset in [0, 1]]
                # Add dual neighbours to left and right (slightly above ring node)
                ring_base_neighbours += [((2 * y + 1) * length + (id + offset) % length) % num_base_nodes for offset in [0, 1]]
                # Add neighbours above to the left and right
                ring_ring_neighbours += [((y + 1) * length + (id + offset) % length) % num_ring_nodes for offset in [0, 1]]
                # Add neighbours below to the left and right
                ring_ring_neighbours += [((y - 1) * length + (id + offset) % length) % num_ring_nodes for offset in [0, 1]]
                # Add dual neighbour above
                ring_base_neighbours += [((2 * y + 2) * length + id % length) % num_base_nodes]
                # Add dual neighbour below
                ring_base_neighbours += [((2 * y - 1) * length + id % length) % num_base_nodes]
            else:
                # Add dual neighbours to left and right (slightly below ring node)
                ring_base_neighbours = [(2 * y * length + (id + offset) % length) % num_base_nodes for offset in [-1, 0]]
                # Add dual neighbours to left and right (slightly above ring node)
                ring_base_neighbours += [((2 * y + 1) * length + (id + offset) % length) % num_base_nodes for offset in [-1, 0]]
                # Add neighbours above to the left and right
                ring_ring_neighbours += [((y + 1) * length + (id + offset) % length) % num_ring_nodes for offset in [-1, 0]]
                # Add neighbours below to the left and right
                ring_ring_neighbours += [((y - 1) * length + (id + offset) % length) % num_ring_nodes for offset in [-1, 0]]
                # Add dual neighbour above
                ring_base_neighbours += [((2 * y + 2) * length + (id) % length) % num_base_nodes]
                # Add dual neighbour below
                ring_base_neighbours += [((2 * y - 1) * length + (id) % length) % num_base_nodes]
            for neighbour in ring_ring_neighbours:
                bss_data.ring_network.nodes[id].add_neighbour(bss_data.ring_network.nodes[neighbour], dimensions)
            for neighbour in ring_base_neighbours:
                bss_data.ring_network.nodes[id].add_ring_neighbour(bss_data.base_network.nodes[neighbour], dimensions)
                bss_data.base_network.nodes[neighbour].add_ring_neighbour(bss_data.ring_network.nodes[id], dimensions)
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
                bss_data.base_network.nodes[id].add_neighbour(bss_data.base_network.nodes[neighbour], dimensions)
        return bss_data

    def set_dimensions(self, dimensions: np.ndarray) -> None:
        self.dimensions = dimensions.copy()
        self.dimensions = dimensions.copy()
        self.base_network.dimensions = dimensions.copy()
        self.ring_network.dimensions = dimensions.copy()

    def plot_radial_distribution(self) -> None:
        distances_from_centre = [np.linalg.norm(node.coord - np.mean(self.dimensions, axis=0)) for node in self.base_network.nodes]

        # Calculate the KDE
        kde = gaussian_kde(distances_from_centre)
        radii = np.linspace(0, np.linalg.norm(self.dimensions[1] - self.dimensions[0]) / 2, 1000)
        density = kde(radii)

        # Normalize by the area of the annulus
        bin_width = radii[1] - radii[0]
        areas = 2 * np.pi * radii * bin_width
        density_normalized = density / areas

        plt.plot(radii, density_normalized, label="Base")
        plt.title("Radial distribution of nodes in the base network from the centre")
        plt.xlabel("Distance from centre (Bohr radii)")
        plt.ylabel("Density (Atoms Bohr radii ^ - 2)")
        plt.show()

    def scale(self, scale_factor: float) -> None:
        self.dimensions *= scale_factor
        self.base_network.scale(scale_factor)
        self.ring_network.scale(scale_factor)

    def export(self, path: Path) -> None:
        self.base_network.export(self.dimensions, path)
        self.ring_network.export(self.dimensions, path)
        self.export_fixed_rings(path.joinpath("fixed_rings.txt"))

    def export_fixed_rings(self, path: Path) -> None:
        with open(path, "w") as file:
            for ring in self.fixed_rings:
                file.write(f"{ring}\n")

    def check(self) -> bool:
        if self.base_network.check() and self.ring_network.check():
            return True
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, BSSData):
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

    def delete_node(self, node: BSSNode) -> None:
        for neighbour in node.neighbours:
            neighbour.delete_neighbour(node)
        for ring_neighbour in node.ring_neighbours:
            ring_neighbour.delete_ring_neighbour(node)
        self.base_network.delete_node(node)
        self.ring_network.delete_node(node)

    def add_node(self, node: BSSNode) -> None:
        self.base_network.add_node(node)
        self.ring_network.add_node(node)
        for neighbour in node.neighbours:
            neighbour.add_neighbour(node, self.dimensions)
        for ring_neighbour in node.ring_neighbours:
            ring_neighbour.add_ring_neighbour(node, self.dimensions)

    def merge_rings(self, rings_to_merge: list[BSSNode], reference_coord: Optional[np.ndarray] = None) -> None:
        peripheral_rings = settify([neighbour for ring_node in rings_to_merge for neighbour in ring_node.neighbours
                                    if neighbour not in rings_to_merge])
        peripheral_nodes = settify([neighbour for ring_node in rings_to_merge for neighbour in ring_node.ring_neighbours])
        if reference_coord is None:
            new_coord = np.mean([base_node.coord for base_node in peripheral_nodes], axis=0)
        else:
            new_coord = np.mean([reference_coord + pbc_vector(reference_coord, base_node.coord, self.dimensions)
                                for base_node in peripheral_nodes], axis=0)
        new_node = BSSNode(new_coord, rings_to_merge[0].type, peripheral_rings, peripheral_nodes)
        self.add_node(new_node)
        for ring_node in rings_to_merge:
            self.delete_node(ring_node)

    def delete_node_and_merge_rings(self, node: BSSNode) -> None:
        rings_to_merge = node.ring_neighbours.copy()
        self.merge_rings(rings_to_merge, np.copy(node.coord))
        self.delete_node(node)

    def manual_bond_addition_deletion(self, node_1: BSSNode, node_2: BSSNode) -> None:
        if node_1 == node_2:
            raise ValueError("Cannot bond or remove bond between the same node.")
        if node_1 not in node_2.neighbours:  # Add the bond and create a new ring node
            new_ring_1, new_ring_2, ring_to_remove = self.get_resulting_rings(node_1, node_2)
            self.add_bond(node_1, node_2)
            self.split_ring(new_ring_1, new_ring_2, ring_to_remove)
            return
        # If it's already bonded, remove the bond and merge the rings
        common_rings: list[BSSNode] = find_common_elements([node_1.ring_neighbours, node_2.ring_neighbours])
        self.delete_bond(node_1, node_2)
        self.merge_rings(common_rings)

    def add_bond(self, node_1: BSSNode, node_2: BSSNode) -> None:
        if node_1 == node_2:
            raise ValueError("Cannot add bond between the same node.")
        if node_1.type == "base" and node_1 not in self.base_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_1.id} is not in the base network")
        if node_2.type == "base" and node_2 not in self.base_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_2.id} is not in the base network")
        if node_1.type == "ring" and node_1 not in self.ring_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_1.id} is not in the ring network")
        if node_2.type == "ring" and node_2 not in self.ring_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_2.id} is not in the ring network")
        node_1.add_neighbour(node_2, self.dimensions)
        node_2.add_neighbour(node_1, self.dimensions)

    def delete_bond(self, node_1: BSSNode, node_2: BSSNode) -> None:
        if node_1 == node_2:
            raise ValueError("Cannot remove bond between the same node.")
        if node_1.type == "base" and node_1 not in self.base_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_1.id} is not in the base network")
        if node_2.type == "base" and node_2 not in self.base_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_2.id} is not in the base network")
        if node_1.type == "ring" and node_1 not in self.ring_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_1.id} is not in the ring network")
        if node_2.type == "ring" and node_2 not in self.ring_network.nodes:
            raise ValueError(f"Cannot add bond between nodes {node_1.id} and {node_2.id} because node {node_2.id} is not in the ring network")
        node_1.delete_neighbour(node_2)
        node_2.delete_neighbour(node_1)

    def get_undercoordinated_nodes(self, network: BSSNetwork, target_coordination: int) -> list[BSSNode]:
        return [node for node in network.nodes if node.num_neighbours < target_coordination]

    @staticmethod
    def check_undercoordinated(undercoordinated_nodes: list[BSSNode], ring_walk: list[BSSNode]) -> None:
        if len(undercoordinated_nodes) % 2 != 0:
            raise InvalidUndercoordinatedNodesException("Number of undercoordinated nodes is odd, so cannot bond them.")
        # Check there are no three consecutive undercoordinated nodes in the ring walk
        for node_1, node_2, node_3 in zip(ring_walk, ring_walk[1:] + ring_walk[:1], ring_walk[2:] + ring_walk[:2]):
            if node_1 in undercoordinated_nodes and node_2 in undercoordinated_nodes and node_3 in undercoordinated_nodes:
                raise InvalidUndercoordinatedNodesException("There are three consecutive undercoordinated nodes in the ring walk.")
        # Islands are two undercoordinated nodes that are bonded
        islands = []
        for i, node in enumerate(undercoordinated_nodes):
            next_node = undercoordinated_nodes[(i + 1) % len(undercoordinated_nodes)]
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
    def arrange_undercoordinated(undercoordinated_nodes: list[BSSNode]) -> tuple[list[BSSNode], list[BSSNode]]:
        common_rings = find_common_elements([node.ring_neighbours for node in undercoordinated_nodes])
        if len(common_rings) != 1:
            raise InvalidUndercoordinatedNodesException("Undercoordinated nodes do not share a common ring, so cannot bond them.")
        common_ring: BSSNode = common_rings[0]
        ring_walk = common_ring.get_ring_walk()
        undercoordinated_nodes.sort(key=lambda node: ring_walk.index(node))
        return undercoordinated_nodes, ring_walk

    @staticmethod
    def bond_undercoordinated_nodes(bss_data: BSSData) -> BSSData:
        if not bss_data.get_undercoordinated_nodes(bss_data.base_network, 3):
            print("No undercoordinated nodes in base network, leaving network unchanged")
            return bss_data
        potential_network_1 = copy.deepcopy(bss_data)
        potential_network_2 = copy.deepcopy(bss_data)
        potential_network_1.flip_flop(direction=1)
        potential_network_2.flip_flop(direction=-1)
        if potential_network_1.flopped and potential_network_2.flopped:
            raise CouldNotBondUndercoordinatedNodesException("Could not bond undercoordinated nodes, both networks are invalid")
        elif potential_network_1.flopped:
            return potential_network_2
        elif potential_network_2.flopped:
            return potential_network_1
        else:
            return BSSData.compare_networks(potential_network_1, potential_network_2)

    @staticmethod
    def compare_networks(network_1: BSSData, network_2: BSSData) -> BSSData:
        network_1_strain = network_1.base_network.evaluate_strain(ideal_bond_length=1, ideal_bond_angle=120)
        network_2_strain = network_2.base_network.evaluate_strain(ideal_bond_length=1, ideal_bond_angle=120)
        print(f"Network 1 strain: {network_1_strain:.2f} Network 2 strain: {network_2_strain:.2f}")
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
        undercoordinated_nodes = self.get_undercoordinated_nodes(self.base_network, 3)
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
        for node in self.base_network.nodes:
            node.sort_neighbours_clockwise(node.neighbours, self.dimensions)
        for ring_node in self.ring_network.nodes:
            ring_node.sort_neighbours_clockwise(node.ring_neighbours, self.dimensions)

    def get_resulting_rings(self, node_1: BSSNode, node_2: BSSNode) -> tuple[BSSNode, BSSNode, BSSNode]:
        """
        Given two nodes, construct the two new ring nodes needed and find out the one to remove
        Returns a tuple of ring nodes, the first two are to be created, the last is to be removed
        """
        try:
            common_ring = find_common_elements([node_1.ring_neighbours, node_2.ring_neighbours])[0]
        except IndexError:
            raise InvalidUndercoordinatedNodesException("Undercoordinated nodes do not share a common ring, so cannot bond them.")
        ring_walk = common_ring.get_ring_walk()
        index_1 = ring_walk.index(node_1)
        index_2 = ring_walk.index(node_2)
        # get base network nodes
        if index_1 < index_2:
            new_ring_node_1_ring_nodes = ring_walk[index_1:index_2 + 1]
            new_ring_node_2_ring_nodes = ring_walk[index_2:] + ring_walk[:index_1 + 1]
        else:
            new_ring_node_1_ring_nodes = ring_walk[index_2:index_1 + 1]
            new_ring_node_2_ring_nodes = ring_walk[index_1:] + ring_walk[:index_2 + 1]
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
        new_ring_node_1 = BSSNode(np.mean([ring_node.coord for ring_node in new_ring_node_1_ring_nodes], axis=0),
                                  "ring", new_ring_node_1_nodes, new_ring_node_1_ring_nodes)
        new_ring_node_2 = BSSNode(np.mean([ring_node.coord for ring_node in new_ring_node_2_ring_nodes], axis=0),
                                  "ring", new_ring_node_2_nodes, new_ring_node_2_ring_nodes)
        return new_ring_node_1, new_ring_node_2, common_ring

    def split_ring(self, new_ring_node_1: BSSNode, new_ring_node_2: BSSNode, ring_to_remove: BSSNode) -> None:
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

        node_colours = ['red' if data['source'] == 'base_network' else 'blue' for _, data in graph.nodes(data=True)]
        edge_colours = []
        for _, _, data in graph.edges(data=True):
            if data['source'] == 'base_network':
                edge_colours.append('black')
            elif data['source'] == 'ring_network':
                edge_colours.append('green')
            else:
                edge_colours.append('blue')
        node_sizes = [30 if data['source'] == 'base_network' else 10 for _, data in graph.nodes(data=True)]
        edge_widths = [2 if data['source'] == 'base_network' else 1 for _, _, data in graph.edges(data=True)]
        pos = nx.get_node_attributes(graph, "pos")
        pos_labels = {node: (x + offset, y + offset) for node, (x, y) in pos.items()}
        nx.draw(graph, pos, node_color=node_colours, edge_color=edge_colours, node_size=node_sizes, width=edge_widths)
        plt.plot([self.dimensions[0][0], self.dimensions[0][0]], [self.dimensions[0][1], self.dimensions[1][1]], "--", color="gray")  # left line
        plt.plot([self.dimensions[0][0], self.dimensions[1][0]], [self.dimensions[0][1], self.dimensions[0][1]], "--", color="gray")  # bottom line
        plt.plot([self.dimensions[1][0], self.dimensions[1][0]], [self.dimensions[0][1], self.dimensions[1][1]], "--", color="gray")  # right line
        plt.plot([self.dimensions[0][0], self.dimensions[1][0]], [self.dimensions[1][1], self.dimensions[1][1]], "--", color="gray")  # top line

        if base_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={node.id: node.id for node in self.base_network.nodes},
                                    font_size=7, font_color="gray")
            nx.draw_networkx_labels(graph, pos_labels, labels={node.id: node.id for node in self.base_network.nodes},
                                    font_size=7, font_color="gray")
        if ring_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={node.id + id_shift: node.id for node in self.ring_network.nodes},
                                    font_size=7, font_color="purple")
        plt.gca().set_aspect('equal', adjustable='box')

    def draw_graph_pretty(self, title: str = "BSS Network", window_title: str = "BSS Network Viewer",
                          draw_dimensions: bool = False, threshold_size: int = 10) -> None:
        if not self.ring_network.nodes:
            print("No nodes to draw.")
            return

        plt.axis("off")
        ax = plt.gca()
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(window_title)
        fig.suptitle(title)
        patches = []
        colours = []
        lines = []
        white_patches = []
        red_patches = []
        for ring_node in self.ring_network.nodes:
            pbc_coords = np.array([ring_node.coord - pbc_vector(base_node.coord, ring_node.coord, self.dimensions) for base_node in ring_node.ring_neighbours])
            polygon = Polygon(pbc_coords, closed=True)
            if ring_node.id in self.fixed_rings:
                # Fixed rings are coloured red
                red_patches.append(polygon)
            elif len(pbc_coords) >= threshold_size:
                # Rings that are larger than the given threshhold are coloured white
                white_patches.append(polygon)
            else:
                patches.append(polygon)
                colours.append(len(pbc_coords))
            for i in range(len(pbc_coords)):
                line = Line2D([pbc_coords[i, 0], pbc_coords[(i + 1) % len(pbc_coords), 0]],
                              [pbc_coords[i, 1], pbc_coords[(i + 1) % len(pbc_coords), 1]], color="black")
                lines.append(line)

        # Create a PatchCollection from the list of patches
        patch_collection = PatchCollection(patches, cmap='cividis', alpha=0.4)

        # Set the colours of the patches based on the list of colours
        patch_collection.set_array(np.array(colours))
        ax.add_collection(patch_collection)

        # Add the red patches to the plot
        for red_patch in red_patches:
            ax.add_patch(Polygon(red_patch.get_xy(), closed=True, color='red'))

        # Add the white patches to the plot
        for white_patch in white_patches:
            ax.add_patch(Polygon(white_patch.get_xy(), closed=True, color='white'))
        # Add the lines to the plot
        for line in lines:
            ax.add_line(line)
            if draw_dimensions:
                plt.plot([self.dimensions[0][0], self.dimensions[0][0]], [self.dimensions[0][1], self.dimensions[1][1]], "--", color="gray")  # left line
                plt.plot([self.dimensions[0][0], self.dimensions[1][0]], [self.dimensions[0][1], self.dimensions[0][1]], "--", color="gray")  # bottom line
                plt.plot([self.dimensions[1][0], self.dimensions[1][0]], [self.dimensions[0][1], self.dimensions[1][1]], "--", color="gray")  # right line
                plt.plot([self.dimensions[0][0], self.dimensions[1][0]], [self.dimensions[1][1], self.dimensions[1][1]], "--", color="gray")  # top line
            ax.set_aspect('equal', adjustable='box')
            ax.autoscale_view()

        colorbar_axes = inset_axes(ax,
                                   width="100%",  # width = 100% of parent_bbox width
                                   height="5%",  # height : 5%
                                   loc='lower center',
                                   bbox_to_anchor=(0.0, -0.05, 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0)

        # Create the colorbar in the new axes.
        colour_bar = plt.colorbar(patch_collection, cax=colorbar_axes, orientation='horizontal', pad=0.2)
        colour_bar.locator = MaxNLocator(integer=True)
        colour_bar.update_ticks()
        # Set the label for the colorbar.
        label = plt.gcf().text(0.515, 0.12, "Ring Size", ha='center')

        def on_resize(event: ResizeEvent):
            # Get the height of the axes in pixels.
            axes_height = colorbar_axes.get_window_extent().height
            # Calculate the new font size (you may need to adjust the scaling factor).
            new_font_size = axes_height / 2.5
            # Update the title font size.
            label.set_fontsize(new_font_size)
            label.set_position((0.515, 0.12))

        def on_key_press(event: KeyEvent):
            if event.key in ["q", "escape"]:
                plt.close()

        # Connect the resize event to the on_resize function.
        plt.gcf().canvas.mpl_connect('resize_event', on_resize)
        plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
        plt.gcf().canvas.mpl_connect('resize_event', on_resize)

        # Trigger the resize event to set the initial font size.
        plt.gcf().canvas.draw()

    def get_ring_size_limits(self) -> tuple[int, int]:
        max_ring_size = max([len(ring_node.ring_neighbours) for ring_node in self.ring_network.nodes])
        min_ring_size = min([len(ring_node.ring_neighbours) for ring_node in self.ring_network.nodes])
        return min_ring_size, max_ring_size

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

    def __deepcopy__(self: BSSData, memo) -> BSSData:
        copied_base_nodes: list[BSSNode] = [BSSNode(coord=np.copy(node.coord), id=id, type="base") for id, node in enumerate(self.base_network.nodes)]
        copied_ring_nodes: list[BSSNode] = [BSSNode(coord=np.copy(node.coord), id=id, type="ring") for id, node in enumerate(self.ring_network.nodes)]
        for node in self.base_network.nodes:
            for neighbour in node.neighbours:
                copied_base_nodes[node.id].add_neighbour(copied_base_nodes[neighbour.id], self.dimensions)
            for ring_neighbour in node.ring_neighbours:
                copied_base_nodes[node.id].add_ring_neighbour(copied_ring_nodes[ring_neighbour.id], self.dimensions)
        for node in self.ring_network.nodes:
            for neighbour in node.neighbours:
                copied_ring_nodes[node.id].add_neighbour(copied_ring_nodes[neighbour.id], self.dimensions)
            for ring_neighbour in node.ring_neighbours:
                copied_ring_nodes[node.id].add_ring_neighbour(copied_base_nodes[ring_neighbour.id], self.dimensions)
        return BSSData(BSSNetwork(copied_base_nodes, "base", np.copy(self.dimensions)),
                       BSSNetwork(copied_ring_nodes, "ring", np.copy(self.dimensions)),
                       np.copy(self.dimensions), copy.copy(self.fixed_rings))

    def __repr__(self) -> str:
        string = f"BSSData with dimensions:\nxlo: {self.xlo}\txhi: {self.xhi}\tylo: {self.ylo}\tyhi: {self.yhi}\n"
        string += f"Base network:\n{self.base_network}\nRing network:\n{self.ring_network}"
        return string
