from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from .other_utils import pbc_vector


def angle_to_node(node_1: BSSNode, node_2: BSSNode, dimensions: np.ndarray) -> float:
    """
    Returns the angle between the x-axis and the vector from the node
    to the given node in radians, range of -pi to pi.
    """
    periodic_vector = pbc_vector(node_1.coord, node_2.coord, dimensions)
    return -np.arctan2(periodic_vector[1], periodic_vector[0])


@dataclass
class BSSNode:
    coord: np.ndarray
    type: str
    neighbours: list[BSSNode] = field(default_factory=lambda: [])
    ring_neighbours: list[BSSNode] = field(default_factory=lambda: [])
    id: Optional[int] = None

    def check(self, dimensions: np.ndarray) -> bool:
        valid = True
        if self.type not in ("base", "ring"):
            print(f"Node {self.id} has invalid type {self.type}")
            valid = False
        for neighbour in self.neighbours:
            if neighbour == self:
                print(f"Node {self.id} {self.type} has itself as neighbour")
                valid = False
            if self not in neighbour.neighbours:
                print(f"Node {self.id} {self.type} has neighbour {neighbour.id}, but neighbour does not have node as neighbour")
                valid = False
            if self.type != neighbour.type:
                print(f"Node {self.id} {self.type} has neighbour {neighbour.id}, but neighbour has different type")
                valid = False
        for ring_neighbour in self.ring_neighbours:
            if self not in ring_neighbour.ring_neighbours:
                print(f"Node {self.id} {self.type} has ring neighbour {ring_neighbour.id}, but ring neighbour does not have node as ring neighbour")
                valid = False
            if self.type == ring_neighbour.type:
                print(f"Node {self.id} {self.type} has ring neighbour {ring_neighbour.id}, but ring neighbour has same type")
                valid = False
        for neighbour_list in [self.neighbours, self.ring_neighbours]:
            sorted_neighbours = sorted(neighbour_list, key=lambda node: angle_to_node(self, node, dimensions))
            if neighbour_list != sorted_neighbours:
                print(f"Node {self.id} {self.type} neighbours are not in clockwise order")
                valid = False
        return valid

    def sort_neighbours_clockwise(self, nodes: list[BSSNode], dimensions: np.ndarray) -> None:
        """
        Sorts the given neighbours in clockwise order.
        """
        nodes.sort(key=lambda node: angle_to_node(self, node, dimensions))

    def sort_neighbours_clockwise(self, nodes: list[BSSNode], dimensions: np.ndarray) -> None:
        """
        Sorts the given neighbours in clockwise order.
        """
        nodes.sort(key=lambda node: angle_to_node(self, node, dimensions))

    def get_ring_walk(self) -> list[BSSNode]:
        """
        Returns a list of nodes such that the order is how they are connected in the ring.
        """
        def depth_first_search(node: BSSNode, walk: list[BSSNode]) -> Optional[list[BSSNode]]:
            walk.append(node)
            if len(walk) == len(self.ring_neighbours):
                return walk
            for neighbour in node.neighbours:
                if neighbour in self.ring_neighbours and neighbour not in walk:
                    result = depth_first_search(neighbour, walk)
                    if result is not None:
                        return result
            walk.pop()  # backtrack
            return None

        walk = depth_first_search(self.ring_neighbours[0], [])
        if walk is None:
            raise ValueError(f"Could not find ring walk for node {self.id} ({self.type}) ring_neighbours: {[ring_neighbour.id for ring_neighbour in self.ring_neighbours]}")
        return walk

    def get_angles(self) -> Iterator[tuple[BSSNode, BSSNode, BSSNode]]:
        for i in range(len(self.neighbours)):
            node_1 = self.neighbours[i]
            node_2 = self.neighbours[(i + 1) % len(self.neighbours)]
            yield (node_1, self, node_2)

    def add_neighbour(self, neighbour: BSSNode, dimensions: np.ndarray) -> None:
        self.neighbours.append(neighbour)
        self.sort_neighbours_clockwise(self.neighbours, dimensions)

    def delete_neighbour(self, neighbour: BSSNode) -> None:
        self.neighbours.remove(neighbour)

    def add_ring_neighbour(self, neighbour: BSSNode, dimensions: np.ndarray) -> None:
        self.ring_neighbours.append(neighbour)
        self.sort_neighbours_clockwise(self.ring_neighbours, dimensions)

    def delete_ring_neighbour(self, neighbour: BSSNode) -> None:
        self.ring_neighbours.remove(neighbour)

    def translate(self, vector: np.ndarray) -> None:
        self.coord += vector

    def scale(self, scale_factor: float) -> None:
        self.coord *= scale_factor

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
        if isinstance(other, BSSNode):
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
