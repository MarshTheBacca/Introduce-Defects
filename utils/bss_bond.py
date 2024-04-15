from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .bss_node import BSSNode
from .other_utils import pbc_vector


@dataclass
class BSSBond:
    node_1: BSSNode
    node_2: BSSNode

    def __post_init__(self) -> None:
        self.type = f"{self.node_1.type}-{self.node_2.type}"

    @property
    def length(self) -> float:
        return np.linalg.norm(self.node_1.coord - self.node_2.coord)

    def pbc_length(self, dimensions: np.ndarray) -> float:
        return np.linalg.norm(pbc_vector(self.node_1.coord, self.node_2.coord, dimensions))

    def check(self) -> bool:
        if self.node_1 == self.node_2:
            print(f"Bond ({self.type} between node_1: {self.node_1.id} node_2: {self.node_2.id} bonds identical nodes")
            return False
        return True

    def __eq__(self, other) -> bool:
        if isinstance(other, BSSBond):
            return ((self.node_1 == other.node_1 and self.node_2 == other.node_2) or
                    (self.node_1 == other.node_2 and self.node_2 == other.node_1))
        return False

    def __repr__(self) -> str:
        return f"Bond of type {self.type} between {self.node_1.id} and {self.node_2.id}"
