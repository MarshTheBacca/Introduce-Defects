from __future__ import annotations
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import collections
from typing import Optional
from .netmc_data import NetMCData, NetworkType

def get_angles(selected_node: int, bonded_nodes: np.ndarray) -> list[tuple[int, int, int]]:
    angles, num_bonded_nodes = [], len(bonded_nodes)
    if num_bonded_nodes < 2:
        return []
    for i in range(num_bonded_nodes):
        bonded_node_1 = bonded_nodes[i]
        bonded_node_2 = bonded_nodes[(i + 1) % num_bonded_nodes]
        angles.append((bonded_node_1, selected_node, bonded_node_2))
    return angles


@dataclass
class LAMMPSData:
    atoms: list[LAMMPSAtom]
    bonds: list[LAMMPSBond]
    angles: list[LAMMPSAngle]
    dimensions: np.ndarray = None
    masses: Optional[dict[str, float]] = None

    def __post_init__(self):
        if not self.check_coords():
            raise ValueError("The coordinates of the atoms are not consistent.")
        self.num_atoms = len(self.atoms)
        self.num_bonds = len(self.bonds)
        self.num_angles = len(self.angles)
        self.atom_labels = set(atom.label for atom in self.atoms)
        self.bond_labels = set(bond.label for bond in self.bonds)
        self.angle_labels = set(angle.label for angle in self.angles)
        self.num_atom_types = len(self.atom_labels)
        self.num_bond_types = len(self.bond_labels)
        self.num_angle_types = len(self.angle_labels)
        if self.dimensions is None:
            self.dimensions = self.get_dimensions()
    
    def check_coords(self) -> bool:
        valid = True
        most_common_coord_dim = collections.Counter(atom.coord_dim for atom in self.atoms).most_common(1)[0][0]
        for atom in self.atoms:
            if atom.coord_dim != most_common_coord_dim:
                print(f"Atom {atom.id} has {atom.coord_dim} dimensions, but {most_common_coord_dim} are expected.")
                valid = False
        return valid

    def get_dimensions(self, tolerance: float = 0.5) -> np.ndarray:
        dims = []
        for axis in self.dimensions:
            minimum = min(atom.coord[axis] for atom in self.atoms)
            maximum = max(atom.coord[axis] for atom in self.atoms)
            dims.append([minimum + tolerance, maximum + tolerance])
        return np.array(dims)

    @staticmethod
    def from_netmc_data(netmc_data: NetMCData, network_type: NetworkType, atom_lable: str, dimensions: np.ndarray = None) -> LAMMPSData:
        # NetMC data contains only one atom type, which is not knowable from the files
        # We can also not deduce the desired dimensions, atomic masses or bond/angle types
        atoms = []
        bonds = []
        angles = []
        if network_type == NetworkType.A:
            node_id = 1
            bond_id = 1
            angle_id = 1
            for coord, net in zip(netmc_data.crds_a, netmc_data.net_a):
                atoms.append(LAMMPSAtom(id=node_id, label=atom_lable, coord=coord))
                for bonded_node in net:
                    bonds.append(LAMMPSBond(id=bond_id, label=f"{atom_lable}-{atom_lable}", atom_ids=(node_id, bonded_node + 1)))
                    bond_id += 1
                for angle in get_angles(node_id, net):
                    angles.append(LAMMPSAngle(id=angle_id, label=f"{atom_lable}-{atom_lable}-{atom_lable}", atom_ids=angle))
                    angle_id += 1
                node_id += 1
            return LAMMPSData(atoms=atoms, bonds=bonds, angles=angles, dimensions=dimensions)


@dataclass
class LAMMPSDataEntry:
    id: int
    label: str

@dataclass
class LAMMPSAtom(LAMMPSDataEntry):
    coord: np.ndarray

    def __post_init__(self):
        self.style = "atomic"
        self.coord_dim = len(self.coord)

@dataclass
class LAMMPSMolecule(LAMMPSAtom):
    molecule_id: int
    def __post_init__(self):
        super().__post_init__()
        self.style = "molecular"


@dataclass
class LAMMPSBond(LAMMPSDataEntry):
    atom_ids: tuple[int, int]

@dataclass
class LAMMPSAngle(LAMMPSDataEntry):
    atom_ids: tuple[int, int, int]
    