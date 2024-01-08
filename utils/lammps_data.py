from __future__ import annotations
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import collections
from typing import Optional, IO
from .netmc_data import NetMCData, NetworkType
from abc import ABC, abstractmethod


LAMMPS_DEFAULT_DESCRIPTION = "Written by lammps_data.py made by Marshall Hunt"

LAMMPS_PARAMETER_DEFAULTS = {"atoms": 0, "bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0, "atom types": 0,
                             "bond types": 0, "angle types": 0, "dihedral types": 0, "improper types": 0,
                             "extra bond per atom": 0, "extra angle per atom": 0, "extra dihedral per atom": 0, "extra improper per atom": 0, "extra special per atom": 0,
                             "ellipsoids": 0, "lines": 0, "triangles": 0, "bodies": 0,
                             "xlo xhi": np.array([-0.5, 0.5]), "ylo yhi": np.array([-0.5, 0.5]), "zlo zhi": np.array([-0.5, 0.5]), "xy xz yz": np.array([0, 0, 0])}

PARAMETER_TO_VARIABLE_MAPPING = {"atoms": "num_atoms", "bonds": "num_bonds", "angles": "num_angles",
                                 "dihedrals": "num_dihedrals", "impropers": "num_impropers",
                                 "atom types": "num_atom_labels", "bond types": "num_bond_labels", "angle types": "num_angle_labels",
                                 "xlo xhi": "dimensions", "ylo yhi": "dimensions", "zlo zhi": "dimensions"}


def get_angles(selected_node: int, bonded_nodes: np.ndarray) -> list[tuple[int, int, int]]:
    """Returns a list of angles in the form (bonded_node_1, selected_node, bonded_node_2)"""
    angles, num_bonded_nodes = [], len(bonded_nodes)
    if num_bonded_nodes < 2:
        return []
    for i in range(num_bonded_nodes):
        bonded_node_1 = bonded_nodes[i]
        bonded_node_2 = bonded_nodes[(i + 1) % num_bonded_nodes]
        angles.append((bonded_node_1, selected_node, bonded_node_2))
    return angles


def dict_to_2d_array(dictionary: dict) -> np.ndarray:
    """Converts a dictionary to a 2D array with the keys in the first column and the values in the second column."""
    return np.array([[key, value] for key, value in dictionary.items()])


def write_section(header: str, data: list[list] | np.ndarray, data_file: IO) -> None:
    """Writes a section of the datafile with a header and the data."""
    data_file.write(f"\n{header}\n\n")
    for row in data:
        for element in row:
            data_file.write(f"{element:<20} ")
        data_file.write("\n")


def out_of_bounds(coords: np.ndarray, dimensions: np.ndarray) -> bool:
    """Returns True if the coordinates are out of bounds."""
    if len(coords) != len(dimensions):
        raise ValueError(f"Coordinates {coords} and dimensions {dimensions} have different lengths.")
    for coord, dimension in zip(coords, dimensions):
        if coord < dimension[0] or coord > dimension[1]:
            return True
    return False


def expand_dims(coords: np.ndarray, dimensions: np.ndarray, tolerance: float = 0.5) -> np.ndarray:
    """Expands the dimensions to include the coordinates."""
    if len(coords) != len(dimensions):
        raise ValueError(f"Coordinates {coords} and dimensions {dimensions} have different lengths.")
    for coord, dimension in zip(coords, dimensions):
        if coord < dimension[0]:
            dimension[0] = coord - tolerance
        elif coord > dimension[1]:
            dimension[1] = coord + tolerance
    return dimensions


def get_dims(coords: np.ndarray, tolerance: float = 0) -> np.ndarray:
    """Returns the dimensions of the coordinates."""
    return np.array([[min(coords[:, 0]) + tolerance, max(coords[:, 0]) + tolerance],
                     [min(coords[:, 1]) + tolerance, max(coords[:, 1]) + tolerance]])


@dataclass
class LAMMPSData:
    atoms: list[LAMMPSAtom]
    bonds: list[LAMMPSBond]
    angles: list[LAMMPSAngle]
    dimensions: Optional[np.ndarray] = None
    masses: Optional[dict[str, float]] = None

    def __post_init__(self):
        if not self.check_atoms():
            print("Multiple atom types and mixtures of 2D and 3D coordinates are not supported")
            raise ValueError("Atom are inconsistent.")
        self.num_atoms = len(self.atoms)
        self.num_bonds = len(self.bonds)
        self.num_angles = len(self.angles)
        if self.dimensions is None:
            self.dimensions = self.get_dimensions()
        self.xlo_xhi = self.dimensions[0]
        self.ylo_yhi = self.dimensions[1]
        self.zlo_zhi = self.dimensions[2] if len(self.dimensions) == 3 else np.array([0, 0])
        self.atom_labels = {i + 1: label for i, label in enumerate(set(atom.label for atom in self.atoms))}
        self.bond_labels = {i + 1: label for i, label in enumerate(set(bond.label for bond in self.bonds))}
        self.angle_labels = {i + 1: label for i, label in enumerate(set(angle.label for angle in self.angles))}
        self.num_atom_labels = len(self.atom_labels)
        self.num_bond_labels = len(self.bond_labels)
        self.num_angle_labels = len(self.angle_labels)

    def export(self, output_path: Path, description: str = LAMMPS_DEFAULT_DESCRIPTION) -> None:
        """Exports the data to a LAMMPS data file."""
        with open(output_path, 'w') as data_file:
            data_file.write(f"{description}\n\n")
            # Write parameters that are not defaults
            for param, default_value in LAMMPS_PARAMETER_DEFAULTS.items():
                variable_name = PARAMETER_TO_VARIABLE_MAPPING[param]
                value = getattr(self, variable_name, None)
                if value is not None and value != default_value:
                    data_file.write(f"{value} {param}\n")
            # Write labels so you can use them later in the datafile
            write_section("Atom Type Labels", dict_to_2d_array(self.atom_labels), data_file)
            write_section("Bond Type Labels", dict_to_2d_array(self.bond_labels), data_file)
            write_section("Angle Type Labels", dict_to_2d_array(self.angle_labels), data_file)
            # Write masses
            if self.masses is not None:
                write_section("Masses", dict_to_2d_array(self.masses), data_file)
            # Write atoms, bonds and angles
            write_section(f"Atoms # {self.atoms[0].style}", [atom.export_array() for atom in self.atoms], data_file)
            write_section("Bonds", [bond.export_array() for bond in self.bonds], data_file)
            write_section("Angles", [angle.export_array() for angle in self.angles], data_file)

    def check_atoms(self) -> bool:
        """Returns True if all atoms have the same number of dimensions and style."""
        valid = True
        most_common_coord_dim = collections.Counter(atom.coord_dim for atom in self.atoms).most_common(1)[0][0]
        most_common_style = collections.Counter(atom.style for atom in self.atoms).most_common(1)[0][0]
        for atom in self.atoms:
            if atom.coord_dim != most_common_coord_dim:
                print(f"Atom {atom.id} has {atom.coord_dim} dimensions, but the most common is {most_common_coord_dim}.")
                valid = False
            if atom.label != most_common_style:
                print(f"Atom {atom.id} has style {atom.label}, but the most common is {most_common_style}.")
                valid = False
        return valid

    def get_dimensions(self, tolerance: float = 0.5) -> np.ndarray:
        dims = []
        for axis in range(len(self.atoms[0].coord)):
            minimum = min(atom.coord[axis] for atom in self.atoms)
            maximum = max(atom.coord[axis] for atom in self.atoms)
            dims.append([minimum + tolerance, maximum + tolerance])
        return np.array(dims)

    @staticmethod
    def from_netmc_data(netmc_data: NetMCData, network_type: NetworkType,
                        atom_label: str, dimensions: np.ndarray | None = None, atom_style: str = "atomic") -> LAMMPSData:
        # NetMC data contains only one atom type, which is not knowable from the files
        # We can also not deduce the desired dimensions, atomic masses or bond/angle types
        if network_type == NetworkType.A:
            crds = netmc_data.crds_a
            nets = netmc_data.net_a.nets
        elif network_type == NetworkType.B:
            crds = netmc_data.crds_b
            nets = netmc_data.net_b.nets
        else:
            raise ValueError(f"Network type {network_type} is not supported.")
        if dimensions is None:
            dimensions = get_dims(crds, tolerance=0.5)
        data = LAMMPSData(atoms=[], bonds=[], angles=[], dimensions=dimensions)
        for i, coord in enumerate(crds):
            if atom_style == "atomic":
                data.add_atom(LAMMPSAtom(id=i + 1, label=atom_label, coord=coord))
            elif atom_style == "molecular":
                # Since we cannot deduce what molecule an atom belongs to from NetMC data, we set molecule_id to be the same as id
                data.add_atom(LAMMPSMolecule(id=i + 1, label=atom_label, molecule_id=i + 1, coord=coord))
        data.atoms = sorted(data.atoms, key=lambda atom: atom.id)
        bond_id, angle_id = 1, 1
        for i, net in enumerate(nets):
            for bonded_node in net:
                data.add_structure(LAMMPSBond(id=bond_id, label=f"{atom_label}-{atom_label}",
                                              atoms=[data.atoms[i], data.atoms[bonded_node]]))
                bond_id += 1
            for angle in get_angles(i, net):
                data.add_structure(LAMMPSAngle(id=angle_id, label=f"{atom_label}-{atom_label}-{atom_label}",
                                               atoms=[data.atoms[angle[0]], data.atoms[angle[1]], data.atoms[angle[2]]]))
                angle_id += 1
        return data

    def add_atom_label(self, label: str) -> None:
        if self.atom_labels is None:
            self.atom_labels = {}
        self.atom_labels[len(self.atom_labels) + 1] = label
        self.num_atom_labels += 1

    def add_bond_label(self, label: str) -> None:
        if self.bond_labels is None:
            self.bond_labels = {}
        self.bond_labels[len(self.bond_labels) + 1] = label
        self.num_bond_labels += 1

    def add_angle_label(self, label: str) -> None:
        if self.angle_labels is None:
            self.angle_labels = {}
        self.angle_labels[len(self.angle_labels) + 1] = label
        self.num_angle_labels += 1

    def add_mass(self, label: str, mass: float) -> None:
        if self.masses is None:
            self.masses = {}
        self.masses[label] = mass

    def add_atom(self, atom: LAMMPSAtom, expand_dims=False) -> None:
        if out_of_bounds(atom.coord, self.dimensions):
            if not expand_dims:
                raise ValueError(f"Atom {atom.id} {atom.label} is out of bounds. Use expand_dims=True to expand the dimensions.")
            else:
                self.dimensions = expand_dims(atom.coord, self.dimensions)
        if atom.label not in self.atom_labels.values():
            self.add_atom_label(atom.label)
        self.atoms.append(atom)
        self.num_atoms += 1

    def add_structure(self, structure):
        for atom in structure.atoms:
            if atom not in self.atoms:
                raise ValueError(f"Atom {atom.id} {atom.label} is not in the data.")
        structure.add_to_data(self)


@dataclass
class LAMMPSStructure(ABC):
    id: int
    atoms: list[LAMMPSAtom]
    label: str

    @abstractmethod
    def add_to_data(self, data):
        pass

    def export_array(self) -> list:
        return [self.id, self.label, *[atom.id for atom in self.atoms]]


class LAMMPSBond(LAMMPSStructure):
    def add_to_data(self, data: LAMMPSData) -> None:
        data.bonds.append(self)
        data.num_bonds += 1
        if self.label not in data.bond_labels.values():
            data.bond_labels[len(data.bond_labels) + 1] = self.label
            data.num_bond_labels += 1


class LAMMPSAngle(LAMMPSStructure):
    def add_to_data(self, data: LAMMPSData) -> None:
        data.angles.append(self)
        data.num_angles += 1
        if self.label not in data.angle_labels.values():
            data.angle_labels[len(data.angle_labels) + 1] = self.label
            data.num_angle_labels += 1


@dataclass
class LAMMPSAtom:
    id: int
    label: str
    coord: np.ndarray

    def __post_init__(self):
        self.coord_dim = len(self.coord)
        if self.coord_dim not in (2, 3):
            raise ValueError(f"Atom {self.id} {self.label} has {self.coord_dim} dimensions, but only 2 or 3 are supported.")

    @property
    def style(self) -> str:
        return "atomic"

    def export_array(self) -> list:
        return [self.id, self.label, *self.coord]


@dataclass
class LAMMPSMolecule(LAMMPSAtom):
    molecule_id: int

    def __post_init__(self):
        super().__post_init__()

    @property
    def style(self) -> str:
        return "molecular"

    def export_array(self) -> list:
        return [self.id, self.molecule_id, self.label, *self.coord]
