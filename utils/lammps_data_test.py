from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional, Iterator

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from .netmc_data import NetMCNetwork

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


def get_pbc_vector(atom_1: LAMMPSAtom, atom_2: LAMMPSAtom, dimensions: np.ndarray) -> np.ndarray:
    """Returns the vector difference between two atoms, taking into account periodic boundary conditions."""
    pbc_vector = np.subtract(atom_2.coord, atom_1.coord)
    dimension_ranges = dimensions[1] - dimensions[0]
    pbc_vector = (pbc_vector + dimension_ranges / 2) % dimension_ranges - dimension_ranges / 2
    return pbc_vector


@dataclass
class LAMMPSData:
    atoms: list[LAMMPSAtom] = field(default_factory=list)
    masses: Optional[dict[str, float]] = None

    def __post_init__(self):
        if self.num_atoms != 0:
            self.refresh_dimensions()
        else:
            self.dimensions = np.array([[0, 0], [0, 0]])

    def get_pbc_vector(self, atom_1: LAMMPSAtom, atom_2: LAMMPSAtom) -> np.ndarray:
        return get_pbc_vector(atom_1, atom_2, self.dimensions)

    def refresh_dimensions(self, add_coord: Optional[np.ndarray] = None, del_coord: Optional[np.ndarray] = None) -> None:
        if add_coord is not None:
            self.dimensions[0] = np.minimum(self.dimensions[0], add_coord)
            self.dimensions[1] = np.maximum(self.dimensions[1], add_coord)
        elif del_coord is not None:
            # Only update if the deleted coordinate was at the boundary
            if np.any(self.dimensions[0] == del_coord):
                self.dimensions[0] = np.min([atom.coord for atom in self.atoms], axis=0)
            if np.any(self.dimensions[1] == del_coord):
                self.dimensions[1] = np.max([atom.coord for atom in self.atoms], axis=0)
        else:
            self.dimensions = np.array([np.min([atom.coord for atom in self.atoms], axis=0), np.max([atom.coord for atom in self.atoms], axis=0)])

    def make_3d(self) -> None:
        if self.num_dimensions == 3:
            raise ValueError("Data is already 3D.")
        self.dimensions = np.column_stack((self.dimensions, np.zeros(self.dimensions.shape[0])))
        for atom in self.atoms:
            atom.make_3d()

    @property
    def xlo(self) -> float:
        return self.dimensions[0][0]

    @property
    def xhi(self) -> float:
        return self.dimensions[1][0]

    @property
    def ylo(self) -> float:
        return self.dimensions[0][1]

    @property
    def yhi(self) -> float:
        return self.dimensions[1][1]

    @property
    def zlo(self) -> float:
        if self.num_dimensions == 2:
            raise ValueError("Data is 2D.")
        return self.dimensions[0][2]

    @property
    def zhi(self) -> float:
        if self.num_dimensions == 2:
            raise ValueError("Data is 2D.")
        return self.dimensions[1][2]

    @property
    def xlo_xhi(self) -> np.ndarray:
        return np.array([self.xlo, self.xhi])

    @property
    def ylo_yhi(self) -> np.ndarray:
        return np.array([self.ylo, self.yhi])

    @property
    def zlo_zhi(self) -> np.ndarray:
        return np.array([self.zlo, self.zhi])

    def export(self, output_path: Path, description: str = LAMMPS_DEFAULT_DESCRIPTION) -> None:
        """Exports the data to a LAMMPS data file."""
        bonds = self.get_bonds()
        angles = self.get_angles()
        self.num_bonds = len(bonds)
        self.num_angles = len(angles)
        with open(output_path, 'w') as data_file:
            data_file.write(f"{description}\n\n")
            # Write parameters that are not defaults
            for param, default_value in LAMMPS_PARAMETER_DEFAULTS.items():
                variable_name = PARAMETER_TO_VARIABLE_MAPPING[param]
                value = getattr(self, variable_name, None)
                if value is not None and value != default_value:
                    if isinstance(value, np.ndarray):
                        value = " ".join(str(element) for element in value)
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
            write_section("Bonds", [bond.export_array() for bond in bonds], data_file)
            write_section("Angles", [angle.export_array() for angle in angles], data_file)

    def check(self) -> None:
        valid = True
        for atom in self.atoms:
            if atom.num_dimensions != self.num_dimensions:
                print(f"Atom {atom.id} has {atom.num_dimensions} dimensions, but the data has {self.num_dimensions}.")
                valid = False
            for neighbour in atom.neighbours:
                if atom not in neighbour.neighbours:
                    print(f"Atom {atom.id} is a neighbour of atom {neighbour.id}, but atom {neighbour.id} is not a neighbour of atom {atom.id}.")
                    valid = False
            for atom in self.atoms:
                if not np.isfinite(atom.coord).all():
                    print(f"Atom {atom.id} ({atom.label}) has non-finite position: {atom.coord}")
                    valid = False
        if not valid:
            raise ValueError("Data is not valid.")

    @staticmethod
    def from_netmc_network(netmc_network: NetMCNetwork, atom_label: str, atomic_mass: Optional[float] = None,
                           dimensions: Optional[np.ndarray] = None, atom_style: str = "atomic") -> LAMMPSData:
        # NetMC data contains only one atom type, which is not knowable from the files
        # We can also not deduce the desired dimensions, atomic masses or bond/angle types
        data = LAMMPSData()
        if atomic_mass is not None:
            data.add_mass(atom_label, atomic_mass)
        for i, node in enumerate(netmc_network.nodes):
            if atom_style == "atomic":
                data.add_atom(LAMMPSAtom(coord=node.coord, label=atom_label))
            elif atom_style == "molecular":
                # Since we cannot deduce what molecule an atom belongs to from NetMC data, we set molecule_id to be the same as id
                data.add_atom(LAMMPSMolecule(coord=node.coord, label=atom_label))
        data.atoms = sorted(data.atoms, key=lambda atom: atom.id)
        for node in netmc_network.nodes:
            for neighbour in node.neighbours:
                data.add_bond(data.atoms[node.id], data.atoms[neighbour.id])
        return data

    def add_mass(self, label: str, mass: float) -> None:
        if self.masses is None:
            self.masses = {}
        self.masses[label] = mass

    def scale_coords(self, scale_factor: float) -> None:
        for atom in self.atoms:
            atom.coord *= scale_factor

    def export_bonds_pairs(self, path: Path) -> None:
        with open(path, 'w') as bond_file:
            for bond in self.bonds:
                bond_file.write(f"{bond.atom_1.id}    {bond.atom_2.id}\n")

    def add_atom(self, atom: LAMMPSAtom) -> None:
        if atom.num_dimensions != self.num_dimensions:
            raise ValueError(f"Atom {atom.id} has {atom.num_dimensions} dimensions, but the data has {self.num_dimensions}.")
        self.atoms.append(atom)
        for neighbour in atom.neighbours:
            neighbour.add_neighbour(atom)
        atom.id = self.num_atoms
        if atom.style == "molecular":
            atom.molecule_id = atom.id
        self.refresh_dimensions(add_coord=atom.coord)

    def delete_atom(self, atom: Optional[LAMMPSAtom] = None) -> None:
        for neighbour in atom.neighbours:
            neighbour.delete_neighbour(atom)
        self.atoms.remove(atom)
        for atom in self.atoms:
            atom.id = self.atoms.index(atom) + 1
        self.refresh_dimensions(del_coord=atom.coord)

    def add_bond(self, atom_1: LAMMPSAtom, atom_2: LAMMPSAtom) -> None:
        atom_1.add_neighbour(atom_2)
        atom_2.add_neighbour(atom_1)

    def delete_bond(self, atom_1, atom_2) -> None:
        atom_1.delete_neighbour(atom_2)
        atom_2.delete_neighbour(atom_1)

    def bond_atoms_within_distance(self, distance: float) -> list[tuple[int, int]]:
        """
        Forms bonds between all atoms within a certain distance from one another
        """
        coords = np.array([atom.coord for atom in self.atoms])
        tree = KDTree(coords)
        bond_pairs = tree.query_pairs(distance)
        for bond in bond_pairs:
            # No need to add 1 to the indices since atom indexes are 0-based even though atom ids are 1-based
            atom_1 = self.atoms[bond[0]]
            atom_2 = self.atoms[bond[1]]
            self.add_bond(atom_1, atom_2)

    def draw_graph(self, atom_labels_to_plot: list[str], bond_labels_to_plot: list[str],
                   atom_colours: dict, bond_colours: dict, atom_sizes: dict,
                   atom_labels: bool = False, bond_labels: bool = False, offset: float = 0.01) -> None:
        bonds = self.get_bonds()
        for atom_label in atom_colours.keys():
            if atom_label not in atom_labels_to_plot:
                raise ValueError(f"Atom {atom_label} is not in the data.")
        for atom_label in atom_sizes.keys():
            if atom_label not in atom_labels_to_plot:
                raise ValueError(f"Atom {atom_label} is not in the data.")
        for bond_label in bond_colours.keys():
            if bond_label not in bond_labels_to_plot:
                raise ValueError(f"Bond {bond_label} is not in the data.")
        graph = nx.Graph()
        for atom_label in atom_labels_to_plot:
            for atom in [atom for atom in self.atoms if atom.label == atom_label]:
                graph.add_node(atom.id, pos=atom.coord)
        for bond_label in bond_labels_to_plot:
            graph.add_edges_from([(bond.atom_1.id, bond.atom_2.id) for bond in bonds if bond.label == bond_label and bond.pbc_length(self.dimensions) > 10 * bond.length])
        node_colors = [atom_colours[atom.label] for atom in self.atoms]
        edge_colors = [bond_colours[bond.label] for bond in bonds]
        node_sizes = [atom_sizes[atom.label] for atom in self.atoms]
        pos = nx.get_node_attributes(graph, "pos")
        if self.num_dimensions == 2:
            pos_labels = {node: (x + offset, y + offset) for node, (x, y) in pos.items()}
        elif self.num_dimensions == 3:
            pos_labels = {node: (x + offset, y + offset, z + offset) for node, (x, y, z) in pos.items()}
        nx.draw(graph, pos, node_color=node_colors, edge_color=edge_colors, node_size=node_sizes)
        if atom_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={atom.id: atom.id for atom in self.atoms}, font_size=7, font_color="gray")
        if bond_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={bond.id: bond.id for bond in bonds}, font_size=7, font_color="gray")

    def _generate_labels(self, items) -> dict[int, str]:
        labels = {}
        seen = set()
        for item in items:
            if item.label not in seen:
                seen.add(item.label)
                labels[len(seen)] = item.label
        return labels

    def get_angles(self) -> list[LAMMPSAngle]:
        return [angle for atom in self.atoms for angle in atom.get_angles()]

    def get_bonds(self) -> list[LAMMPSBond]:
        bonds = []
        for atom in self.atoms:
            for neighbour in atom.neighbours:
                if atom.id < neighbour.id:
                    bonds.append(LAMMPSBond(atom_1=atom, atom_2=neighbour))
        return bonds

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        return len(self.get_bonds)

    @property
    def num_angles(self) -> int:
        return len(self.get_angles)

    @property
    def atom_labels(self) -> dict[int, str]:
        return self._generate_labels(self.atoms)

    @property
    def bond_labels(self) -> dict[int, str]:
        return self._generate_labels(self.bonds)

    @property
    def angle_labels(self) -> dict[int, str]:
        return self._generate_labels(self.angles)

    @property
    def num_atom_labels(self) -> int:
        return len(self.atom_labels)

    @property
    def num_bond_labels(self) -> int:
        return len(self.bond_labels)

    @property
    def num_angle_labels(self) -> int:
        return len(self.angle_labels)

    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions[0])


@dataclass
class LAMMPSBond():
    atom_1: LAMMPSAtom
    atom_2: LAMMPSAtom

    def __post_init__(self):
        # Sort the atoms in the bond label so that we don't duplicate equivalent bond labels
        self.label = '-'.join(sorted([self.atom_1.label, self.atom_2.label]))

    def export_array(self) -> list[int, str, int, int]:
        return [self.id, self.label, self.atom_1.id, self.atom_2.id]

    @property
    def length(self) -> float:
        return np.linalg.norm(self.atom_1.coord - self.atom_2.coord)

    def pbc_length(self, dimensions: np.ndarray) -> float:
        return np.linalg.norm(get_pbc_vector(self.atom_1, self.atom_2, dimensions))

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSBond):
            return (self.atom_1 == other.atom_1 and
                    self.atom_2 == other.atom_2)
        return False


class LAMMPSAngle():
    atom_1: LAMMPSAtom
    atom_2: LAMMPSAtom
    atom_3: LAMMPSAtom

    def __post_init__(self):
        self.label = f"{self.atom_1.label}-{self.atom_2.label}-{self.atom_3.label}"

    def export_array(self) -> list[int, str, int, int, int]:
        return [self.id, self.label, self.atom_1.id, self.atom_2.id, self.atom_3.id]

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSAngle):
            return ((self.atom_1 == other.atom_1 and
                    self.atom_2 == other.atom_2 and
                    self.atom_3 == other.atom_3) or
                    (self.atom_1 == other.atom_3 and
                    self.atom_2 == other.atom_2 and
                    self.atom_3 == other.atom_1))
        return False


@dataclass
class LAMMPSAtom:
    coord: np.ndarray
    label: str
    neighbours: list[LAMMPSAtom] = field(default_factory=list)
    id: Optional[int] = None

    def __post_init__(self):
        self.style = "atomic"

    def get_bonds(self) -> list[LAMMPSBond]:
        bonds = []
        for neighbour in self.neighbours:
            bonds.append(LAMMPSBond(self, neighbour))
        return bonds

    def get_angles(self) -> list[LAMMPSAngle]:
        angles = []
        for i in range(len(self.neighbours)):
            node_1 = self.neighbours[i]
            node_2 = self.neighbours[(i + 1) % len(self.neighbours)]
            angles.append(LAMMPSAngle(node_1, self, node_2))
        return angles

    def add_neighbour(self, neighbour: LAMMPSAtom) -> None:
        self.neighbours.append(neighbour)

    def delete_neighbour(self, neighbour: LAMMPSAtom) -> None:
        self.neighbours.remove(neighbour)

    def translate(self, vector: np.array) -> None:
        self.coord += vector

    def make_3d(self) -> None:
        if self.num_dimensions == 3:
            raise ValueError("Atom is already 3D.")
        self.coord = np.append(self.coord, 0)

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    @property
    def z(self) -> float:
        try:
            return self.coord[2]
        except IndexError:
            raise ValueError("Atom is not 3D.")

    @property
    def num_dimensions(self) -> int:
        return len(self.coord)

    @property
    def export_array(self) -> list[int, str, float, float, float]:
        return [self.id, self.label, *self.coord]

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSAtom):
            return (self.id == other.id and
                    self.label == other.label and
                    np.array_equal(self.coord, other.coord) and
                    self.neighbours == other.neighbours)
        return False

    def __repr__(self) -> str:
        return f"Atom {self.id} ({self.label}) at {self.coord} with neighbours: {[neighbour.id for neighbour in self.neighbours]}"


@dataclass
class LAMMPSMolecule(LAMMPSAtom):
    molecule_id: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.style = "molecular"

    def export_array(self) -> list[int, int, str, float, float, float]:
        return [self.id, self.molecule_id, self.label, *self.coord]

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSMolecule):
            return (self.id == other.id and
                    self.molecule_id == other.molecule_id and
                    self.label == other.label and
                    np.array_equal(self.coord, other.coord) and
                    self.neighbours == other.neighbours)
        return False
