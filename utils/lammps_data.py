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


@dataclass
class LAMMPSData:
    atoms: list[LAMMPSAtom] = field(default_factory=list)
    bonds: list[LAMMPSBond] = field(default_factory=list)
    angles: list[LAMMPSAngle] = field(default_factory=list)
    dimensions: Optional[np.ndarray] = None
    masses: Optional[dict[str, float]] = None

    def __post_init__(self):
        self.num_atoms = len(self.atoms)
        self.num_bonds = len(self.bonds)
        self.num_angles = len(self.angles)
        if self.dimensions is None:
            self.dimensions = self.get_dimensions()
        self.atom_labels = {i + 1: label for i, label in enumerate(set(atom.label for atom in self.atoms))}
        self.bond_labels = {i + 1: label for i, label in enumerate(set(bond.label for bond in self.bonds))}
        self.angle_labels = {i + 1: label for i, label in enumerate(set(angle.label for angle in self.angles))}
        self.num_atom_labels = len(self.atom_labels)
        self.num_bond_labels = len(self.bond_labels)
        self.num_angle_labels = len(self.angle_labels)

    @property
    def xlo_xhi(self) -> np.ndarray:
        return self.dimensions[0]

    @property
    def ylo_yhi(self) -> np.ndarray:
        return self.dimensions[1]

    @property
    def zlo_zhi(self) -> np.ndarray:
        if len(self.dimensions) == 3:
            return self.dimensions[2]
        return np.array([0, 0])

    def export(self, output_path: Path, description: str = LAMMPS_DEFAULT_DESCRIPTION) -> None:
        """Exports the data to a LAMMPS data file."""
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
            write_section("Bonds", [bond.export_array() for bond in self.bonds], data_file)
            write_section("Angles", [angle.export_array() for angle in self.angles], data_file)

    def check(self):
        """Checks the data for consistency."""
        print("Checking data...")
        valid = True
        if not self.check_atoms():
            print("Atoms invalid")
            valid = False
        if not self.check_bonds():
            print("Bonds invalid")
            valid = False
        if not self.check_angles():
            print("Angles invalid")
            valid = False
        most_common_coord_dim = collections.Counter(atom.coord_dim for atom in self.atoms).most_common(1)[0][0]
        if most_common_coord_dim != len(self.dimensions):
            print(f"Dimensions {self.dimensions} do not match the most common coordinate dimension {most_common_coord_dim}.")
            valid = False
        if not valid:
            raise ValueError("Data is invalid.")

    def check_atoms(self) -> bool:
        """Returns True if all atoms have the same number of dimensions and style."""
        print("Checking atoms...")
        valid = True
        print(self.atoms)
        for atom in self.atoms:
            print(atom.coord_dim, atom.style)
        most_common_coord_dim = collections.Counter(atom.coord_dim for atom in self.atoms).most_common(1)[0][0]
        most_common_style = collections.Counter(atom.style for atom in self.atoms).most_common(1)[0][0]
        for atom in self.atoms:
            if atom.coord_dim != most_common_coord_dim:
                print(f"Atom {atom.id} has {atom.coord_dim} dimensions, but the most common is {most_common_coord_dim}.")
                valid = False
            if atom.label != most_common_style:
                print(f"Atom {atom.id} has style {atom.style}, but the most common is {most_common_style}.")
                valid = False
            if atom.label not in self.atom_labels.values():
                print(f"Atom {atom.id} has label {atom.label} which is not in the list of labels: {self.atom_labels.values()}")
                valid = False
            if out_of_bounds(atom.coord, self.dimensions):
                print(f"Atom {atom.id} {atom.label} is out of bounds.")
                valid = False
        return valid

    def check_bonds(self) -> bool:
        """Returns True if all bonds are unique, their atom pairs are in the data, and their labels are in the list of bond labels."""
        print("Checking bonds...")
        valid = True
        for i, bond in enumerate(self.bonds):
            if bond.label not in self.bond_labels.values():
                print(f"Bond {bond.id} {bond.label} is not in the list of labels: {self.bond_labels.values()}")
                valid = False
            for atom in bond.atoms:
                if atom not in self.atoms:
                    print(f"Atom {atom.id} {atom.label} in bond {bond.id} {bond.label} is not in the data.")
                    valid = False
            # The use of [i + 1:] means that we only check each bond once, since we have already checked the other bonds before it
            for other_bond in self.bonds[i + 1:]:
                if set(bond.atoms) == set(other_bond.atoms):
                    print(f"Bond {bond.id} {bond.label} is equivalent to bond {other_bond.id} {other_bond.label}.")
                    valid = False
        return valid

    def check_angles(self) -> bool:
        print("Checking angles...")
        valid = True
        for i, angle in enumerate(self.angles):
            if angle.label not in self.angle_labels.values():
                print(f"Angle {angle.id} {angle.label} is not in the list of labels: {self.angle_labels.values()}")
                valid = False
            for atom in angle.atoms:
                if atom not in self.atoms:
                    print(f"Atom {atom.id} {atom.label} in angle {angle.id} {angle.label} is not in the data.")
                    valid = False
            for other_angle in self.angles[i + 1:]:
                if angle.atoms[1] == other_angle.atoms[1] and set(angle.atoms[::2]) == set(other_angle.atoms[::2]):
                    print(f"Angle {angle.id} {angle.label} is equivalent to angle {other_angle.id} {other_angle.label}.")
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
    def from_netmc_network(netmc_network: NetMCNetwork, atom_label: str, atomic_mass: Optional[float] = None,
                           dimensions: Optional[np.ndarray] = None, atom_style: str = "atomic") -> LAMMPSData:
        # NetMC data contains only one atom type, which is not knowable from the files
        # We can also not deduce the desired dimensions, atomic masses or bond/angle types
        if dimensions is None:
            dimensions = np.array([[netmc_network.xlo, netmc_network.xhi],
                                   [netmc_network.ylo, netmc_network.yhi]])
        data = LAMMPSData(atoms=[], bonds=[], angles=[], dimensions=dimensions)
        if atomic_mass is not None:
            data.add_mass(atom_label, atomic_mass)
        for i, node in enumerate(netmc_network.nodes):
            if atom_style == "atomic":
                data.add_atom(LAMMPSAtom(coord=node.coord, label=atom_label))
            elif atom_style == "molecular":
                # Since we cannot deduce what molecule an atom belongs to from NetMC data, we set molecule_id to be the same as id
                data.add_atom(LAMMPSMolecule(coord=node.coord, label=atom_label, molecule_id=i + 1))
        data.atoms = sorted(data.atoms, key=lambda atom: atom.id)
        for node in netmc_network.nodes:
            for neighbour in node.neighbours:
                data.add_structure(LAMMPSBond(atoms=[data.atoms[node.id], data.atoms[neighbour.id]],
                                              label=f"{atom_label}-{atom_label}"))
            for angle in get_angles(node.id, [neighbour.id for neighbour in node.neighbours]):
                data.add_structure(LAMMPSAngle(atoms=[data.atoms[angle[0]], data.atoms[angle[1]], data.atoms[angle[2]]],
                                               label=f"{atom_label}-{atom_label}-{atom_label}"))

        return data

    def add_atom(self, atom: LAMMPSAtom, expand_dims=True) -> None:
        if out_of_bounds(atom.coord, self.dimensions):
            if not expand_dims:
                raise ValueError(f"Atom {atom.id} {atom.label} is out of bounds. Use expand_dims=True to expand the dimensions.")
            else:
                self.dimensions = expand_dimensions(atom.coord, self.dimensions)
        if atom.label not in self.atom_labels.values():
            self.add_atom_label(atom.label)
        atom.id = self.num_atoms + 1
        if atom.style == "molecular":
            atom.molecule_id = atom.id
        self.atoms.append(atom)
        self.num_atoms += 1

    def add_label(self, label: str, label_dict: dict, num_labels: int) -> tuple[dict, int]:
        if label_dict is None:
            label_dict = {}
        label_dict[len(label_dict) + 1] = label
        num_labels += 1
        return label_dict, num_labels

    def add_atom_label(self, label: str) -> None:
        self.atom_labels, self.num_atom_labels = self.add_label(label, self.atom_labels, self.num_atom_labels)

    def add_bond_label(self, label: str) -> None:
        self.bond_labels, self.num_bond_labels = self.add_label(label, self.bond_labels, self.num_bond_labels)

    def add_angle_label(self, label: str) -> None:
        self.angle_labels, self.num_angle_labels = self.add_label(label, self.angle_labels, self.num_angle_labels)

    def add_mass(self, label: str, mass: float) -> None:
        if self.masses is None:
            self.masses = {}
        self.masses[label] = mass

    def scale_coords(self, scale_factor: float) -> None:
        for atom in self.atoms:
            atom.coord *= scale_factor
        self.dimensions *= scale_factor

    def get_pbc_vector(self, atom_1: LAMMPSAtom, atom_2: LAMMPSAtom) -> np.ndarray:
        """Returns the vector difference between two atoms, taking into account periodic boundary conditions."""
        if len(atom_1.coord) != len(atom_2.coord) or len(atom_1.coord) != len(self.dimensions):
            raise ValueError("Atoms must have the same number of dimensions.")
        pbc_vector = np.subtract(atom_2.coord, atom_1.coord)
        dimension_ranges = self.dimensions[:, 1] - self.dimensions[:, 0]
        pbc_vector = (pbc_vector + dimension_ranges / 2) % dimension_ranges - dimension_ranges / 2
        return pbc_vector

    def get_atom_connections(self, atom: LAMMPSAtom) -> Iterator[LAMMPSAtom]:
        """Returns a list of atoms bonded to the given atom."""
        for bond in self.bonds:
            if bond.atoms[0] == atom:
                yield bond.atoms[1]
            elif bond.atoms[1] == atom:
                yield bond.atoms[0]

    def export_bonds(self, path: Path) -> None:
        with open(path, 'w') as bond_file:
            for bond in self.bonds:
                bond_file.write(f"{bond.atoms[0].id} {bond.atoms[1].id}\n")

    def make_3d(self) -> None:
        if len(self.dimensions) == 3:
            return
        self.dimensions = np.array([self.dimensions[0], self.dimensions[1], np.array([0, 0])])
        for atom in self.atoms:
            atom.coord = np.append(atom.coord, 0)

    def remove_atom(self, atom: Optional[LAMMPSAtom] = None, all=False) -> None:
        """Removes an atom and all structures containing it."""
        if atom is None:
            if all:
                self.atoms = []
            else:
                raise ValueError("No atom was specified.")
        else:
            self.atoms.remove(atom)
            for bond in self.bonds:
                if atom in bond.atoms:
                    self.remove_bond(bond)
            for angle in self.angles:
                if atom in angle.atoms:
                    self.remove_angle(angle)

    def remove_bond(self, bond: Optional[LAMMPSBond] = None, all: bool = False) -> None:
        """Removes a bond."""
        if bond is None:
            if all:
                self.bonds = []
            else:
                raise ValueError("No bond was specified.")
        else:
            self.bonds.remove(bond)

    def remove_angle(self, angle: Optional[LAMMPSAngle] = None, all: bool = False) -> None:
        """Removes an angle."""
        if angle is None:
            if all:
                self.angles = []
            else:
                raise ValueError("No angle was specified.")
        else:
            self.angles.remove(angle)

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
            self.add_structure(LAMMPSBond(atom_1, atom_2, label=f"{atom_1.label}-{atom_2.label}"))

    def get_coords(self, atom_label: str) -> np.ndarray:
        return np.array([atom.coord for atom in self.atoms if atom.label == atom_label])

    def draw_graph(self, atom_labels_to_plot: list[str], bond_labels_to_plot: list[str],
                   atom_colours: dict, bond_colours: dict, atom_sizes: dict, bond_widths: dict,
                   atom_labels: bool = False, bond_labels: bool = False, offset: float = 0.01) -> None:
        for atom in atom_colours.keys():
            if atom not in atom_labels_to_plot:
                raise ValueError(f"Atom {atom} is not in the data.")
        for atom in atom_sizes.keys():
            if atom not in atom_labels_to_plot:
                raise ValueError(f"Atom {atom} is not in the data.")
        for bond in bond_colours.keys():
            if bond not in bond_labels_to_plot:
                raise ValueError(f"Bond {bond} is not in the data.")
        for bond in bond_widths.keys():
            if bond not in bond_labels_to_plot:
                raise ValueError(f"Bond {bond} is not in the data.")

        graph = nx.Graph()
        for atom in atom_labels_to_plot:
            graph.add_nodes_from([atom.id for atom in self.atoms if atom.label == atom])
        for bond in bond_labels_to_plot:
            graph.add_edges_from([(bond.atoms[0].id, bond.atoms[1].id) for bond in self.bonds if bond.label == bond])
        node_colors = [atom_colours[atom.label] for atom in self.atoms]
        edge_colors = [bond_colours[bond.label] for bond in self.bonds]
        node_sizes = [atom_sizes[atom.label] for atom in self.atoms]
        edge_widths = [bond_widths[bond.label] for bond in self.bonds]
        pos = nx.get_node_attributes(graph, "pos")
        pos_labels = {node: (x + offset, y + offset) for node, (x, y) in pos.items()}
        nx.draw(graph, pos, node_color=node_colors, edge_color=edge_colors, node_size=node_sizes, width=edge_widths)
        if atom_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={atom.id: atom.id for atom in self.atoms}, font_size=7, font_color="gray")
        if bond_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={bond.id: bond.id for bond in self.bonds}, font_size=7, font_color="gray")

    def add_structure(self, structure: LAMMPSStructure) -> None:
        for atom in structure.atoms:
            if atom not in self.atoms:
                raise ValueError(f"Atom {atom.id} {atom.label} is not in the data.")
        structure.add_to_data(self)


@dataclass
class LAMMPSStructure(ABC):
    atoms: list[LAMMPSAtom]
    label: str
    id: Optional[int] = None

    @abstractmethod
    def add_to_data(self, data: LAMMPSData) -> None:
        pass

    def export_array(self) -> list:
        return [self.id, self.label, *[atom.id for atom in self.atoms]]


class LAMMPSBond(LAMMPSStructure):
    def __post_init__(self):
        # Sort the atoms in the bond label so that we don't duplicate equivalent bond labels
        self.label = '-'.join(sorted(self.label.split('-')))

    def add_to_data(self, data: LAMMPSData) -> None:
        """
        Adds the bond to the data if it is not already in the data.
        """
        for bond in data.bonds:
            if self.atoms[0] and self.atoms[1] in bond.atoms:
                return
        self.id = data.num_bonds + 1
        data.bonds.append(self)
        data.num_bonds += 1
        if self.label not in data.bond_labels.values():
            data.bond_labels[len(data.bond_labels) + 1] = self.label
            data.num_bond_labels += 1


class LAMMPSAngle(LAMMPSStructure):
    def add_to_data(self, data: LAMMPSData) -> None:
        for angle in data.angles:
            if self.atoms[0] == angle.atoms[0] and self.atoms[1] == angle.atoms[1] and self.atoms[2] == angle.atoms[2]:
                return
            if self.atoms[0] == angle.atoms[2] and self.atoms[1] == angle.atoms[1] and self.atoms[2] == angle.atoms[0]:
                return
        self.id = data.num_angles + 1
        data.angles.append(self)
        data.num_angles += 1
        if self.label not in data.angle_labels.values():
            data.angle_labels[len(data.angle_labels) + 1] = self.label
            data.num_angle_labels += 1


@dataclass
class LAMMPSAtom:
    coord: np.ndarray
    label: str
    id: Optional[int] = None

    def __post_init__(self):
        self.coord_dim = len(self.coord)
        if self.coord_dim not in (2, 3):
            raise ValueError(f"Atom {self.id} {self.label} has {self.coord_dim} dimensions, but only 2 or 3 are supported.")

    @property
    def style(self) -> str:
        return "atomic"

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    @property
    def z(self) -> float:
        if len(self.coord) == 2:
            raise ValueError("Atom does not have a z coordinate.")
        return self.coord[2]

    def set_x(self, value: float) -> None:
        self.coord[0] = value

    def set_y(self, value: float) -> None:
        self.coord[1] = value

    def set_z(self, value: float) -> None:
        if len(self.coord) == 2:
            raise ValueError("Atom does not have a z coordinate.")
        self.coord[2] = value

    def export_array(self) -> list:
        return [self.id, self.label, *self.coord]

    def __eq__(self, other) -> bool:
        return (self.id == other.id and
                self.label == other.label and
                np.array_equal(self.coord, other.coord))


@dataclass
class LAMMPSMolecule(LAMMPSAtom):
    molecule_id: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()

    @property
    def style(self) -> str:
        return "molecular"

    def export_array(self) -> list:
        return [self.id, self.molecule_id, self.label, *self.coord]
