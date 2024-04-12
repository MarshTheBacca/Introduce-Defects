from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from .bss_data import BSSNetwork

LAMMPS_DEFAULT_DESCRIPTION = "Written by lammps_data.py made by Marshall Hunt"

LAMMPS_PARAMETER_DEFAULTS = {"atoms": 0, "bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0, "atom types": 0,
                             "bond types": 0, "angle types": 0, "dihedral types": 0, "improper types": 0,
                             "extra bond per atom": 0, "extra angle per atom": 0, "extra dihedral per atom": 0, "extra improper per atom": 0, "extra special per atom": 0,
                             "ellipsoids": 0, "lines": 0, "triangles": 0, "bodies": 0,
                             "xlo xhi": np.array([-0.5, 0.5]), "ylo yhi": np.array([-0.5, 0.5]), "zlo zhi": np.array([-0.5, 0.5]), "xy xz yz": np.array([0, 0, 0])}

PARAMETER_TO_VARIABLE_MAPPING = {"atoms": "num_atoms", "bonds": "num_bonds", "angles": "num_angles",
                                 "dihedrals": "num_dihedrals", "impropers": "num_impropers",
                                 "atom types": "num_atom_labels", "bond types": "num_bond_labels", "angle types": "num_angle_labels",
                                 "dihedral types": "num_dihedral_labels", "improper types": "num_improper_labels",
                                 "extra bond per atom": "extra_bond_per_atom", "extra angle per atom": "extra_angle_per_atom",
                                 "extra dihedral per atom": "extra_dihedral_per_atom", "extra improper per atom": "extra_improper_per_atom",
                                 "extra special per atom": "extra_special_per_atom",
                                 "ellipsoids": "ellipsoids", "lines": "lines", "triangles": "triangles", "bodies": "bodies",
                                 "xlo xhi": "xlo_xhi", "ylo yhi": "ylo_yhi", "zlo zhi": "zlo_zhi", "xy xz yz": "xy_xz_yz"}


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


def remove_comment(line: str) -> str:
    """Removes comments from a line of a string."""
    return line.split("#")[0].strip()


def get_lines_between_headers(lines: list[str], start_header: str, headers: tuple(str)) -> list[str]:
    header_to_index = {line: lines.index(line) for line in lines if line in headers}
    header_indexes = sorted(header_to_index.values())
    start_index = header_to_index[start_header] + 1
    header_index = header_indexes.index(header_to_index[start_header])
    if header_index == len(header_indexes) - 1:
        end_index = len(lines)
    else:
        end_index = header_indexes[header_index + 1] - 1
    return lines[start_index:end_index]


@dataclass
class LAMMPSData:
    atoms: list[LAMMPSAtom] = field(default_factory=list)
    dimensions: np.ndarray = field(default_factory=lambda: np.array([[0, 0], [1, 1]]))
    masses: dict[str, float] = field(default_factory=lambda: dict)

    def __post_init__(self) -> None:
        self.num_atoms = len(self.atoms)
        self.atom_labels = self._generate_labels([atom.label for atom in self.atoms])
        self.num_atom_labels = len(self.atom_labels)

        self.bonds: list[LAMMPSBond] = []
        for atom in self.atoms:
            for neighbour in atom.neighbours:
                if atom.id < neighbour.id:
                    self.bonds.append(LAMMPSBond(atom_1=atom, atom_2=neighbour))
        self.num_bonds = len(self.bonds)
        self.bond_labels = self._generate_labels([bond.label for bond in self.bonds])
        self.num_bond_labels = len(self.bond_labels)

        self.angles: list[LAMMPSAngle] = []
        for atom in self.atoms:
            self.angles.extend(atom.get_angles())
        self.num_angles = len(self.angles)
        self.angle_labels = self._generate_labels([angle.label for angle in self.angles])
        self.num_angle_labels = len(self.angle_labels)

    def _generate_labels(self, labels: list[str]) -> dict[str, int]:
        label_dict = {}
        for label in labels:
            if label not in label_dict:
                label_dict[label] = 1
                continue
            label_dict[label] += 1
        return label_dict

    def export(self, output_path: Path, description: str = LAMMPS_DEFAULT_DESCRIPTION) -> None:
        """Exports the data to a LAMMPS data file."""
        with open(output_path, 'w') as data_file:
            data_file.write(f"{description}\n\n")
            # Write parameters that are not defaults
            for param, default_value in LAMMPS_PARAMETER_DEFAULTS.items():
                variable_name = PARAMETER_TO_VARIABLE_MAPPING[param]
                if variable_name == "zlo_zhi" and self.num_dimensions == 2:
                    continue
                value = getattr(self, variable_name, None)
                if value is not None:
                    if isinstance(value, np.ndarray):
                        if not np.array_equal(value, default_value):
                            value = " ".join(str(element) for element in value)
                            data_file.write(f"{value} {param}\n")
                    elif value != default_value:
                        data_file.write(f"{value} {param}\n")

            # Write labels so you can use them later in the datafile
            write_section("Atom Type Labels", [[i + 1, label] for i, label in enumerate(self.atom_labels)], data_file)
            write_section("Bond Type Labels", [[i + 1, label] for i, label in enumerate(self.bond_labels)], data_file)
            write_section("Angle Type Labels", [[i + 1, label] for i, label in enumerate(self.angle_labels)], data_file)
            # Write masses
            if self.masses:
                write_section("Masses", dict_to_2d_array(self.masses), data_file)
            # Write atoms, bonds and angles
            write_section(f"Atoms # {self.atoms[0].style}", [atom.export_array for atom in self.atoms], data_file)
            write_section("Bonds", [bond.export_array for bond in self.bonds], data_file)
            write_section("Angles", [angle.export_array for angle in self.angles], data_file)

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
    def from_data_file(path: Path) -> LAMMPSData:
        HEADER_NAMES = ("Atoms", "Bonds", "Angles", "Atom Type Lables", "Bond Type Labels", "Angle Type Labels", "Masses")
        with open(path, "r") as data_file:
            raw_lines = data_file.read().split("\n")
        lines = []
        for line in raw_lines:
            stripped_line = remove_comment(line)
            if stripped_line:
                lines.append(stripped_line)
        parm_lines = lines[:min(header_to_index.values())]
        atom_lines = get_lines_between_headers(lines, "Atoms", HEADER_NAMES)
        bond_lines = get_lines_between_headers(lines, "Bonds", HEADER_NAMES)

    @staticmethod
    def from_bss_network(bss_network: BSSNetwork, atom_label: str, atomic_mass: Optional[float] = None,
                         atom_style: str = "atomic") -> LAMMPSData:
        # BSS data contains only one atom type, which is not knowable from the files
        # We can also not deduce the desired dimensions, atomic masses or bond/angle types
        data = LAMMPSData([], bss_network.dimensions, {})
        if atomic_mass is not None:
            data.add_mass(atom_label, atomic_mass)
        for node in bss_network.nodes:
            if atom_style == "atomic":
                data.add_atom(LAMMPSAtom(coord=node.coord, label=atom_label))
            elif atom_style == "molecular":
                # Since we cannot deduce what molecule an atom belongs to from BSS data, we set molecule_id to be the same as id
                data.add_atom(LAMMPSMolecule(coord=node.coord, label=atom_label))
            else:
                raise ValueError(f"Atom style {atom_style} is not supported.")
        for node in bss_network.nodes:
            for neighbour in node.neighbours:
                if node.id < neighbour.id:
                    data.add_bond(data.atoms[node.id], data.atoms[neighbour.id])
        return data

    def add_mass(self, label: str, mass: float) -> None:
        self.masses[label] = mass

    def add_atom(self, atom: LAMMPSAtom) -> None:
        if atom.num_dimensions != self.num_dimensions:
            raise ValueError(f"Atom {atom.id} has {atom.num_dimensions} dimensions, but the data has {self.num_dimensions}.")
        self.atoms.append(atom)
        self.num_atoms += 1
        for neighbour in atom.neighbours:
            neighbour.add_neighbour(atom)
        atom.id = self.num_atoms
        if atom.style == "molecular":
            atom.molecule_id = self.num_atoms
        if atom.label not in self.atom_labels:
            self.atom_labels[atom.label] = 1
            self.num_atom_labels += 1
        else:
            self.atom_labels[atom.label] += 1

    def delete_atom(self, atom: LAMMPSAtom) -> None:
        for neighbour in atom.neighbours:
            neighbour.delete_neighbour(atom)
        for bond in self.bonds:
            if bond.atom_1 == atom or bond.atom_2 == atom:
                self.bonds.remove(bond)
                self.num_bonds -= 1
                if self.bond_labels[bond.label] == 1:
                    self.bond_labels.pop(bond.label)
                    self.num_bond_labels -= 1
                else:
                    self.bond_labels[bond.label] -= 1
        for angle in self.angles:
            if angle.atom_1 == atom or angle.atom_2 == atom or angle.atom_3 == atom:
                self.angles.remove(angle)
                self.num_angles -= 1
                if self.angle_labels[angle.label] == 1:
                    self.angle_labels.pop(angle.label)
                    self.num_angle_labels -= 1
                else:
                    self.angle_labels[angle.label] -= 1
        self.atoms.remove(atom)
        self.num_atoms -= 1
        if self.atom_labels[atom.label] == 1:
            self.atom_labels.pop(atom.label)
            self.num_atom_labels -= 1
        for i, atom in enumerate(self.atoms, start=1):
            atom.id = i
        for i, bond in enumerate(self.bonds, start=1):
            bond.id = i
        for i, angle in enumerate(self.angles, start=1):
            angle.id = i

    def add_bond(self, atom_1: LAMMPSAtom, atom_2: LAMMPSAtom) -> None:
        old_angles = atom_1.get_angles() + atom_2.get_angles()
        atom_1.add_neighbour(atom_2)
        atom_2.add_neighbour(atom_1)
        new_bond = LAMMPSBond(atom_1=atom_1, atom_2=atom_2)
        self.bonds.append(new_bond)
        self.num_bonds += 1
        new_bond.id = self.num_bonds
        if new_bond.label not in self.bond_labels:
            self.bond_labels[new_bond.label] = 1
            self.num_bond_labels += 1
        else:
            self.bond_labels[new_bond.label] += 1
        new_angles = atom_1.get_angles() + atom_2.get_angles()
        self._handle_new_angles(old_angles, new_angles)

    def delete_bond(self, atom_1: LAMMPSAtom, atom_2: LAMMPSAtom) -> None:
        old_angles = atom_1.get_angles() + atom_2.get_angles()
        atom_1.delete_neighbour(atom_2)
        atom_2.delete_neighbour(atom_1)
        bond_to_remove = LAMMPSBond(atom_1=atom_1, atom_2=atom_2)
        self.bonds.remove(bond_to_remove)
        self.num_bonds -= 1
        for bond in self.bonds:
            bond.id = self.bonds.index(bond) + 1
        if self.bond_labels[bond_to_remove.label] == 1:
            self.bond_labels.pop(bond_to_remove.label)
            self.num_bond_labels -= 1
        new_angles = atom_1.get_angles() + atom_2.get_angles()
        self._handle_new_angles(old_angles, new_angles)

    def _handle_new_angles(self, old_angles: list[LAMMPSAngle], new_angles: list[LAMMPSAngle]) -> None:
        for angle in old_angles:
            self.angles.remove(angle)
            self.num_angles -= 1
            if self.angle_labels[angle.label] == 1:
                self.angle_labels.pop(angle.label)
                self.num_angle_labels -= 1
            else:
                self.angle_labels[angle.label] -= 1
        for angle in new_angles:
            self.angles.append(angle)
            self.num_angles += 1
            if angle.label not in self.angle_labels:
                self.angle_labels[angle.label] = 1
                self.num_angle_labels += 1
            else:
                self.angle_labels[angle.label] += 1
        for i, angle in enumerate(self.angles, start=1):
            angle.id = i

    def export_bonds_pairs(self, path: Path) -> None:
        with open(path, 'w') as bond_file:
            for bond in self.bonds:
                bond_file.write(f"{bond.atom_1.id}    {bond.atom_2.id}\n")

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

    def scale_coords(self, scale_factor: float) -> None:
        for atom in self.atoms:
            atom.coord *= scale_factor
        self.dimensions *= scale_factor

    def draw_graph(self, atom_labels_to_plot: list[str], bond_labels_to_plot: list[str],
                   atom_colours: dict, bond_colours: dict, atom_sizes: dict,
                   atom_labels: bool = False, bond_labels: bool = False, offset: float = 0.01) -> None:
        for atom_label in atom_colours:
            if atom_label not in atom_labels_to_plot:
                raise ValueError(f"Atom {atom_label} is not in the data.")
        for atom_label in atom_sizes:
            if atom_label not in atom_labels_to_plot:
                raise ValueError(f"Atom {atom_label} is not in the data.")
        for bond_label in bond_colours:
            if bond_label not in bond_labels_to_plot:
                raise ValueError(f"Bond {bond_label} is not in the data.")
        graph = nx.Graph()
        for atom_label in atom_labels_to_plot:
            for atom in [atom for atom in self.atoms if atom.label == atom_label]:
                graph.add_node(atom.id, pos=atom.coord)
        for bond_label in bond_labels_to_plot:
            for bond in [bond for bond in self.bonds if bond.label == bond_label]:
                if not bond.is_pbc_bond(self.dimensions):
                    graph.add_edge(bond.atom_1.id, bond.atom_2.id)
        node_colours = [atom_colours[atom.label] for atom in self.atoms]
        edge_colours = [bond_colours[bond.label] for bond in self.bonds]
        node_sizes = [atom_sizes[atom.label] for atom in self.atoms]
        pos = nx.get_node_attributes(graph, "pos")
        if self.num_dimensions == 2:
            pos_labels = {node: (x + offset, y + offset) for node, (x, y) in pos.items()}
        elif self.num_dimensions == 3:
            pos_labels = {node: (x + offset, y + offset, z + offset) for node, (x, y, z) in pos.items()}
        nx.draw(graph, pos, node_color=node_colours, edge_color=edge_colours, node_size=node_sizes)
        if atom_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={atom.id: atom.id for atom in self.atoms}, font_size=7, font_color="gray")
        if bond_labels:
            nx.draw_networkx_labels(graph, pos_labels, labels={bond.id: bond.id for bond in self.bonds}, font_size=7, font_color="gray")

    def make_3d(self, zlo: float, zhi: float) -> None:
        """
        Adds a z dimension to the data, all atoms have a z coordinate of 0.
        """
        if self.num_dimensions == 3:
            raise ValueError("Data is already 3D.")
        self.dimensions = np.column_stack((self.dimensions, np.array([zlo, zhi]))).T
        for atom in self.atoms:
            atom.make_3d()

    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions[0])

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
            raise ValueError("Data is 2D. Cannot get zlo.")
        return self.dimensions[0][2]

    @property
    def zhi(self) -> float:
        if self.num_dimensions == 2:
            raise ValueError("Data is 2D. Cannot get zhi.")
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

    def __repr__(self) -> str:
        string = f"LAMMPSData with {self.num_atoms} atoms, {self.num_bonds} bonds and {self.num_angles} angles.\n"
        for label in self.atom_labels:
            string += f"{label}: {self.atom_labels[label]}\t"
        string += f"\nxlo xhi: {self.xlo} {self.xhi}\tylo yhi: {self.ylo} {self.yhi}"
        if self.num_dimensions == 3:
            string += f"\tzlo zhi: {self.zlo} {self.zhi}"
        for atom in self.atoms:
            string += f"\n{atom}"
        return string


@dataclass
class LAMMPSBond():
    atom_1: LAMMPSAtom
    atom_2: LAMMPSAtom

    def __post_init__(self):
        # Sort the atoms in the bond label so that we don't duplicate equivalent bond labels
        self.label = '-'.join(sorted([self.atom_1.label, self.atom_2.label]))
        self.length = np.linalg.norm(self.atom_1.coord - self.atom_2.coord)

    @property
    def export_array(self) -> list[int, str, int, int]:
        return [self.id, self.label, self.atom_1.id, self.atom_2.id]

    def pbc_length(self, dimensions: np.ndarray) -> float:
        return np.linalg.norm(pbc_vector(self.atom_1.coord, self.atom_2.coord, dimensions))

    def is_pbc_bond(self, dimensions: np.ndarray) -> bool:
        """
        Identifies if the bond crosses the periodic boundary. So if the length of the bond is 
        more than 10% longer than the distance between the two nodes with periodic boundary conditions,
        then it is considered a periodic bond.
        """
        if self.length > self.pbc_length(dimensions) * 1.1:
            return True
        return False

    def get_pbc_vector(self, dimensions: np.ndarray) -> np.ndarray:
        return pbc_vector(self.atom_1.coord, self.atom_2.coord, dimensions)

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSBond):
            return (self.atom_1 == other.atom_1 and
                    self.atom_2 == other.atom_2)
        return False

    def __repr__(self) -> str:
        return f"Bond {self.id} {self.label} between atoms {self.atom_1.id} and {self.atom_2.id}"


@dataclass
class LAMMPSAngle():
    atom_1: LAMMPSAtom
    atom_2: LAMMPSAtom
    atom_3: LAMMPSAtom
    id: Optional[int] = None

    def __post_init__(self):
        self.label = f"{self.atom_1.label}-{self.atom_2.label}-{self.atom_3.label}"

    @property
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

    def __repr__(self) -> str:
        return f"Angle {self.id} {self.label} between atoms {self.atom_1.id}, {self.atom_2.id}, {self.atom_3.id}"


@dataclass
class LAMMPSAtom:
    coord: np.ndarray
    label: str
    neighbours: list[LAMMPSAtom] = field(default_factory=list)
    id: Optional[int] = None

    def __post_init__(self):
        self.style = "atomic"
        self.num_dimensions = len(self.coord)

    def get_angles(self) -> list[LAMMPSAngle]:
        angles = []
        for i in range(len(self.neighbours)):
            node_1 = self.neighbours[i]
            node_2 = self.neighbours[(i + 1) % len(self.neighbours)]
            angles.append(LAMMPSAngle(node_1, self, node_2))
        return angles

    def sort_nodes_clockwise(self, nodes: list[LAMMPSAtom]) -> None:
        """
        Sorts the given nodes in clockwise order.
        """
        def angle_to_node(node):
            """
            Returns the angle between the x-axis and the vector from the node
            to the given node in radians, range of -pi to pi.
            """
            dx = node.x - self.x
            dy = node.y - self.y
            return np.arctan2(dy, dx)

        nodes.sort(key=angle_to_node)

    def add_neighbour(self, neighbour: LAMMPSAtom) -> None:
        self.neighbours.append(neighbour)
        self.sort_nodes_clockwise(self.neighbours)

    def delete_neighbour(self, neighbour: LAMMPSAtom) -> None:
        self.neighbours.remove(neighbour)

    def translate(self, vector: np.array) -> None:
        self.coord += vector

    def make_3d(self) -> None:
        if self.num_dimensions == 3:
            raise ValueError("Atom is already 3D.")
        self.coord = np.append(self.coord, 0)
        self.num_dimensions = 3

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
    def export_array(self) -> list[int, str, float, float, float]:
        if self.num_dimensions == 2:
            return [self.id, self.label, *self.coord, 0]
        return [self.id, self.label, *self.coord]

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSAtom):
            return (self.id == other.id and
                    self.label == other.label and
                    np.array_equal(self.coord, other.coord) and
                    [neighbour.id for neighbour in self.neighbours] == [neighbour.id for neighbour in other.neighbours])
        return False

    def __repr__(self) -> str:
        string = f"LAMMPSAtom {self.id} {self.label} at {self.coord} with neighbours: "
        for neighbour in self.neighbours:
            string += f"{neighbour.id}, "
        return string


@dataclass
class LAMMPSMolecule(LAMMPSAtom):
    molecule_id: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.style = "molecular"

    @property
    def export_array(self) -> list[int, int, str, float, float, float]:
        """
        Returns the array to be written to the data file.
        """
        if self.num_dimensions == 2:
            return [self.id, self.molecule_id, self.label, *self.coord, 0]
        return [self.id, self.molecule_id, self.label, *self.coord]

    def __eq__(self, other) -> bool:
        if isinstance(other, LAMMPSMolecule):
            return (self.id == other.id and
                    self.molecule_id == other.molecule_id and
                    self.label == other.label and
                    np.array_equal(self.coord, other.coord) and
                    [neighbour.id for neighbour in self.neighbours] == [neighbour.id for neighbour in other.neighbours])
        return False

    def __repr__(self) -> str:
        string = f"LAMMPSMolecule id: {self.id} mol id: {self.molecule_id} {self.label} at {self.coord} with neighbours: "
        for neighbour in self.neighbours:
            string += f"{neighbour.id}, "
        return string
