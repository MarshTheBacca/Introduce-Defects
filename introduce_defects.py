import shutil
from collections import Counter
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
from scipy.spatial import KDTree

from utils import (CouldNotBondUndercoordinatedNodesException,
                   InvalidNetworkException,
                   InvalidUndercoordinatedNodesException, LAMMPSAngle,
                   LAMMPSAtom, LAMMPSBond, LAMMPSData, NetMCData, NetMCNetwork,
                   NetMCNode)

# The comment he's included in C.data 'Atoms' line is wrong, the atoms are being stored as regular atoms, not molecules
# since there is a missing molecule ID column.

# Comment for Masses in C.data is incorrect, should be 'C' not 'Si'

# For some reason, he's written Si.data, SiO2.data and Si2O3.data atoms as molecule types, but with unique molecule IDs

# In C.data, Si.data, all atoms have a z coordinate of 0
# In SiO2.data, all Si atoms alternate between z = 5 and z = 11.081138669036534
# In SiO2.data, the first 1/3 of O atoms have z = 8.040569334518267, then they alternate between z = 3.9850371001619402, z = 12.096101568874595
# In Si2O3.data, all atoms have z = 5

# Need to have a look at his C++ source code to see how *.data, *.in and *.lammps are being used
# And therefore if any corrections are in order for how he writes these files.
matplotlib.use('TkAgg')

# Constants
ANGSTROM_TO_BOHR = 1 / 0.52917721090380
O_O_DISTANCE_FACTOR = np.sqrt(8 / 3)
SI_O_LENGTH_ANGSTROM = 1.609
H_ANGLE_DEGREES = 19.5

# Scaling factors
SI_O_LENGTH_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM
O_O_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * O_O_DISTANCE_FACTOR
H_BOHR = ANGSTROM_TO_BOHR * np.sin(np.radians(H_ANGLE_DEGREES)) * SI_O_LENGTH_ANGSTROM


def plot_bilayer(lammps_data: LAMMPSData, path: Path, file_extension: str) -> None:
    si_coords = lammps_data.get_coords("Si")
    o_coords = lammps_data.get_coords("O")
    plt.scatter(si_coords[:, 0], si_coords[:, 1], color='y', s=0.4)
    plt.scatter(o_coords[:, 0], o_coords[:, 1], color='r', s=0.4)
    plt.savefig(path.joinpath(f"Bilayer_Atoms.{file_extension}"))

    for bond in lammps_data.bonds:
        # Only need to check for O-Si because bond labels are
        # sorted when the LAMMPSBond object is created in the __post_init__ method
        if bond.label == "O-Si":
            atom_1_crds = bond.atoms[0].coord
            atom_2_crds = bond.atoms[1].coord
            atom_2_crds = np.add(atom_1_crds, pbc_vector(atom_1_crds, atom_2_crds, lammps_data.dimensions))
            plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[1], atom_2_crds[1]], color='k')
    plt.savefig(path.joinpath(f"Bilayer_Si_O_Bonds.{file_extension}"))
    plt.clf()

    plt.scatter(si_coords[:, 0], si_coords[:, 2], color='y', s=0.4)
    plt.scatter(o_coords[:, 0], o_coords[:, 2], color='r', s=0.4)
    for bond in lammps_data.bonds:
        if bond.label == "O-O":
            atom_1_crds = bond.atoms[0].coord
            atom_2_crds = bond.atoms[1].coord
            atom_2_crds = np.add(atom_1_crds, pbc_vector(atom_1_crds, atom_2_crds, lammps_data.dimensions))
            plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[2], atom_2_crds[2]], color='k')
    plt.savefig(path.joinpath(f"Bilayer_O_O_Bonds.{file_extension}"))
    for bond in lammps_data.bonds:
        if bond.label != "O-O":
            atom_1_crds = bond.atoms[0].coord
            atom_2_crds = bond.atoms[1].coord
            atom_2_crds = np.add(atom_1_crds, pbc_vector(atom_1_crds, atom_2_crds, lammps_data.dimensions))
            plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[2], atom_2_crds[2]], color='k')
    plt.savefig(path.joinpath(f"Bilayer_All_Bonds.{file_extension}"))


def pbc_vector(vector1: np.ndarray, vector2: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    Calculate the vector difference between two vectors, taking into account periodic boundary conditions.
    """
    if len(vector1) != len(vector2) or len(vector1) != len(dimensions):
        raise ValueError("Vectors must have the same number of dimensions.")
    difference_vector = np.subtract(vector2, vector1)
    dimension_ranges = dimensions[:, 1] - dimensions[:, 0]
    half_dimension_ranges = dimension_ranges / 2
    difference_vector = (difference_vector + half_dimension_ranges) % dimension_ranges - half_dimension_ranges
    return difference_vector


def netmc_to_triangle_raft(netmc_network: NetMCNetwork) -> LAMMPSData:
    # This will only work for exactly hexagonal networks
    SI_SI_DISTANCE_FACTOR = np.sqrt(32.0 / 9.0)
    SI_SI_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * SI_SI_DISTANCE_FACTOR
    # The last vector was originally [-0.5, -np.sqrt(3) / 3], but this seems wrong vvvvv
    DISPLACEMENT_VECTORS_NORM = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
    DISPLACEMENT_VECTORS_FACTORED = DISPLACEMENT_VECTORS_NORM * 0.5

    triangle_raft_lammps_data = LAMMPSData.from_netmc_data(netmc_network, atom_label="Si", atomic_mass=28.1, atom_style="atomic")
    triangle_raft_lammps_data.scale_coords(SI_SI_DISTANCE_BOHR)
    triangle_raft_lammps_data.add_atom_label("O")
    triangle_raft_lammps_data.add_mass("O", 15.995)

    dimension_ranges = triangle_raft_lammps_data.dimensions[:, 1] - triangle_raft_lammps_data.dimensions[:, 0]
    for si_atom in triangle_raft_lammps_data.atoms:
        bonded_si_atoms = triangle_raft_lammps_data.get_bonded_atoms(si_atom)
        for bonded_si_atom in bonded_si_atoms:
            vector_between_si_atoms = pbc_vector(si_atom.coord, bonded_si_atom.coord, dimension_ranges)
            normalized_vector = vector_between_si_atoms / np.linalg.norm(vector_between_si_atoms)
            dot_product_grades = np.abs(np.dot(normalized_vector, DISPLACEMENT_VECTORS_NORM.T))
            selected_vector_index = np.argmin(dot_product_grades)
            midpoint = (si_atom.coord + vector_between_si_atoms / 2) % dimension_ranges

            if dot_product_grades[selected_vector_index] < 0.1:
                oxygen_coord = midpoint + DISPLACEMENT_VECTORS_FACTORED[selected_vector_index] % dimension_ranges
            else:
                oxygen_coord = midpoint
            triangle_raft_lammps_data.add_atom(LAMMPSAtom("O", oxygen_coord))
            triangle_raft_lammps_data.add_structure(LAMMPSBond(si_atom, triangle_raft_lammps_data.atoms[-1]))
            triangle_raft_lammps_data.add_structure(LAMMPSBond(bonded_si_atom, triangle_raft_lammps_data.atoms[-1]))
            triangle_raft_lammps_data.add_structure(LAMMPSAngle(si_atom, triangle_raft_lammps_data.atoms[-1], bonded_si_atom))
    triangle_raft_lammps_data.check()
    return triangle_raft_lammps_data


# Intercept is defined when initalising DrawLineWidget as 1? No idea why
def write_lammps_files(path: Path, prefix: str, non_defect_netmc_network: NetMCNetwork, triangle_raft: bool, bilayer: bool, common_files_path: Path):
    print("Writing LAMMPS files...")
    netmc_data = NetMCData.from_files(path, prefix=prefix)
    scaling_factor = np.sqrt(netmc_data.base_network.num_nodes / non_defect_netmc_network.num_nodes)
    si_lammps_data = LAMMPSData.from_netmc_data(netmc_data.base_network, atom_label="Si", atomic_mass=29.977, atom_style="atomic")
    si_lammps_data.scale_coords(scaling_factor)
    # Original function wrote dim lows as 0
    si_lammps_data.export(path.joinpath("Si.data"))
    shutil.copyfile(common_files_path.joinpath("Si.in"), path.joinpath("Si.in"))
    shutil.copyfile(common_files_path.joinpath("PARM_Si.lammps"), path.joinpath("PARM_Si.lammps"))

    c_lammps_data = LAMMPSData.from_netmc_data(netmc_data.base_network, atom_label="C", atomic_mass=12.0000, atom_style="atomic")
    c_lammps_data.scale_coords(1.42)
    # Original function wrote dim lows as 0 and did not include bonds or angles
    c_lammps_data.export(path.joinpath("C.data"))
    shutil.copyfile(common_files_path.joinpath("C.in"), path.joinpath("C.in"))
    shutil.copyfile(common_files_path.joinpath("PARM_C.lammps"), path.joinpath("PARM_C.lammps"))

    if triangle_raft:
        print("Writing triangle raft files...")
        # Should we use isotopic masses or abundance based masses?
        # For some reason the original function uses Si = 32.01 and O = 28.1, but I think this is wrong

        # For Si2O3 coords, the first 2/5 of the coords are Si, the last 3/5 are O. All atoms z = 5.0
        # For SiO2 coords, the first 1/3 are Si, the rest are O. Si z alternates between 5 and 11.081138669036534 (5 + 2 * Si-O length)
        # The O coords in SiO2, the first 1/4 are z = 8.040569334518267, last 3/4 alternate between 3.9850371001619402, 12.096101568874595 (8.040569334518267 + 2 * Si-O length)

        shutil.copyfile(common_files_path.joinpath("Si2O3.in"), path.joinpath("Si2O3.in"))

        triangle_raft_lammps_data = netmc_to_triangle_raft(netmc_data.base_network)
        triangle_raft_lammps_data.export(path.joinpath("Si2O3.data"))

        with open(path.joinpath("PARM_Si2O3.lammps"), 'w') as output_file:
            output_file.write(f"pair_style lj/cut {O_O_DISTANCE_BOHR}\n")
            output_file.write(f"pair_coeff * * 0.1 {O_O_DISTANCE_BOHR / 2**(1 / 6)} {O_O_DISTANCE_BOHR}")
            output_file.write("pair_modify shift yes\n")
            output_file.write("special_bonds lj 0.0 1.0 1.0\n")
            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 2.86667626014\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

        triangle_raft_lammps_data.export_bonds(path.joinpath("Si2O3_harmpairs.dat"))

    if bilayer:
        print("Writing bilayer files...")
        shutil.copyfile(common_files_path.joinpath("SiO2.in"), path.joinpath("SiO2.in"))
        triangle_raft_lammps_data = netmc_to_triangle_raft(netmc_data)
        triangle_raft_lammps_data.make_3d()
        # All atoms now have z = 0
        bilayer_lammps_data = LAMMPSData()
        bilayer_lammps_data.add_atom_label("Si")
        bilayer_lammps_data.add_atom_label("O")
        bilayer_lammps_data.add_mass("Si", 28.1)
        bilayer_lammps_data.add_mass("O", 15.995)
        for atom in triangle_raft_lammps_data.atoms:
            if atom.label == "Si":
                bottom_si_atom = LAMMPSAtom(atom.coord + np.array([0, 0, 5]), "Si")
                top_si_atom = LAMMPSAtom(atom.coord + np.array([0, 0, 5 + 2 * SI_O_LENGTH_BOHR]), "Si")
                central_o_atom = LAMMPSAtom(atom.coord + np.array([0, 0, 5 + SI_O_LENGTH_BOHR]), "O")
                bilayer_lammps_data.add_atom(bottom_si_atom)
                bilayer_lammps_data.add_atom(top_si_atom)
                bilayer_lammps_data.add_atom(central_o_atom)
            if atom.label == "O":
                bottom_o_atom = LAMMPSAtom(atom.coord + np.array([0, 0, 5 - H_BOHR]), "O")
                top_o_atom = LAMMPSAtom(atom.coord + np.array([0, 0, 5 + H_BOHR + 2 * SI_O_LENGTH_BOHR]), "O")
                bilayer_lammps_data.add_atom(top_o_atom)
                bilayer_lammps_data.add_atom(bottom_o_atom)

        bilayer_lammps_data.bond_atoms_within_distance(1.1 * SI_O_LENGTH_BOHR)
        bilayer_lammps_data.export(path.joinpath("SiO2.data"))
        bilayer_lammps_data.export_bonds(path.joinpath("SiO2_harmpairs.dat"))

        with open(path.joinpath("PARM_SiO2.lammps"), 'w') as output_file:
            output_file.write('pair_style lj/cut {:}\n'.format(O_O_DISTANCE_BOHR))
            output_file.write(f"pair_coeff * * 0.1 {O_O_DISTANCE_BOHR / 2**(1 / 6)} {O_O_DISTANCE_BOHR}\n")
            output_file.write("pair_modify shift yes\n")
            output_file.write("special_bonds lj 0.0 1.0 1.0\n")
            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 3.0405693345182674\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

    print("Finished writing LAMMPS files.")


def get_target_files(file_names: list, extension: str) -> list:
    return [name for name in file_names if name.endswith(extension)]


def find_prefix(path):
    file_names = [path.name for path in Path.iterdir(path) if path.is_file()]
    EXTENSIONS = ["aux.dat", "crds.dat", "net.dat", "dual.dat"]
    all_prefixes = [name[:-(len(ext) + 3)] for ext in EXTENSIONS for name in get_target_files(file_names, ext)]
    totals = Counter(all_prefixes)
    potential_prefixes = [prefix for prefix, total in totals.items() if total == 8]
    if len(potential_prefixes) > 1:
        print(f"Multiple file prefixes available: {'\t'.join(potential_prefixes)}")
        print(f"Selecting prefix: {potential_prefixes[0]}")
    elif potential_prefixes:
        print(f"Selecting prefix: {potential_prefixes[0]}")
    else:
        print(f"No valid prefixes found in {path}")
        exit(1)
    return potential_prefixes[0]


def get_undercoordinated_nodes(netmc_network: NetMCNetwork) -> list[NetMCNode]:
    undercoordinated_nodes = []
    for node in netmc_network.nodes:
        if node.num_neighbours == 2:
            undercoordinated_nodes.append(node)
        elif node.num_neighbours != 3:
            raise ValueError(f"Node {node} has {node.num_neighbours} neighbours, expected 2 or 3.")
    return undercoordinated_nodes


def get_ring_to_undercoordinated_nodes_map(undercoordinated_nodes: list[NetMCNode]) -> dict[NetMCNode, list[NetMCNode]]:
    ring_to_undercoordinated_nodes = {}
    for node in undercoordinated_nodes:
        for ring in node.ring_neighbours:
            if ring not in ring_to_undercoordinated_nodes:
                ring_to_undercoordinated_nodes[ring] = []
            ring_to_undercoordinated_nodes[ring].append(node)
    return ring_to_undercoordinated_nodes


def find_common_elements(lists: list[list]) -> list:
    first_list = lists[0]
    common_elements = [element for element in first_list if all(element in lst for lst in lists[1:])]
    return common_elements


def get_ring_walk(ring: NetMCNode) -> list[NetMCNode]:
    """
    Returns a list of nodes such that the order is how they are connected in the ring.
    """
    walk = [ring.ring_neighbours[0]]
    while len(walk) < len(ring.ring_neighbours):
        current_node = walk[-1]
        for neighbour in current_node.neighbours:
            if neighbour in ring.ring_neighbours and neighbour not in walk:
                walk.append(neighbour)
                break
    return walk


def save_plot(output_path: Path, netmc_data: NetMCData, counter: int) -> None:
    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(12, 24)
    plt.ylim(12, 24)
    netmc_data.draw_graph()
    plt.savefig(output_path.joinpath(f"test_{counter}.png"))
    plt.clf()


class GraphPlotter:
    def __init__(self, netmc_data: NetMCData, output_path: Path):
        self.netmc_data = netmc_data
        self.output_path = output_path
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 10)
        self.fig.canvas.manager.window.title("NetMC Defect Editor")
        self.nodes = np.array([node.coord for node in self.netmc_data.base_network.nodes])
        self.click = False

    def on_key_press(self, event: KeyEvent) -> None:
        if event.key in ["q", "escape"]:
            plt.close(self.fig)

    def save_data(self) -> None:
        print(f"Saving data to {str(self.output_path)}...")
        self.netmc_data.check()
        self.netmc_data.export(self.output_path, "test")
        plt.close(self.fig)

    def on_press(self, event: MouseEvent):
        self.click = True

    def on_release(self, event: MouseEvent):
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode != '':
            return
        if self.click:
            self.onclick(event)
        self.click = False

    def onclick(self, event: MouseEvent):
        if event.dblclick:
            return
        try:
            if event.button == 1:  # Left click
                node, _ = self.netmc_data.base_network.get_nearest_node(np.array([event.xdata, event.ydata]))
                self.netmc_data.delete_node_and_merge_rings(node)
                print(f"Max ring size: {self.netmc_data.ring_network.max_ring_connections}")
                self.refresh_plot()

            elif event.button == 3:  # Right click
                print("Bonding undercoordinated nodes...")
                try:
                    self.netmc_data = NetMCData.bond_undercoordinated_nodes(self.netmc_data)
                    self.save_data()
                except InvalidUndercoordinatedNodesException as e:
                    if str(e) == "Number of undercoordinated nodes is odd, so cannot bond them.":
                        print("Number of undercoordinated nodes is odd, so cannot bond them.\n Please select another node to delete.")
                    elif str(e) == "There are three consecutive undercoordinated nodes in the ring walk.":
                        print("There are three consecutive undercoordinated nodes in the ring walk.\n Please select another node to delete.")
                    elif str(e) == "There are an odd number of undercoordinated nodes between two adjacent undercoordinated nodes.":
                        print("There are an odd number of undercoordinated nodes between two adjacent undercoordinated nodes.\n"
                              "This means we would have to bond an undercoordinated node to one of its own neighbours, which is not allowed.\n"
                              "Please select another node to delete.")
                    else:
                        raise
        except ValueError as e:
            if str(e) != "'x' must be finite, check for nan or inf values":
                raise

    def refresh_plot(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()

        plt.gca().set_aspect('equal', adjustable='box')
        self.netmc_data.draw_graph()

        # Only set xlim and ylim if they have been changed from their default values
        if xlim != (0.0, 1.0):
            self.ax.set_xlim(xlim)
        if ylim != (0.0, 1.0):
            self.ax.set_ylim(ylim)

        plt.pause(0.001)

    def plot(self):
        self.refresh_plot()
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.netmc_data.check()
        plt.title("NetMC Defect Editor")
        plt.show()


def main():
    cwd = Path.cwd()
    input_path = cwd.joinpath("test_input")
    output_path = cwd.joinpath("output_files")
    prefix = find_prefix(input_path)
    non_defect_netmc_data = NetMCData.from_files(input_path, prefix)
    non_defect_netmc_data.zero_coords()
    plotter = GraphPlotter(non_defect_netmc_data, output_path)
    plotter.plot()


if __name__ == "__main__":
    main()
