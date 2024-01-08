import numpy as np
from typing import TypeAlias, IO
import shutil
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
from utils import Node

Network: TypeAlias = dict[int, dict]
LAMMPS_DATA_DEFAULTS = {"atoms": 0, "bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0, "atom types": 0,
                        "bond types": 0, "angle types": 0, "dihedral types": 0, "improper types": 0,
                        "extra bond per atom": 0, "extra angle per atom": 0, "extra dihedral per atom": 0, "extra improper per atom": 0, "extra special per atom": 0,
                        "ellipsoids": 0, "lines": 0, "triangles": 0, "bodies": 0,
                        "xlo xhi": np.array([-0.5, 0.5]), "ylo yhi": np.array([-0.5, 0.5]), "zlo zhi": np.array([-0.5, 0.5]), "xy xz yz": np.array([0, 0, 0])}


class ConnectionType(Enum):
    net: str = "net"
    dual: str = "dual"


def plot_bilayer():
    plt.scatter(bilayer_si_crds[:, 0], bilayer_si_crds[:, 1], color='y', s=0.4)
    plt.scatter(bilayer_o_crds[:, 0], bilayer_o_crds[:, 1], color='r', s=0.4)
    plt.savefig('bilayer atoms')
    plt.clf()
    plt.scatter(bilayer_si_crds[:, 0], bilayer_si_crds[:, 1], color='y', s=0.4)
    plt.scatter(bilayer_o_crds[:, 0], bilayer_o_crds[:, 1], color='r', s=0.4)
    for i in range(bilayer_harmpairs.shape[0]):
        atom_1_crds = bilayer_crds[int(bilayer_harmpairs[i, 0]), :]
        atom_2_crds = bilayer_crds[int(bilayer_harmpairs[i, 1]), :]
        atom_2_crds = np.add(atom_1_crds, pbc_vector(
            atom_1_crds, atom_2_crds, scaled_dimensions))
        if int(bilayer_harmpairs[i, 0]) >= 4 * num_nodes_a or int(bilayer_harmpairs[i, 1]) >= 4 * num_nodes_a:
            plt.plot([atom_1_crds[0], atom_2_crds[0]], [
                     atom_1_crds[1], atom_2_crds[1]], color='k')
    plt.title('Si-O')
    plt.savefig('bilayer SiO bond')
    plt.clf()
    plt.scatter(bilayer_si_crds[:, 0], bilayer_si_crds[:, 1], color='y', s=0.4)
    plt.scatter(bilayer_o_crds[:, 0], bilayer_o_crds[:, 1], color='r', s=0.4)
    for i in range(bilayer_harmpairs.shape[0]):
        atom_1_crds = bilayer_crds[int(bilayer_harmpairs[i, 0]), :]
        atom_2_crds = bilayer_crds[int(bilayer_harmpairs[i, 1]), :]
        atom_2_crds = np.add(atom_1_crds, pbc_vector(
            atom_1_crds, atom_2_crds, scaled_dimensions))
        if int(bilayer_harmpairs[i, 0]) < 4 * num_nodes_a and int(bilayer_harmpairs[i, 1]) < 4 * num_nodes_a:
            plt.plot([atom_1_crds[0], atom_2_crds[0]], [
                     atom_1_crds[1], atom_2_crds[1]], color='k')
    plt.title('O-O')
    plt.savefig('bilayer OO bond')
    plt.clf()

    plt.scatter(bilayer_si_crds[:, 0], bilayer_si_crds[:, 2], color='y', s=0.4)
    plt.scatter(bilayer_o_crds[:, 0], bilayer_o_crds[:, 2], color='r', s=0.4)
    for i in range(bilayer_harmpairs.shape[0]):
        atom_1_crds = bilayer_crds[int(bilayer_harmpairs[i, 0]), :]
        atom_2_crds = bilayer_crds[int(bilayer_harmpairs[i, 1]), :]
        atom_2_crds = np.add(atom_1_crds, pbc_vector(
            atom_1_crds, atom_2_crds, scaled_dimensions))
        plt.plot([atom_1_crds[0], atom_2_crds[0]], [
                 atom_1_crds[2], atom_2_crds[2]], color='k')

    plt.savefig('bilayer all')
    plt.clf()


def nets_to_bonding_pairs(nets: np.ndarray) -> np.ndarray:
    pairs = np.array([[node, bonded_node] for node in range(len(nets))
                     for bonded_node in nets[node] if bonded_node > node])
    return pairs


def nets_to_angle_triples(nets: np.ndarray) -> np.ndarray:
    angles = np.array([[nets[i][j], i, nets[i][(j+1) % len(nets[i])]]
                      for i in range(len(nets)) for j in range(len(nets[i]))])
    return angles


def get_lammps_atoms_array_molecular(coords: np.array, molecule_ids: np.array = None, atom_types: list = None) -> list:
    # Return is of type list because molecule_ids may be non-numeric because you can set up molecule lables in LAMMPS
    if coords.shape[1] == 2:
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
    if molecule_ids is None:
        molecule_ids = np.linespace(1, coords.shape[0], coords.shape[0])
    if atom_types is None:
        atom_types = [1] * coords.shape[0]
    if len(molecule_ids) != coords.shape[0] or len(atom_types) != coords.shape[0]:
        raise ValueError(
            "Length of molecule_ids and atom_types must match length of coords.")
    atoms_array = []
    for i, (coordinate, molecule_id, atom_type) in enumerate(zip(coords, molecule_ids, atom_types)):
        atoms_array.append([i + 1, molecule_id, atom_type,
                           coordinate[0], coordinate[1], coordinate[2]])
    return atoms_array


def dimensions_to_lammps_parameters(dimensions: np.ndarray) -> dict:
    DIM_LABLES = ("xlo xhi", "ylo yhi", "zlo zhi")
    if len(dimensions) not in (2, 3):
        raise ValueError("Dimensions must be 2 or 3 dimensional.")
    return {DIM_LABLES[i]: dimensions[i] for i in range(len(dimensions))}


def write_section(header: str, data: list | np.ndarray, data_file: IO) -> None:
    data_file.write(f"\n{header}\n\n")
    for row in data:
        for element in row:
            data_file.write(f"{element:<20} ")
        data_file.write("\n")


def dict_to_2d_array(dictionary: dict) -> np.ndarray:
    return np.array([[key, value] for key, value in dictionary.items()])


def write_lammps_data(path: Path, description: str, parameters: dict, masses: dict,
                      atom_lables: dict, atoms: list[list], atom_type: str, bonds: list[list], angles: list[list] = None) -> None:
    ATOM_TYPE_DESCRIPTIONS = {"atomic": "atom types are atomic, ie columns are: atom_id, atom_type, x, y, z",
                              "molecular": "atom types are molecular, ie columns are: atom_id, molecule_id, atom_type, x, y, z"}
    with open(path, 'w') as data_file:
        data_file.write(f"{description}\n\n")
        for parameter, value in parameters.items():
            if parameter not in LAMMPS_DATA_DEFAULTS:
                raise ValueError(
                    f"Parameter {parameter} is not a valid LAMMPS data parameter.")
            if value != LAMMPS_DATA_DEFAULTS[parameter]:
                data_file.write(f'{parameter} {value}\n')
        write_section("Atom Type Lables",
                      dict_to_2d_array(atom_lables), data_file)
        write_section("Masses", dict_to_2d_array(masses), data_file)
        data_file.write(f"\n{ATOM_TYPE_DESCRIPTIONS[atom_type]}\n")
        write_section(f"Atoms # {atom_type}", atoms, data_file)
        write_section("Bonds", bonds, data_file)
        if angles is not None:
            write_section("Angles", angles, data_file)


def pbc_vector(vector1: np.ndarray, vector2: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    Calculate the vector difference between two vectors, taking into account periodic boundary conditions.
    """
    if len(vector1) != len(vector2) or len(vector1) != len(dimensions):
        raise ValueError("Vectors must have the same number of dimensions.")
    difference_vector = np.subtract(vector2, vector1)
    dimension_ranges = dimensions[:, 1] - dimensions[:, 0]
    difference_vector = (difference_vector + dimension_ranges /
                         2) % dimension_ranges - dimension_ranges / 2
    return difference_vector


def check_out_of_bounds(coords: np.ndarray, dimensions: np.ndarray) -> None:
    print("Checking if any coordinates are out of bounds...")
    print("Dimensions: xlo: {:<10} xhi: {:<10}\n           ylo: {:<10} yhi: {:<10}").format(
        *dimensions.flatten())
    for atom, coord in enumerate(coords):
        if coord[0] < dimensions[0][0] or coord[0] > dimensions[0][1]:
            print(f"Atom {atom} x coordinate out of bounds: {coord[0]}")
        elif coord[1] < dimensions[1][0] or coord[1] > dimensions[1][1]:
            print(f"Atom {atom} y coordinate out of bounds: {coord[1]}")


def make_crds_marks_bilayer(path: Path, intercept, triangle_raft: bool, bilayer: bool, common_files_path: Path):
    AREA = 1

    # Constants
    ANGSTROM_TO_BOHR = 1 / 0.52917721090380
    SI_SI_DISTANCE_FACTOR = np.sqrt(32.0 / 9.0)
    O_O_DISTANCE_FACTOR = np.sqrt(8.0 / 3.0)
    SI_O_LENGTH_ANGSTROM = 1.609
    H_ANGLE_DEGREES = 19.5

    # Scaling factors
    AREA_SCALING = np.sqrt(AREA)
    SI_SI_DISTANCE_BOHR = ANGSTROM_TO_BOHR * \
        SI_O_LENGTH_ANGSTROM * SI_SI_DISTANCE_FACTOR
    SI_O_LENGTH_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM
    O_O_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * O_O_DISTANCE_FACTOR
    H_BOHR = ANGSTROM_TO_BOHR * \
        np.sin(np.radians(H_ANGLE_DEGREES)) * SI_O_LENGTH_ANGSTROM

    # NetMC Aux dimensions are [[xhi, yhi], [xlo, ylo]]
    dimensions = np.genfromtxt(path.joinpath("testA_a_aux.dat"), skip_header=3)
    # Whereas LAMMPS dimensions are [[xlo, xhi], [ylo, yhi]]
    dimensions = dimensions.transpose()[:, ::-1]

    # Must open the file because nodes may have varying number of connections and therefore cannot be read into a numpy array
    with open(path.joinpath("testA_a_net.dat"), 'r') as f:
        nets = [np.array(line.strip().split(), dtype=np.int64) for line in f]
    nets += 1  # LAMMPS atom ids start at 1, not 0
    node_crds = np.genfromtxt(path.joinpath("testA_crds_a.dat"), 'r')
    dual_crds = np.genfromtxt(path.joinpath("testA_b_crds.dat"), 'r')

    num_nodes_b = dual_crds.shape[0]
    num_nodes_a = node_crds.shape[0]

    scaling_factor = np.sqrt(dual_crds.shape[0] / num_nodes_b)
    scaled_dimensions = dimensions * scaling_factor

    node_crds *= scaling_factor
    monolayer_bonding_pairs = nets_to_bonding_pairs(nets)
    monolayer_angles = nets_to_angle_triples(nets)

    shutil.copyfile(common_files_path.joinpath(
        "PARM_Si.lammps"), path.joinpath("PARM_Si.lammps"))
    shutil.copyfile(common_files_path.joinpath(
        "PARM_C.lammps"), path.joinpath("PARM_C.lammps"))
    shutil.copyfile(common_files_path.joinpath(
        "Si.in"), path.joinpath("Si.in"))
    shutil.copyfile(common_files_path.joinpath("C.in"), path.joinpath("C.in"))

    LAMMPS_DATA_TITLE = "DATA FILE Produced from netmc results (cf David Morley)"

    # Original function wrote dim lows as 0
    write_lammps_data(path.joinpath("Si.data"), LAMMPS_DATA_TITLE,
                      {"atoms": num_nodes_a,
                       "bonds": monolayer_bonding_pairs.shape[0],
                       "angles": monolayer_angles.shape[0],
                       "atom types": 1,
                       "bond types": 1,
                       "angle types": 1} | dimensions_to_lammps_parameters(scaled_dimensions),
                      {"Si": 27.9769265}, {1: "Si"},
                      get_lammps_atoms_array_molecular(node_crds),
                      "molecular",
                      nets_to_bonding_pairs(nets),
                      nets_to_angle_triples(nets))

    tersoff_crds = node_crds * 1.42
    tersoff_dims = scaled_dimensions * 1.42

    # Original function and this function don't include bonds or angles, currently will produce an error because function requires them
    # Original function wrote dim lows as 0
    write_lammps_data(path.joinpath("C.data"), LAMMPS_DATA_TITLE,
                      {"atoms": tersoff_crds.shape[0],
                       "atom types": 1} | dimensions_to_lammps_parameters(tersoff_dims),
                      {"C": 12.0000}, {1: "C"},
                      get_lammps_atoms_array_molecular(tersoff_crds),
                      "atomic")

    if triangle_raft:
        # Should we use isotopic masses or abundance based masses?

        # For Si2O3 coords, the first 2/5 of the coords are Si, the last 3/5 are O. All atoms z = 5.0
        # For SiO2 coords, the first 1/3 are Si, the rest are O. Si z alternates between 5 and 11.081138669036534 (5 + 2 * Si-O length)
        # The O coords in SiO2, the first 1/4 are z = 8.040569334518267, last 3/4 alternate between 3.9850371001619402, 12.096101568874595 (8.040569334518267 + 2 * Si-O length)

        shutil.copyfile(common_files_path.joinpath(
            "Si2O3.in"), path.joinpath("Si2O3.in"))

        # The last vector was originally [-0.5, -np.sqrt(3) / 3], but this seems wrong
        DISPLACEMENT_VECTORS_NORM = np.array(
            [[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
        DISPLACEMENT_VECTORS_FACTORED = DISPLACEMENT_VECTORS_NORM * 0.5

        triangle_raft_dimensions = dimensions * SI_SI_DISTANCE_BOHR * AREA_SCALING
        dimension_ranges = triangle_raft_dimensions[:,
                                                    1] - triangle_raft_dimensions[:, 0]

        triangle_raft_si_crds = node_crds * SI_SI_DISTANCE_BOHR * AREA_SCALING

        dict_sio = {i: [] for i in range(
            int(num_nodes_a * 3 / 2), int(num_nodes_a * 5 / 2))}

        tringle_raft_num_nodes = num_nodes_a
        for index, net in enumerate(nets):
            si_atom = index + 1  # LAMMPS atom ids start at 1, not 0
            for connected_si_atom in net:
                vector_between_si_atoms = pbc_vector(
                    tringle_raft_si_crds[index], tringle_raft_si_crds[connected_si_atom - 1], triangle_raft_dimensions)
                normalized_vector = vector_between_si_atoms / \
                    np.linalg.norm(vector_between_si_atoms)
                dot_product_grades = np.abs(
                    np.dot(normalized_vector, DISPLACEMENT_VECTORS_NORM.T))
                selected_vector_index = np.argmin(dot_product_grades)
                midpoint = (
                    tringle_raft_si_crds[index] + vector_between_si_atoms / 2 + dimension_ranges)
                if dot_product_grades[selected_vector_index] < 0.1:
                    oxygen_coord = midpoint + \
                        DISPLACEMENT_VECTORS_FACTORED[selected_vector_index] % dimension_ranges
                else:
                    oxygen_coord = midpoint % dimension_ranges
                triangle_raft_num_nodes += 1
                oxygen_atom_id = triangle_raft_num_nodes
                triangle_raft_bonding_pairs = np.vstack(
                    (triangle_raft_bonding_pairs, np.array([si_atom, oxygen_atom_id])))
                triangle_raft_

        for pair_index, (node_1, node_2) in enumerate(monolayer_bonding_pairs):
            atom_coordinates_1 = triangle_raft_si_crds[node_1]
            atom_coordinates_2 = triangle_raft_si_crds[node_2]

            distance_vector = pbc_vector(
                atom_coordinates_1, atom_coordinates_2, triangle_raft_dimensions)
            normalized_vector = distance_vector / \
                np.linalg.norm(distance_vector)

            dot_product_grades = np.abs(
                np.dot(normalized_vector, DISPLACEMENT_VECTORS_NORM.T))
            selected_vector_index = np.argmin(dot_product_grades)

            unperturbed_oxygen_coordinates = atom_coordinates_1 + distance_vector / 2
            oxygen_coordinates = unperturbed_oxygen_coordinates + \
                (DISPLACEMENT_VECTORS_FACTORED[selected_vector_index]
                 if dot_product_grades[selected_vector_index] < 0.1 else 0)
            oxygen_coordinates = (oxygen_coordinates +
                                  dimension_ranges) % dimension_ranges

            triangle_raft_o_crds = np.vstack(
                (triangle_raft_o_crds, oxygen_coordinates)) if pair_index else oxygen_coordinates
            pair = np.array([[pair_index, node_1 + num_nodes_a * 1.5],
                            [pair_index, node_2 + num_nodes_a * 1.5]])
            triangle_raft_bonding_pairs = np.vstack(
                (triangle_raft_bonding_pairs, pair)) if pair_index else pair

            dict_sio.setdefault(
                int(node_1 + num_nodes_a * 1.5), []).append(pair_index)
            dict_sio.setdefault(
                int(node_2 + num_nodes_a * 1.5), []).append(pair_index)

        harmpairs_list = []
        for i in range(int(num_nodes_a * 1.5), int(num_nodes_a * 2.5)):
            pairs = [dict_sio[str(i)][j:k+1] for j in range(2)
                     for k in range(j + 1, 3)]
            harmpairs_list.extend(pairs)

        triangle_raft_bonding_pairs = np.array(harmpairs_list)
        triangle_raft_crds = np.vstack(
            (triangle_raft_o_crds, triangle_raft_si_crds))

        check_out_of_bounds(triangle_raft_crds, triangle_raft_dimensions)

        n_bonds = triangle_raft_bonding_pairs.shape[0]
        n_bond_types = 2

        with open(path.joinpath("PARM_Si2O3.lammps"), 'w') as output_file:
            output_file.write(
                f"pair_style lj/cut {O_O_DISTANCE_BOHR * intercept}\n")
            output_file.write(
                f"pair_coeff * * 0.1 {O_O_DISTANCE_BOHR * intercept / 2**(1 / 6)} {O_O_DISTANCE_BOHR * intercept}")
            output_file.write("pair_modify shift yes\n")
            output_file.write("special_bonds lj 0.0 1.0 1.0\n")
            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 2.86667626014\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

        with open(path.joinpath("Si2O3.data"), 'w') as f:
            f.write('DATA FILE Produced from netmc results (cf David Morley)\n')
            f.write('{:} atoms\n'.format(triangle_raft_crds.shape[0]))
            f.write('{:} bonds\n'.format(int(n_bonds)))
            f.write('0 angles\n')
            f.write('0 dihedrals\n')
            f.write('0 impropers\n')
            f.write('2 atom types\n')
            f.write('{:} bond types\n'.format(int(n_bond_types)))
            f.write('0 angle types\n')
            f.write('0 dihedral types\n')
            f.write('0 improper types\n')
            f.write('0.00000 {:<5} xlo xhi\n'.format(
                triangle_raft_dimensions[0]))
            f.write('0.00000 {:<5} ylo yhi\n'.format(
                triangle_raft_dimensions[1]))
            f.write('\n')
            f.write('# Pair Coeffs\n')
            f.write('#\n')
            f.write('# 1  O\n')
            f.write('# 2  Si\n')
            f.write('\n')
            f.write('# Bond Coeffs\n')
            f.write('# \n')
            f.write('# 1  O-O\n')
            f.write('# 2  Si-O\n')
            f.write('# 3  O-O rep\n')
            f.write('# 4  Si-Si rep\n')
            f.write('\n')
            f.write(' Masses\n')
            f.write('\n')
            f.write('1 28.10000 # O \n')
            f.write('2 32.01000 # Si\n')
            f.write('\n')
            f.write(' Atoms # molecular\n')
            f.write('\n')

            for i in range(triangle_raft_si_crds.shape[0]):
                f.write('{:<4} {:<4} {:<4} {:<24} {:<24} {:<24} # Si\n'.format(int(i + 1),
                                                                               int(i + 1),
                                                                               2, triangle_raft_si_crds[i, 0],
                                                                               triangle_raft_si_crds[i, 1], 5.0))
            for i in range(triangle_raft_o_crds.shape[0]):
                f.write('{:<4} {:<4} {:<4} {:<24} {:<24} {:<24} # O\n'.format(int(i + 1 + triangle_raft_si_crds.shape[0]),
                                                                              int(
                                                                                  i + 1 + triangle_raft_si_crds.shape[0]),
                                                                              1, triangle_raft_o_crds[i, 0],
                                                                              triangle_raft_o_crds[i, 1], 5.0))

            f.write('\n')
            f.write(' Bonds\n')
            f.write('\n')
            for i in range(triangle_raft_bonding_pairs.shape[0]):
                pair1 = triangle_raft_bonding_pairs[i, 0]
                if pair1 < triangle_raft_o_crds.shape[0]:
                    pair1_ref = pair1 + 1 + triangle_raft_si_crds.shape[0]
                else:
                    pair1_ref = pair1 + 1 - triangle_raft_o_crds.shape[0]
                pair2 = triangle_raft_bonding_pairs[i, 1]
                if pair2 < triangle_raft_o_crds.shape[0]:
                    pair2_ref = pair2 + 1 + triangle_raft_si_crds.shape[0]
                else:
                    pair2_ref = pair2 + 1 - triangle_raft_o_crds.shape[0]

                if triangle_raft_bonding_pairs[i, 0] < triangle_raft_o_crds.shape[0] and triangle_raft_bonding_pairs[i, 1] < triangle_raft_o_crds.shape[0]:

                    f.write('{:} {:} {:} {:} \n'.format(int(i + 1), 1, int(pair1_ref),
                                                        int(pair2_ref)))
                else:

                    f.write('{:} {:} {:} {:} \n'.format(int(i + 1), 2, int(pair1_ref),
                                                        int(pair2_ref)))

        with open(path.joinpath("Si2O3_harmpairs.dat"), 'w') as f:
            f.write('{:}\n'.format(triangle_raft_bonding_pairs.shape[0]))
            for i in range(triangle_raft_bonding_pairs.shape[0]):
                if triangle_raft_bonding_pairs[i, 0] < triangle_raft_o_crds.shape[0] and triangle_raft_bonding_pairs[i, 1] < triangle_raft_o_crds.shape[0]:
                    f.write('{:<10} {:<10} \n'.format(int(triangle_raft_bonding_pairs[i, 0] + 1),
                                                      int(triangle_raft_bonding_pairs[i, 1] + 1)))
                else:
                    f.write('{:<10} {:<10} \n'.format(int(triangle_raft_bonding_pairs[i, 0] + 1),
                                                      int(triangle_raft_bonding_pairs[i, 1] + 1)))

    if bilayer:
        shutil.copyfile(common_files_path.joinpath(
            "SiO2.in"), path.joinpath("SiO2.in"))
        print("############ Bilayer ###############")
        # Si Atoms
        for i in range(triangle_raft_si_crds.shape[0]):
            if i == 0:
                bilayer_si_crds = np.asarray([[triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5],
                                              [triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + 2 * SI_O_LENGTH_BOHR]])
            else:
                bilayer_si_crds = np.vstack((bilayer_si_crds, np.asarray([[triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5],
                                                                          [triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + 2 * SI_O_LENGTH_BOHR]])))
        # O ax Atoms
        for i in range(triangle_raft_si_crds.shape[0]):
            if i == 0:
                bilayer_o_crds = np.asarray(
                    [triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + SI_O_LENGTH_BOHR])
            else:
                bilayer_o_crds = np.vstack((bilayer_o_crds, np.asarray(
                    [triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + SI_O_LENGTH_BOHR])))
        # O eq
        for i in range(triangle_raft_o_crds.shape[0]):
            bilayer_o_crds = np.vstack((bilayer_o_crds, np.asarray(
                [triangle_raft_o_crds[i, 0], triangle_raft_o_crds[i, 1], 5 - H_BOHR])))
            bilayer_o_crds = np.vstack((bilayer_o_crds, np.asarray(
                [triangle_raft_o_crds[i, 0], triangle_raft_o_crds[i, 1], 5 + H_BOHR + 2 * SI_O_LENGTH_BOHR])))

        bilayer_crds = np.vstack((bilayer_o_crds, bilayer_si_crds))

        dict_sio2 = {}

        # Harmpairs
        # O ax
        for i in range(triangle_raft_si_crds.shape[0]):
            if i == 0:
                bilayer_harmpairs = np.asarray([[i, 4 * num_nodes_a + 2 * i],  # 3200
                                                [i, 4 * num_nodes_a + \
                                                    1 + 2 * i],  # 3201
                                                [i, map_index_to_bilayer(
                                                    dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][0])[0]],
                                                [i, map_index_to_bilayer(
                                                    dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][0])[1]],
                                                [i, map_index_to_bilayer(
                                                    dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][1])[0]],
                                                [i, map_index_to_bilayer(
                                                    dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][1])[1]],
                                                [i, map_index_to_bilayer(
                                                    dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][2])[0]],
                                                [i, map_index_to_bilayer(dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][2])[1]]])
            else:
                bilayer_harmpairs = np.vstack((bilayer_harmpairs, np.asarray([[i, 4 * num_nodes_a + 2 * i],  # 3200
                                                                              # 3201
                                                                              [i, 4 * num_nodes_a + \
                                                                                  1 + 2 * i],
                                                                              [i, map_index_to_bilayer(
                                                                                  dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][0])[0]],
                                                                              [i, map_index_to_bilayer(
                                                                                  dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][0])[1]],
                                                                              [i, map_index_to_bilayer(
                                                                                  dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][1])[0]],
                                                                              [i, map_index_to_bilayer(
                                                                                  dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][1])[1]],
                                                                              [i, map_index_to_bilayer(
                                                                                  dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][2])[0]],
                                                                              [i, map_index_to_bilayer(dict_sio['{:}'.format(int(3 * num_nodes_a / 2 + i))][2])[1]]])))
        # Si - O cnxs
        for i in range(triangle_raft_bonding_pairs.shape[0]):
            atom_1 = map_index_to_bilayer(triangle_raft_bonding_pairs[i, 0])
            atom_2 = map_index_to_bilayer(triangle_raft_bonding_pairs[i, 1])

            bilayer_harmpairs = np.vstack((bilayer_harmpairs, np.asarray(
                [[atom_1[0], atom_2[0]], [atom_1[1], atom_2[1]]])))

        for vals in dict_sio.keys():
            dict_sio2['{:}'.format(int(vals) - 3 * num_nodes_a / 2 + 4 * num_nodes_a)] = [
                map_index_to_bilayer(dict_sio["{:}".format(vals)][i]) for i in range(3)]

        plot_bilayer()

        n_bonds = bilayer_harmpairs.shape[0]

        with open(path.joinpath("PARM_SiO2.lammps"), 'w') as output_file:
            output_file.write(
                'pair_style lj/cut {:}\n'.format(O_O_DISTANCE_BOHR * intercept))
            output_file.write('pair_coeff * * 0.1 {:} {:}\n'.format(
                O_O_DISTANCE_BOHR * intercept / 2**(1 / 6), O_O_DISTANCE_BOHR * intercept))
            output_file.write('pair_modify shift yes\n'.format())
            output_file.write('special_bonds lj 0.0 1.0 1.0\n'.format())

            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 3.0405693345182674\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

        with open(path.joinpath("SiO2.data"), 'w') as f:
            f.write('DATA FILE Produced from netmc results (cf David Morley)\n')
            f.write('{:} atoms\n'.format(bilayer_crds.shape[0]))
            f.write('{:} bonds\n'.format(int(n_bonds)))
            f.write('0 angles\n')
            f.write('0 dihedrals\n')
            f.write('0 impropers\n')
            f.write('2 atom types\n')
            f.write('0 bond types\n')
            f.write('{:} bond types\n'.format(int(n_bond_types)))
            f.write('0 angle types\n')
            f.write('0 dihedral types\n')
            f.write('0 improper types\n')
            f.write('0.00000 {:<5} xlo xhi\n'.format(scaled_dimensions[0]))
            f.write('0.00000 {:<5} ylo yhi\n'.format(scaled_dimensions[1]))
            f.write('0.0000 200.0000 zlo zhi\n')
            f.write('\n')
            f.write('# Pair Coeffs\n')
            f.write('#\n')
            f.write('# 1  O\n')
            f.write('# 2  Si\n')
            f.write('\n')
            f.write('# Bond Coeffs\n')
            f.write('# \n')
            f.write('# 1  O-O\n')
            f.write('# 2  Si-O\n')
            f.write('# 3  O-O rep\n')
            f.write('# 4  Si-Si rep\n')
            f.write('\n')
            f.write(' Masses\n')
            f.write('\n')
            f.write('1 28.10000 # O \n')
            f.write('2 32.01000 # Si\n')
            f.write('\n')
            f.write(' Atoms # molecular\n')
            f.write('\n')

            for i in range(bilayer_si_crds.shape[0]):
                f.write('{:<4} {:<4} {:<4} {:<24} {:<24} {:<24} # Si\n'.format(int(i + 1),
                                                                               int(i + 1),
                                                                               2,
                                                                               bilayer_si_crds[i, 0],
                                                                               bilayer_si_crds[i, 1],
                                                                               bilayer_si_crds[i, 2],
                                                                               ))
            for i in range(bilayer_o_crds.shape[0]):
                f.write('{:<4} {:<4} {:<4} {:<24} {:<24} {:<24} # O\n'.format(int(i + 1 + bilayer_si_crds.shape[0]),
                                                                              int(
                                                                                  i + 1 + bilayer_si_crds.shape[0]),
                                                                              1,
                                                                              bilayer_o_crds[i, 0],
                                                                              bilayer_o_crds[i, 1],
                                                                              bilayer_o_crds[i, 2],
                                                                              ))

            f.write('\n')
            f.write(' Bonds\n')
            f.write('\n')
            for i in range(bilayer_harmpairs.shape[0]):

                pair1 = bilayer_harmpairs[i, 0]
                if pair1 < bilayer_o_crds.shape[0]:
                    pair1_ref = pair1 + 1 + bilayer_si_crds.shape[0]
                else:
                    pair1_ref = pair1 + 1 - bilayer_o_crds.shape[0]
                pair2 = bilayer_harmpairs[i, 1]
                if pair2 < bilayer_o_crds.shape[0]:
                    pair2_ref = pair2 + 1 + bilayer_si_crds.shape[0]
                else:
                    pair2_ref = pair2 + 1 - bilayer_o_crds.shape[0]

                if bilayer_harmpairs[i, 0] < bilayer_o_crds.shape[0] and bilayer_harmpairs[i, 1] < bilayer_o_crds.shape[0]:
                    f.write('{:} {:} {:} {:}\n'.format(
                        int(i + 1), 1, int(pair1_ref), int(pair2_ref)))
                else:
                    f.write('{:} {:} {:} {:}\n'.format(
                        int(i + 1), 2, int(pair1_ref), int(pair2_ref)))

        with open(path.joinpath("SiO2_harmpairs.dat"), 'w') as f:
            f.write('{:}\n'.format(bilayer_harmpairs.shape[0]))
            for i in range(bilayer_harmpairs.shape[0]):
                if bilayer_harmpairs[i, 0] < bilayer_o_crds.shape[0] and bilayer_harmpairs[i, 1] < bilayer_o_crds.shape[0]:
                    f.write('{:<10} {:<10}\n'.format(
                        int(bilayer_harmpairs[i, 0] + 1), int(bilayer_harmpairs[i, 1] + 1)))
                else:
                    f.write('{:<10} {:<10}\n'.format(
                        int(bilayer_harmpairs[i, 0] + 1), int(bilayer_harmpairs[i, 1] + 1)))

    print('Finished')
