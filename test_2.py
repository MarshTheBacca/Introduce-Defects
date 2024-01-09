import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import (LAMMPSAngle, LAMMPSAtom, LAMMPSBond, LAMMPSData, NetMCData,
                   NetworkType)

# Constants
ANGSTROM_TO_BOHR = 1 / 0.52917721090380
O_O_DISTANCE_FACTOR = np.sqrt(8.0 / 3.0)
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
    plt.clf()


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


def netmc_to_triangle_raft(netmc_data: NetMCData) -> LAMMPSData:
    SI_SI_DISTANCE_FACTOR = np.sqrt(32.0 / 9.0)
    SI_SI_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * SI_SI_DISTANCE_FACTOR
    # The last vector was originally [-0.5, -np.sqrt(3) / 3], but this seems wrong vvvvv
    DISPLACEMENT_VECTORS_NORM = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
    DISPLACEMENT_VECTORS_FACTORED = DISPLACEMENT_VECTORS_NORM * 0.5

    triangle_raft_lammps_data = LAMMPSData.from_netmc_data(netmc_data, NetworkType.A, atom_label="Si", atomic_mass=28.1, atom_style="atomic")
    triangle_raft_lammps_data.scale_coords(SI_SI_DISTANCE_BOHR)
    triangle_raft_lammps_data.add_atom_label("O")
    triangle_raft_lammps_data.add_mass("O", 15.995)

    dimension_ranges = triangle_raft_lammps_data.dimensions[:, 1] - triangle_raft_lammps_data.dimensions[:, 0]
    for si_atom in triangle_raft_lammps_data.atoms:
        bonded_si_atoms = triangle_raft_lammps_data.get_bonded_atoms(si_atom)
        for bonded_si_atom in bonded_si_atoms:
            vector_between_si_atoms = pbc_vector(si_atom.coords, bonded_si_atom.coords, dimension_ranges)
            normalized_vector = vector_between_si_atoms / np.linalg.norm(vector_between_si_atoms)
            dot_product_grades = np.abs(np.dot(normalized_vector, DISPLACEMENT_VECTORS_NORM.T))
            selected_vector_index = np.argmin(dot_product_grades)
            midpoint = (si_atom.coords + vector_between_si_atoms / 2) % dimension_ranges

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
def write_lammps_files(path: Path, non_defect_netmc_data: NetMCData, intercept: int, triangle_raft: bool, bilayer: bool, common_files_path: Path):
    print("Writing LAMMPS files...")
    netmc_data = NetMCData.import_data(path, prefix="testA")
    scaling_factor = np.sqrt(netmc_data.num_nodes_b / non_defect_netmc_data.num_nodes_b)

    si_lammps_data = LAMMPSData.from_netmc_data(netmc_data, NetworkType.A, atom_label="Si", atomic_mass=29.977, atom_style="atomic")
    si_lammps_data.scale_coords(scaling_factor)
    # Original function wrote dim lows as 0
    si_lammps_data.export(path.joinpath("Si.data"))
    shutil.copyfile(common_files_path.joinpath("Si.in"), path.joinpath("Si.in"))
    shutil.copyfile(common_files_path.joinpath("PARM_Si.lammps"), path.joinpath("PARM_Si.lammps"))

    c_lammps_data = LAMMPSData.from_netmc_data(netmc_data, NetworkType.A, atom_label="C", atomic_mass=12.0000, atom_style="atomic")
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

        triangle_raft_lammps_data = netmc_to_triangle_raft(netmc_data)
        triangle_raft_lammps_data.export(path.joinpath("Si2O3.data"))

        with open(path.joinpath("PARM_Si2O3.lammps"), 'w') as output_file:
            output_file.write(f"pair_style lj/cut {O_O_DISTANCE_BOHR * intercept}\n")
            output_file.write(f"pair_coeff * * 0.1 {O_O_DISTANCE_BOHR * intercept / 2**(1 / 6)} {O_O_DISTANCE_BOHR * intercept}")
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
                bottom_si_atom = LAMMPSAtom(atom.coords + np.array([0, 0, 5]), "Si")
                top_si_atom = LAMMPSAtom(atom.coords + np.array([0, 0, 5 + 2 * SI_O_LENGTH_BOHR]), "Si")
                central_o_atom = LAMMPSAtom(atom.coords + np.array([0, 0, 5 + SI_O_LENGTH_BOHR]), "O")
                bilayer_lammps_data.add_atom(bottom_si_atom)
                bilayer_lammps_data.add_atom(top_si_atom)
                bilayer_lammps_data.add_atom(central_o_atom)
            if atom.label == "O":
                bottom_o_atom = LAMMPSAtom(atom.coords + np.array([0, 0, 5 - H_BOHR]), "O")
                top_o_atom = LAMMPSAtom(atom.coords + np.array([0, 0, 5 + H_BOHR + 2 * SI_O_LENGTH_BOHR]), "O")
                bilayer_lammps_data.add_atom(top_o_atom)
                bilayer_lammps_data.add_atom(bottom_o_atom)

        bilayer_lammps_data.bond_atoms_within_distance(1.1 * SI_O_LENGTH_BOHR)
        bilayer_lammps_data.export(path.joinpath("SiO2.data"))
        bilayer_lammps_data.export_bonds(path.joinpath("SiO2_harmpairs.dat"))

        with open(path.joinpath("PARM_SiO2.lammps"), 'w') as output_file:
            output_file.write('pair_style lj/cut {:}\n'.format(O_O_DISTANCE_BOHR * intercept))
            output_file.write(f"pair_coeff * * 0.1 {O_O_DISTANCE_BOHR * intercept / 2**(1 / 6)} {O_O_DISTANCE_BOHR * intercept}\n")
            output_file.write("pair_modify shift yes\n")
            output_file.write("special_bonds lj 0.0 1.0 1.0\n")
            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 3.0405693345182674\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

    print("Finished writing LAMMPS files.")
