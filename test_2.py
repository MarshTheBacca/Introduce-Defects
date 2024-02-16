import copy
import shutil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utils import (CouldNotBondUndercoordinatedNodesException,
                   InvalidNetworkException,
                   InvalidUndercoordinatedNodesException, LAMMPSAngle,
                   LAMMPSAtom, LAMMPSBond, LAMMPSData, NetMCData, NetMCNetwork,
                   NetMCNode)

# Constants
ANGSTROM_TO_BOHR = 1 / 0.52917721090380
O_O_DISTANCE_FACTOR = np.sqrt(8 / 3)
SI_O_LENGTH_ANGSTROM = 1.609
H_ANGLE_DEGREES = 19.5

# Scaling factors
SI_O_LENGTH_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM
O_O_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * O_O_DISTANCE_FACTOR
H_BOHR = ANGSTROM_TO_BOHR * np.sin(np.radians(H_ANGLE_DEGREES)) * SI_O_LENGTH_ANGSTROM


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


def netmc_to_triangle_raft(netmc_network: NetMCNetwork) -> LAMMPSData:
    print("Generating triangle raft...")
    # This will only work for exactly hexagonal networks
    SI_SI_DISTANCE_FACTOR = np.sqrt(32) / 3
    SI_SI_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * SI_SI_DISTANCE_FACTOR
    # The last vector was originally [-0.5, -np.sqrt(3) / 3], but this seems wrong vvvvv
    DISPLACEMENT_VECTORS_NORM = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
    DISPLACEMENT_VECTORS_FACTORED = DISPLACEMENT_VECTORS_NORM * 0.5
    triangle_raft_lammps_data = LAMMPSData.from_netmc_network(netmc_network, atom_label="Si", atomic_mass=28.1, atom_style="atomic")
    # triangle_raft_lammps_data.scale_coords(SI_SI_DISTANCE_BOHR)
    triangle_raft_lammps_data.add_mass("O", 15.995)
    dimension_ranges = triangle_raft_lammps_data.dimensions[1] - triangle_raft_lammps_data.dimensions[0]
    nodes_to_add = []
    for bond in triangle_raft_lammps_data.get_bonds():
        vector_between_si_atoms = bond.get_pbc_vector(netmc_network.dimensions)
        normalized_vector = vector_between_si_atoms / np.linalg.norm(vector_between_si_atoms)
        midpoint = (bond.atom_1.coord + vector_between_si_atoms / 2) % dimension_ranges
        gradings = [abs(np.dot(normalized_vector, displacement_vector)) for displacement_vector in DISPLACEMENT_VECTORS_NORM]
        selected_vector = gradings.index(min(gradings))
        if gradings[selected_vector] < 0.1:
            new_o_atom = LAMMPSAtom(midpoint + DISPLACEMENT_VECTORS_FACTORED[selected_vector], "O")
        else:
            new_o_atom = LAMMPSAtom(midpoint, "O")
        new_o_atom.neighbours = [bond.atom_1, bond.atom_2]
        nodes_to_add.append(new_o_atom)
    for node in nodes_to_add:
        triangle_raft_lammps_data.add_atom(node)
    print("Finished generating triangle raft.")
    return triangle_raft_lammps_data


# Intercept is defined when initalising DrawLineWidget as 1? No idea why


cwd = Path.cwd()
netmc_data = NetMCData.gen_hexagonal(100)
netmc_data.scale(2.3)
cwd.joinpath("to_lammps").mkdir(exist_ok=True)
netmc_data.export(cwd.joinpath("to_lammps"), "gi_test")
lammps_data = LAMMPSData.from_netmc_network(netmc_data.base_network, atom_label="Si", atomic_mass=28.1, atom_style="molecular")
for angle in lammps_data.angles:
    print(angle)
lammps_data.export(cwd.joinpath("to_lammps", "gi_test.data"))
lammps_data.draw_graph(["Si"], ["Si-Si"], {"Si": "yellow"}, {"Si-Si": "black"}, {"Si": 30}, True, False)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# tr_lammps_data = netmc_to_triangle_raft(netmc_data.base_network)
# print(tr_lammps_data.num_atoms)

# tr_lammps_data.check()
# tr_lammps_data.draw_graph(["Si", "O"], ["O-Si", "Si-Si", "O-O"], {"Si": "yellow", "O": "red"},
#                           {"O-Si": "black", "Si-Si": "black", "O-O": "black"},
#                           {"Si": 30, "O": 15}, False, False)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
