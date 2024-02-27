# Code to produce coordinates of a lattice of hexagonal nodes and the connections between them
import matplotlib.pyplot as plt
from pathlib import Path
from utils import NetMCData, LAMMPSData
import shutil

cwd = Path(__file__).parent
lammps_netmc_path = cwd.parent.joinpath("LAMMPS-NetMC")
images_path = cwd.parent.joinpath("LAMMPS-NetMC/testing/Images")

# NetMC-LAMMPS output
netmc_data = NetMCData.from_files(lammps_netmc_path.joinpath("testing/output_files"), "test")

# NetMC-LAMMPS input
# netmc_data = NetMCData.from_files(lammps_netmc_path.joinpath("testing/input_files"), "gi_test")

# Introduce defects output
# netmc_data = NetMCData.from_files(cwd.joinpath("output_files"), "test")

# From scratch
# netmc_data = NetMCData.gen_hexagonal(256)
# netmc_data.scale(2.3)

print(netmc_data)
if netmc_data.check():
    print("Data is consistent")
else:
    print("Data is inconsistent")

for node in netmc_data.base_network.nodes:
    if node.num_neighbours != 3:
        print("Node", node.id, "has", node.num_neighbours, "neighbours")
        break
print("Average bond length: ", netmc_data.base_network.get_average_bond_length())

netmc_data.draw_graph2(True)
# netmc_data.draw_graph(True, True, True, False, False, True, True)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(images_path.joinpath("network_background.png"), bbox_inches='tight', transparent=False)
plt.show()
# netmc_data.plot_radial_distribution()

#netmc_data.export(cwd.joinpath("test_input"), "test")


# output_path = cwd.parent.joinpath("LAMMPS-NetMC/testing/input_files")
# for file in output_path.iterdir():
#     if file.is_file() or file.is_symlink():
#         if file.name != "Si_potential.in" and file.name != "Si.in":
#             file.unlink()
# netmc_data.export(output_path, "gi_test")
# lammps_data = LAMMPSData.from_netmc_network(netmc_data.base_network, "Si", 27.9769265, "molecular")
# lammps_data.export(output_path.joinpath("Si.data"))
