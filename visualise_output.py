# Code to produce coordinates of a lattice of hexagonal nodes and the connections between them
import matplotlib.pyplot as plt
from pathlib import Path
from utils import NetMCData
import shutil

def visualise_from_lammps():
    cwd = Path(__file__).parent
    lammps_netmc_path = cwd.parent.joinpath("LAMMPS-NetMC")


    for file_path in cwd.joinpath("test_input").iterdir():
        try:
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_A_crds.dat"),
                cwd.joinpath("test_input/test_A_crds.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_A_aux.dat"),
                cwd.joinpath("test_input/test_A_aux.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_A_net.dat"),
                cwd.joinpath("test_input/test_A_net.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_A_dual.dat"),
                cwd.joinpath("test_input/test_A_dual.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_B_crds.dat"),
                cwd.joinpath("test_input/test_B_crds.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_B_aux.dat"),
                cwd.joinpath("test_input/test_B_aux.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_B_net.dat"),
                cwd.joinpath("test_input/test_B_net.dat"))
    shutil.copy(lammps_netmc_path.joinpath("testing/output_files/test_B_dual.dat"),
                cwd.joinpath("test_input/test_B_dual.dat"))


    netmc_data = NetMCData.from_files(cwd.joinpath("test_input"), "test")
    return netmc_data
        

cwd = Path(__file__).parent


netmc_data = visualise_from_lammps()
# netmc_data = NetMCData.from_files(lammps_netmc_path.joinpath("testing/input_files"), "gi_test")
# netmc_data = NetMCData.gen_hexagonal(100)
# netmc_data.scale(2.3)
print(netmc_data)
if netmc_data.check():
    print("Data is consistent")
else:
    print("Data is inconsistent")
    
print("Average bond length: ", netmc_data.base_network.get_average_bond_length())
netmc_data.draw_graph(True, True, True, False, True, False, True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# netmc_data.export(lammps_netmc_path.joinpath("testing/input_files"), "gi_test")
