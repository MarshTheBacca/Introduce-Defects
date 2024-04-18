import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import matplotlib
from matplotlib import pyplot as plt
from tabulate import tabulate

from utils import (BSSData, DefectIntroducer, LAMMPSData, UserCancelledError,
                   confirm, get_valid_float, get_valid_int, get_valid_str)

matplotlib.use('TkAgg')

NETWORKS_PATH = Path(__file__).parent.joinpath("networks")


class MissingFilesError(Exception):
    pass


# Get some experimental data on energy coefficients for deviations from ideal bond lengths and angles
# Should we use isotopic masses or abundance based masses?


def select_network(path: Path, prompt: str) -> str:
    """
    Select a network to load from the given directory
    Args:
        path: directory to search for networks
    Returns:
        name of the network to load
    Raises:
        MissingFilesError: If no networks are found in the given directory
        UserCancelledError: If the user chooses to exit
    """
    network_array = []
    sorted_paths = sorted(Path.iterdir(path), key=lambda p: p.stat().st_ctime, reverse=True)
    for i, path in enumerate(sorted_paths):
        if path.is_dir():
            name = path.name
            creation_date = datetime.fromtimestamp(path.stat().st_ctime).strftime('%d/%m/%Y %H:%M:%S')
            network_array.append((i + 1, name, creation_date))
    if not network_array:
        raise MissingFilesError(f"No networks found in {path}")
    exit_num: int = len(network_array) + 1
    print(tabulate(network_array, headers=["Number", "Network Name", "Date Modified"], tablefmt="fancy_grid"))
    prompt += f" ({exit_num} to exit):\n"
    option: int = get_valid_int(prompt, 1, exit_num)
    if option == exit_num:
        raise UserCancelledError("User chose to exit")
    return network_array[option - 1][1]


def save_network(bss_data: BSSData) -> None:
    """
    Asks the user if they would like to save the network, and if so, opens a dialog for them to choose a save location
    Also creates a LAMMPS data file
    Args:
        bss_data: BSSData object to save
    """
    if not confirm("Would you like to save the network? (y/n)"):
        return
    # Create a dialog box to get a path to save the network in
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Browse to networks folder", initialdir=NETWORKS_PATH)
    if not directory:
        print("No directory selected. Network will not be saved.")
        return
    networks_path = Path(directory).resolve()

    while True:
        print(f"Max ring size: {bss_data.get_ring_size_limits()[1]}")
        name = get_valid_str("Enter a name for the new network: ", upper=50, forbidden_chars=[" ", "\\", "/", ","])
        save_path = networks_path.joinpath(name)
        try:
            save_path.mkdir(exist_ok=False)
            break
        except FileExistsError:
            print("Network with that name already exists, please choose another name.")
    print("Graphene preset: Bond length 2.1580672 Bohr Radii, Atom Label C, Atomic Mass 12")
    if confirm("Would you like to apply graphene preset? (y/n)"):
        scale = 2.1580672
        atom_label = "C"
        atomic_mass = 12
    else:
        try:
            scale, atom_label, atomic_mass = get_lammps_params()
        except UserCancelledError:
            return
    bss_data.scale(scale)
    print(f"Saving network to {save_path}...")
    bss_data.export(save_path)
    lammps_data = LAMMPSData.from_bss_network(bss_data.base_network, atom_label=atom_label, atomic_mass=atomic_mass, atom_style="molecular")
    lammps_data.export(save_path.joinpath("lammps_network.txt"))


def get_lammps_params() -> tuple[float, str, float]:
    """
    Prompts the user to enter the scale, atom label and atomic mass for the LAMMPS data file
    Returns:
        scale (float): The scale factor for the network
        atom_label (str): The label for the atom in the network
        atomic_mass (float): The atomic mass of the atom in the network
    Raises:
        UserCancelledError: If the user cancels the input
    """
    print("Common scales: Graphene: 2.1580672 Bohr Radii")
    scale = get_valid_float("Enter the desired bond length in Bohr Radii (must be greater than 0, 'c' to cancel)\n", 0, float("inf"), exit_string="c")
    atom_label = get_valid_str("Enter the atom label for the network ('c' to cancel): ", lower=1, upper=2, forbidden_chars=[" ", "\\", "/", ","])
    atomic_mass = get_valid_float("Enter the atomic mass of the atom in the network ('c' to cancel)\n", 0, float("inf"), exit_string="c")
    return scale, atom_label, atomic_mass


def introduce_defects(networks_path: Path) -> None:
    option = get_valid_int("Would you like to load an existing network or create a new one?"
                           "\n1) Load existing network\n2) Create new network\n3) Exit\n", 1, 3)
    if option == 3:
        return
    elif option == 1:
        try:
            chosen_network_name = select_network(networks_path, "Select a network to load")
        except MissingFilesError as e:
            print(e)
            return
        except UserCancelledError:
            return
        bss_data = BSSData.from_files(networks_path.joinpath(chosen_network_name))
    elif option == 2:
        print("\nDue to the process of creating a box-like network, the number of rings must be "
              "equal to an even number squared, eg, 4, 16, 36, 64, 100, etc.\n"
              "However, any number you enter will be rounded down to the nearest even number squared.\n")
        try:
            num_rings = get_valid_int("How many rings would you like to create (minimum 4, 'c' to cancel)?\n", 4, exit_string="c")
        except UserCancelledError:
            return
        bss_data = BSSData.gen_hexagonal(num_rings)
    plotter = DefectIntroducer(bss_data, networks_path)
    plotter.plot()
    if not plotter.closed_properly:
        print("User closed the plot window, exiting without saving...")
        return
    save_network(plotter.bss_data)


def visualise_network(networks_path: Path) -> None:
    print("Note - large networks can take a while to plot, so be patient")
    try:
        chosen_network_name = select_network(networks_path, "Select a network to visualise")
    except MissingFilesError as e:
        print(e)
        return
    except UserCancelledError:
        return
    bss_data = BSSData.from_files(networks_path.joinpath(chosen_network_name))
    if confirm("Enable debugging view? (y/n)"):
        bss_data.draw_graph(True, True, True, True, True, True, True)
    else:
        bss_data.draw_graph_pretty(title=chosen_network_name, draw_dimensions=True)
    plt.show()


def delete_network(networks_path: Path) -> None:
    while True:
        try:
            chosen_network_name = select_network(networks_path, "Select a network to delete")
        except MissingFilesError as e:
            print(e)
            return
        except UserCancelledError:
            return
        if confirm(f"Are you sure you want to delete {chosen_network_name}? (y/n)"):
            shutil.rmtree(networks_path.joinpath(chosen_network_name))
            print(f"Deleted network {chosen_network_name}")


def copy_network(networks_path: Path) -> None:
    try:
        chosen_network_name = select_network(networks_path, "Select a network to copy")
    except MissingFilesError as e:
        print(e)
        return
    except UserCancelledError:
        return
    if chosen_network_name is None:
        return
    new_name = get_valid_str("Enter a name for the copied network ('c' to cancel): ", upper=255, forbidden_chars=[" ", "\\", "/"])
    if new_name == "c":
        return
    shutil.copytree(networks_path.joinpath(chosen_network_name), networks_path.joinpath(new_name))
    print(f"Copied network {chosen_network_name} to {new_name}")


def rename_network(networks_path: Path) -> None:
    try:
        chosen_network_name = select_network(networks_path, "Select a network to rename")
    except MissingFilesError as e:
        print(e)
        return
    except UserCancelledError:
        return
    if chosen_network_name is None:
        return
    new_name = get_valid_str("Enter a new name for the network ('c' to cancel): ", upper=255, forbidden_chars=[" ", "\\", "/"])
    if new_name == "c":
        return
    shutil.move(networks_path.joinpath(chosen_network_name), networks_path.joinpath(new_name))
    print(f"Renamed network {chosen_network_name} to {new_name}")


def create_fixed_rings_file(networks_path: Path) -> None:
    try:
        chosen_network_name = select_network(networks_path, "Select a network to generate a fixed_rings.txt file for")
    except MissingFilesError as e:
        print(e)
        return
    except UserCancelledError:
        return
    bss_data = BSSData.from_files(networks_path.joinpath(chosen_network_name))
    bss_data.draw_graph(True, True, True, False, False, False, True)
    plt.show()
    fixed_rings = []
    num_rings = len(bss_data.ring_network.nodes)
    while True:
        print(f"Current fixed rings: {', '.join(str(ring) for ring in fixed_rings)}")
        try:
            ring = get_valid_int("Enter a ring to fix ('c' to confirm, enter the same ring twice to remove it): ", 0, num_rings, exit_string="c")
        except UserCancelledError:
            break
        if ring in fixed_rings:
            fixed_rings.remove(ring)
        else:
            fixed_rings.append(ring)
    if fixed_rings:
        fixed_rings_path = networks_path.joinpath(chosen_network_name, "fixed_rings.txt")
        with open(fixed_rings_path, "w") as f:
            f.write("\n".join(str(ring) for ring in fixed_rings))
        print(f"Fixed rings written to {fixed_rings_path}")


def main():
    NETWORKS_PATH.mkdir(exist_ok=True)
    while True:
        option = get_valid_int("What would you like to do?\n1) Introduce defects"
                               "\n2) Visualise network"
                               "\n3) Delete network"
                               "\n4) Copy a network"
                               "\n5) Rename a network"
                               "\n6) Create a fixed_rings.txt file for a network"
                               "\n7) Exit\n", 1, 7)
        if option == 1:
            introduce_defects(NETWORKS_PATH)
        elif option == 2:
            visualise_network(NETWORKS_PATH)
        elif option == 3:
            delete_network(NETWORKS_PATH)
        elif option == 4:
            copy_network(NETWORKS_PATH)
        elif option == 5:
            rename_network(NETWORKS_PATH)
        elif option == 6:
            create_fixed_rings_file(NETWORKS_PATH)
        elif option == 7:
            break


if __name__ == "__main__":
    main()
