import shutil
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

from utils import InvalidUndercoordinatedNodesException, LAMMPSData, NetMCData, get_valid_int, get_valid_str

matplotlib.use('TkAgg')


# Get some experimental data on energy coefficients for deviations from ideal bond lengths and angles
# Should we use isotopic masses or abundance based masses?


def get_target_files(file_names: list[str], extension: str) -> list[str]:
    """
    Get all files with a given extension from a list of file names
    Args:
        file_names: list of file names
        extension: extension to search for
    Returns:
        list of file names with the given extension
    """
    return [name for name in file_names if name.endswith(extension)]


def find_prefix(path: Path) -> str:
    """
    Finds the prefix of the files in the given directory
    Args:
        path: directory to search for files
    Returns:
        prefix of the files
    Raises:
        FileNotFoundError: if no valid prefixes are found
    """
    EXTENSIONS = ("aux.dat", "crds.dat", "net.dat", "dual.dat")
    file_names = [path.name for path in Path.iterdir(path) if path.is_file()]
    all_prefixes = [name[:-(len(ext) + 3)] for ext in EXTENSIONS for name in get_target_files(file_names, ext)]
    totals = Counter(all_prefixes)
    potential_prefixes = [prefix for prefix, total in totals.items() if total == 8]
    if not potential_prefixes:
        raise FileNotFoundError(f"No valid prefixes found in {path}")
    elif len(potential_prefixes) > 1:
        print(f"Multiple file prefixes available: {'\t'.join(potential_prefixes)}")
    print(f"Selecting prefix: {potential_prefixes[0]}")
    return potential_prefixes[0]


def select_network(path: Path, prompt: str) -> str | None:
    """
    Select a network to load from the given directory
    Args:
        path: directory to search for networks
    Returns:
        name of the network to load, or None if the user chooses to exit
    """
    network_names: list[str] = [path.name for path in Path.iterdir(path) if path.is_dir()]
    if not network_names:
        print(f"No networks found in {path}")
        return None
    exit_num: int = len(network_names) + 1
    for i, name in enumerate(network_names):
        prompt += f"{i + 1}) {name}\n"
    prompt += f"{exit_num}) Exit\n"
    option: int = get_valid_int(prompt, 1, exit_num)
    if option == exit_num:
        return None
    return network_names[option - 1]


def save_netmc_data(netmc_data: NetMCData, output_path: Path) -> None:
    """
    Asks the user if they would like to save the network, and if so, saves it to the given directory
    Args:
        netmc_data: NetMCData object to save
        output_path: directory to save the network to
    """
    option = get_valid_int("Would you like to save the network?\n1) Yes\n2) No\n", 1, 2)
    if option == 2:
        return
    while True:
        prefix = get_valid_str("Enter a prefix for the network files ('c' to cancel): ",
                               upper=255, forbidden_chars=[" ", "\\", "/"])
        if prefix == "c":
            return
        try:
            Path.mkdir(output_path.joinpath(prefix))
            break
        except FileExistsError:
            print("Network with that prefix already exists, please choose another prefix.")
    print(f"Saving network to {output_path.joinpath(prefix)}...")
    netmc_data.export(output_path.joinpath(prefix), prefix)


def rename_network_files(network_path: Path, old_prefix: str, new_prefix: str) -> None:
    """
    Renames the files in a network directory to use a new prefix
    Args:
        network_path: directory containing the network files
        old_prefix: old prefix to replace
        new_prefix: new prefix to use for the files
    """
    for file in network_path.iterdir():
        if file.is_file() or file.is_symlink():
            new_name = file.name.replace(old_prefix, new_prefix)
            file.rename(file.parent.joinpath(new_name))


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
                    plt.close(self.fig)
                except InvalidUndercoordinatedNodesException as e:
                    if str(e) == "Number of undercoordinated nodes is odd, so cannot bond them.":
                        print("Number of undercoordinated nodes is odd, so cannot bond them.\n Please select another node to delete.")
                    elif str(e) == "There are three consecutive undercoordinated nodes in the ring walk.":
                        print("There are three consecutive undercoordinated nodes in the ring walk.\n Please select another node to delete.")
                    elif str(e) == "There are an odd number of undercoordinated nodes between two pairs of adjacent undercoordinated nodes.":
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
        plt.title("Network Editor")
        plt.show()


def main():
    cwd = Path(__file__).parent
    networks_path = cwd.joinpath("networks")
    networks_path.mkdir(exist_ok=True)

    while True:
        option = get_valid_int("What would you like to do?\n1) Introduce defects"
                               "\n2) Visualise network\n3) Delete network"
                               "\n4) Copy a network\n5) Create a fixed_rings.dat file for a network"
                               "\n6) Exit\n", 1, 6)
        if option == 1:
            while True:
                option = get_valid_int("Would you like to load an existing network or create a new one?"
                                       "\n1) Load existing network\n2) Create new network\n3) Exit\n", 1, 3)
                if option == 1:
                    chosen_network_name = select_network(networks_path, "Select a network to load ('c' to cancel):\n")
                    if chosen_network_name is None:
                        continue
                    netmc_data = NetMCData.from_files(networks_path.joinpath(chosen_network_name),
                                                      chosen_network_name)
                    plotter = GraphPlotter(netmc_data, networks_path)
                    plotter.plot()
                    save_netmc_data(plotter.netmc_data, networks_path)

                elif option == 2:
                    while True:
                        print("Due to the process of creating a box-like network, the number of rings must be "
                              "equal to an even number squared, eg, 4, 16, 36, 64, 100, etc.\n"
                              "However, any number you enter will be rounded down to the nearest even number squared.")
                        num_rings = get_valid_int("How many rings would you like to create (minimum 4, 'c' to cancel)?\n", 4, exit_string="c")
                        if num_rings is None:
                            break
                        netmc_data = NetMCData.gen_hexagonal(num_rings)
                        plotter = GraphPlotter(netmc_data, networks_path)
                        plotter.plot()
                        save_netmc_data(plotter.netmc_data, networks_path)
                elif option == 3:
                    break
        elif option == 2:
            chosen_network_name = select_network(networks_path, "Select a network to visualise ('c' to cancel):\n")
            if chosen_network_name is None:
                continue
            netmc_data = NetMCData.from_files(networks_path.joinpath(chosen_network_name),
                                              chosen_network_name)
            netmc_data.draw_graph2(True)
            plt.show()
        elif option == 3:
            chosen_network_name = select_network(networks_path, "Select a network to delete ('c' to cancel):\n")
            if chosen_network_name is None:
                continue
            confirm = get_valid_int("Are you sure you want to delete this network?\n1) Yes\n2) No\n", 1, 2)
            if confirm == 1:
                shutil.rmtree(networks_path.joinpath(chosen_network_name))
                print(f"Deleted network {chosen_network_name}")
        elif option == 4:
            chosen_network_name = select_network(networks_path, "Select a network to copy ('c' to cancel):\n")
            if chosen_network_name is None:
                continue
            new_prefix = get_valid_str("Enter a prefix for the copied network ('c' to cancel): ", upper=255, forbidden_chars=[" ", "\\", "/"])
            if new_prefix == "c":
                continue
            shutil.copytree(networks_path.joinpath(chosen_network_name), networks_path.joinpath(new_prefix))
            rename_network_files(networks_path.joinpath(new_prefix), chosen_network_name, new_prefix)
            print(f"Copied network {chosen_network_name} to {new_prefix}")

        elif option == 5:
            chosen_network_name = select_network(networks_path, "Select a network to generate a fixed_rings.dat file for ('c' to cancel):\n")
            if chosen_network_name is None:
                continue
            netmc_data = NetMCData.from_files(networks_path.joinpath(chosen_network_name),
                                              chosen_network_name)
            netmc_data.draw_graph(True, True, True, False, False, False, True)
            plt.show()
            fixed_rings = []
            num_rings = len(netmc_data.ring_network.nodes)
            while True:
                print(f"Current fixed rings: {', '.join(str(ring) for ring in fixed_rings)}")
                ring = get_valid_int("Enter a ring to fix ('c' to confirm, enter the same ring twice to remove it): ", 0, num_rings, exit_string="c")
                if ring is None:
                    break
                elif ring in fixed_rings:
                    fixed_rings.remove(ring)
                else:
                    fixed_rings.append(ring)
            if fixed_rings:
                with open(networks_path.joinpath(chosen_network_name, "fixed_rings.dat"), "w") as f:
                    f.write("\n".join(str(ring) for ring in fixed_rings))
                print(f"Fixed rings written to {networks_path.joinpath(chosen_network_name, 'fixed_rings.dat')}")
        elif option == 6:
            break


if __name__ == "__main__":
    main()
