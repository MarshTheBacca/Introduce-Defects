from pathlib import Path
import networkx as nx
from matplotlib import pyplot as plt

from utils import NetMCData
import filecmp


def compare_directories(dir1, dir2):
    comparison = filecmp.dircmp(dir1, dir2)

    # Files that are only in dir1
    for file in comparison.left_only:
        print(f"File only in {dir1}: {file}")

    # Files that are only in dir2
    for file in comparison.right_only:
        print(f"File only in {dir2}: {file}")

    # Files that are common to both directories but have different contents
    for file in comparison.diff_files:
        print(f"Differences in file {file}:")
        diff = filecmp.cmp(dir1 + '/' + file, dir2 + '/' + file)
        if not diff:
            with open(dir1 + '/' + file, 'r') as file1, open(dir2 + '/' + file, 'r') as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()
                for line in difflib.unified_diff(lines1, lines2, fromfile=f"{dir1}/{file}", tofile=f"{dir2}/{file}"):
                    print(line)

    # Recursively compare subdirectories
    for subdir in comparison.common_dirs:
        compare_directories(dir1 + '/' + subdir, dir2 + '/' + subdir)


def plot_graph(graph: nx.Graph) -> None:
    pos = nx.get_node_attributes(graph, "pos")
    label_pos = {node: (x + 0.1, y + 0.1) for node, (x, y) in pos.items()}
    nx.draw(graph, pos, node_size=15, node_color="blue", edge_color="black")
    nx.draw_networkx_labels(graph, label_pos)
    plt.show()


cwd = Path.cwd()
netmc_data = NetMCData.from_files(cwd.joinpath("test_input"), "test")
netmc_data.check()
netmc_data.delete_node(netmc_data.base_network.nodes[0])
netmc_data.check()
output_netmc_data = NetMCData.from_files(cwd.joinpath("test_output"), "test")
output_netmc_data.check()
if netmc_data == output_netmc_data:
    print("Success!")
else:
    if netmc_data.base_network != output_netmc_data.base_network:
        print("Base network not equal!")
        if netmc_data.base_network.nodes != output_netmc_data.base_network.nodes:
            print("Base network nodes not equal!")
        if netmc_data.base_network.bonds != output_netmc_data.base_network.bonds:
            print("Base network bonds not equal!")
        if netmc_data.base_network.ring_bonds != output_netmc_data.base_network.ring_bonds:
            print("Base network ring bonds not equal!")
            print(len(netmc_data.base_network.ring_bonds))
        if netmc_data.base_network.geom_code != output_netmc_data.base_network.geom_code:
            print("Base network geom code not equal!")

    if netmc_data.ring_network != output_netmc_data.ring_network:
        print("Ring network not equal!")
