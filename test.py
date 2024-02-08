from pathlib import Path
import networkx as nx
from matplotlib import pyplot as plt
import difflib
import time

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


t1 = time.perf_counter()
netmc_data = NetMCData.gen_triangle_lattice(10000)
t2 = time.perf_counter()
netmc_data.draw_graph(True, True, True, True, True, False, False)
t3 = time.perf_counter()

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
t4 = time.perf_counter()

print(f"Time to generate lattice: {t2 - t1}")
print(f"Time to draw graph: {t3 - t2}")
print(f"Time to show graph: {t4 - t3}")
