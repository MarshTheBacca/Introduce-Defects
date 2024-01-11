import cv2
from pathlib import Path
from typing import Callable
from collections import Counter
from utils import NetMCData
from matplotlib import pyplot as plt
import networkx as nx


def left_button_function(x: int, y: int):
    print(f"Left button function called with coordinates: ({x}, {y})")


def right_button_function(x: int, y: int):
    print(f"Right button function called with coordinates: ({x}, {y})")


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        left_button_function(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        right_button_function(x, y)
        cv2.destroyAllWindows()


def get_target_files(file_names: list, func: Callable) -> list:
    return_list = []
    for name in file_names:
        if func(name):
            return_list.append(name)
    return return_list


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


def plot_graph(graph: nx.Graph) -> None:
    pos = nx.get_node_attributes(graph, "pos")
    label_pos = {node: (x + 0.1, y + 0.1) for node, (x, y) in pos.items()}
    nx.draw(graph, pos, node_size=15, node_color="blue", edge_color="black")
    nx.draw_networkx_labels(graph, label_pos)
    plt.show()


def main():
    cwd = Path.cwd()
    prefix = find_prefix(cwd.joinpath("test_input"))
    crystal_netmc_data = NetMCData.from_files(cwd.joinpath("test_input"), prefix)
    crystal_netmc_data.zero_coords(crystal_netmc_data.base_network)
    crystal_netmc_data.check()

    plot_graph(crystal_netmc_data.base_network.graph)
    plt.clf()
    image = cv2.imread(str(cwd.joinpath("background.png")))
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
