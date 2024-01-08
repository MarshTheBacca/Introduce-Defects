import cv2
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
import shutil
import numpy as np
from sys import exit
from pathlib import Path
from typing import Callable
from utils import NetMCAux, NetMCNets, NetworkType
from typing import TypeAlias
from enum import Enum

# The comment he's included in C.data 'Atoms' line is wrong, the atoms are being stored as regular atoms, not molecules
# since there is a missing molecule ID column.

# Comment for Masses in C.data is incorrect, should be 'C' not 'Si'

# For some reason, he's written Si.data, SiO2.data and Si2O3.data atoms as molecule types, but with different molecule IDs

# In C.data, Si.data, all atoms have a z coordinate of 0
# In SiO2.data, all Si atoms alternate between z = 5 and z = 11.081138669036534
# In SiO2.data, the first 1/3 of O atoms have z = 8.040569334518267, then they alternate between z = 3.9850371001619402, z = 12.096101568874595
# In Si2O3.data, all atoms have z = 5

# Need to have a look at his C++ source code to see how *.data, *.in and *.lammps are being used
# And therefore if any corrections are in order for how he writes these files.

Network: TypeAlias = dict[int, dict]

class ConnectionType(Enum):
    net: str = "net"
    dual: str = "dual"

CONNECTION_TYPE_MAP ={ConnectionType.net: "net", ConnectionType.dual: "dual"}
X_OFFSET = 100
Y_OFFSET = 100

def get_close_node(node_positions: dict, coordinate: np.ndarray) -> int | None:
    print(node_positions)
    for node in node_positions:
        center_coordinates = np.array([int(scale * node_positions[node][0] + X_OFFSET),
                                       int(scale * node_positions[node][1] + Y_OFFSET)])
        distance = np.linalg.norm(np.subtract(center_coordinates, coordinate))
        print(distance, scale * 2 / 3)
        
        if distance < scale * 2 / 3:
            return node
    return None


def left_click(coordinate: np.ndarray, graph_a: nx.graph, network_a: Network, network_b: Network,
               deleted_nodes: np.ndarray, broken_rings: np.ndarray, undercoordinated_nodes: np.ndarray,
               image: np.ndarray):
    print("Left click detected at coordinates: ", coordinate)
    node_pos = nx.get_node_attributes(graph_a, "pos")
    close_node = get_close_node(node_pos, coordinate)
    if close_node is None:
        raise ValueError("No node found at coordinate: ", coordinate)
    deleted_nodes = np.append(deleted_nodes, close_node)
    broken_rings = np.unique(np.append(broken_rings, network_a[close_node]["dual"]))

    # add newly broken rings to list
    for ring in broken_rings:
        if close_node in network_b[ring]["dual"]:
            network_b[ring]["net"] = np_remove_elements(network_b[ring]["net"], close_node)

    # add the newly undercoordinated nodes to list
    for node in network_a[close_node]["net"]:
        undercoordinated_nodes  = np.append(undercoordinated_nodes, node)
        network_a[node]["net"] = np_remove_elements(network_a[node]["net"], close_node)
        if node in deleted_nodes:
            undercoordinated_nodes = np_remove_elements(undercoordinated_nodes, node)
    del network_a[close_node]
    refresh_new_cnxs(network_a, network_b, graph_a, graph_b, np.array([]), undercoordinated_nodes, image)
    cv2.imshow("image", image)
    print("Waiting...")
    return network_a, network_b, deleted_nodes, broken_rings, undercoordinated_nodes

def rekey(network: Network, deleted_nodes: np.ndarray) -> dict[int, int]:
    key = {}
    for node in network:
        remapped_node = node
        for deleted_node in deleted_nodes:
            if remapped_node >= deleted_node:
                remapped_node -= 1
        key[node] = remapped_node
    return key


def update_graph(graph: nx.graph, network: Network) -> None:
    graph.clear()
    graph = get_graph(network)
    return graph

def draw_circle(image: np.ndarray, coord: tuple[float, float], radius: float,
                color: tuple[int, int, int], thickness: float) -> None:
    x_coord = scale * coord[0] + X_OFFSET
    y_coord = scale * coord[1] + Y_OFFSET
    cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, thickness)


def draw_line(image: np.ndarray, coord_1: tuple[float, float], coord_2: tuple[float, float],
              color: tuple[int, int, int], thickness: int) -> None:
    x_coord_0 = scale * coord_1[0] + X_OFFSET
    y_coord_0 = scale * coord_1[1] + Y_OFFSET
    x_coord_1 = scale * coord_2[0] + X_OFFSET
    y_coord_1 = scale * coord_2[1] + Y_OFFSET
    cv2.line(image, (int(x_coord_0), int(y_coord_0)), (int(x_coord_1), int(y_coord_1)), color, thickness)


def draw_nodes(image: np.ndarray, graph: nx.graph,
               colour: tuple[int, int, int], thickness: float) -> None:
    coords = nx.get_node_attributes(graph, 'pos').values()
    for coord in coords:
        draw_circle(image, coord, int(0.1 * scale), colour, thickness)


def draw_cnxs(image: np.ndarray, graph: nx.graph, 
              colour: tuple[int, int, int], thickness: float) -> None:
    coords = nx.get_node_attributes(graph, 'pos')
    for node in graph:
        edges = graph.edges(node)
        for node_1, node_2 in edges:
            if node_1 < node_2:
                dx = coords[node_2][0] - coords[node_1][0]
                dy = coords[node_2][1] - coords[node_1][1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < 10:
                    draw_line(image, coords[node_1], coords[node_2], colour, thickness)

def refresh_new_cnxs(network_a: Network, network_b: Network,
                     graph_a: nx.graph, graph_b: nx.graph, atoms: np.ndarray,
                     undercoordinated_nodes: np.ndarray, image) -> None:
    graph_a = update_graph(graph_a, network_a)
    graph_b = update_graph(graph_b, network_b)

    draw_nodes(image, graph_a, (255, 0, 0), -1)
    draw_nodes(image, graph_b, (0, 255, 0), -1)
    
    draw_cnxs(image, graph_a, (255, 36, 12), 2)
    draw_cnxs(image, graph_b, (255, 36, 12), 1)

    for node in undercoordinated_nodes:
        if node in network_a:
            draw_circle(image, network_a[node]["crds"], 15, (36, 255, 12))
        else:
            print("Uncoordinated Atoms Doesnt Exist")

    for atom in atoms:
        if atom in network_a:
            atom_index = np.nonzero(atoms==atom)[0][0]
            r, g, b = (int(5 + atom_index * 20)) % 255, 36, int(abs(255 - atom_index * 20))
            draw_circle(image, network_a[atom]["crds"], 10, (r, g, b))
        else:
            print("Starting From atom doesn't Exist!")

    for i in range(0, len(atoms) - 1, 2):
        draw_line(image, network_a[atoms[i]]["crds"], network_a[atoms[i + 1]]["crds"], (1, 1, 1), 5)


def np_remove_elements(array: np.ndarray, elements_to_remove: np.ndarray) -> np.ndarray:
    elements_to_remove = np.array([elements_to_remove]) if np.isscalar(elements_to_remove) else elements_to_remove
    print(array, elements_to_remove)
    mask = np.logical_not(np.isin(array, elements_to_remove))
    print(mask)
    return array[mask]


def find_nodes_connections(network: Network, nodes: np.ndarray, connection_type: ConnectionType) -> np.ndarray:
    connected_nodes = {connected_node for node in nodes for connected_node in network[node][CONNECTION_TYPE_MAP[connection_type]]}
    return np.array(list(connected_nodes))


def remove_connections(network: Network, nodes: np.ndarray,
                       connections_to_remove: np.ndarray, connection_type: ConnectionType) -> Network:
    for node in nodes:
        network[node][CONNECTION_TYPE_MAP[connection_type]] = np_remove_elements(network[node][CONNECTION_TYPE_MAP[connection_type]],
                                                                                 connections_to_remove)
    return network


def add_connections(network: Network, nodes: np.ndarray,
                    connections_to_add: np.ndarray, connection_type: ConnectionType) -> Network:
    for node in nodes:
        network[node][CONNECTION_TYPE_MAP[connection_type]] = np.append(network[node][CONNECTION_TYPE_MAP[connection_type]],
                                                                        connections_to_add)
        network[node][CONNECTION_TYPE_MAP[connection_type]] = np.unique(network[node][CONNECTION_TYPE_MAP[connection_type]])
    return network

def replace_node_ring_connections(network: Network, rings_to_merge: np.ndarray,
                                  merged_ring: int) -> dict[int, dict]:
    for node in network:
        for ring in network[node]["dual"]:
            if ring in rings_to_merge:
                index = np.nonzero(network[node]["dual"] == ring)[0][0]
                network[node]["dual"][index] = merged_ring
    return network

def average_coords(nodes: np.ndarray, network: Network, dims: int = 2) -> np.ndarray:
    coords = np.zeros(dims)
    for node in nodes:
        coords = np.add(coords, network[node]["crds"])
    return np.divide(coords, len(nodes))

def remove_nodes(network: Network, nodes_to_remove: np.ndarray) -> Network:
    for node in nodes_to_remove:
        del network[node]
    return network

def remapped_node(rings_to_merge: np.ndarray, network_a: Network,
                network_b: Network) -> tuple[int, Network, Network]:
    merged_ring = min(rings_to_merge)
    connected_rings = find_nodes_connections(rings_to_merge, network_b, ConnectionType.net)
    network_b[merged_ring]["net"] = connected_rings
    rings_to_remove = np_remove_elements(rings_to_merge, merged_ring)
    remove_connections(network_b, connected_rings, rings_to_remove, ConnectionType.net)
    add_connections(network_b, connected_rings, merged_ring, ConnectionType.net)
    network_b[merged_ring]["dual"] = find_nodes_connections(rings_to_merge, network_b, ConnectionType.dual)
    network_a = replace_node_ring_connections(network_a, rings_to_merge, merged_ring)
    new_crds = average_coords(rings_to_merge, network_b)
    network_b[merged_ring]["crds"] = new_crds
    network_b = remove_nodes(network_b, rings_to_remove)
    print("Merged Rings: ", rings_to_merge)
    return merged_ring, network_a, network_b

def find_paths(node: np.int64, potential_node_cnxs: np.ndarray, network: Network) -> np.ndarray:
    paths = find_shared_cnxs(node, potential_node_cnxs, network)
    if not paths:
        raise ValueError("No paths found")
    return potential_node_cnxs[paths[0]]


def create_secondary_path(node_1: np.int64, node_2: np.int64, undercoordinated_nodes: np.ndarray,
                          broken_rings: np.ndarray, network_a: Network, network_b: Network, 
                          background: np.ndarray, atoms: np.ndarray,
                          graph_a: nx.graph, graph_b: nx.graph, clone: np.ndarray)-> tuple[Network, Network, np.ndarray, np.ndarray, np.ndarray]:
    while undercoordinated_nodes:
        # Check if the newly formed connection is to a site with no further connections
        if node_2 not in undercoordinated_nodes:
        # If so, we travel around to the next undercoordinated atom ...
            node_2 = find_paths(node_2, undercoordinated_nodes, network_a)
        check_undercoordinated(undercoordinated_nodes, network_a)
        node_3 = find_paths(node_2, undercoordinated_nodes, network_a)
        if node_3 not in network_a[node_2]["net"] and node_2 not in network_a[node_3]["net"]:
            network_a[node_2]["net"] = np.append(network_a[node_2]["net"], node_3)
            network_a[node_3]["net"] = np.append(network_a[node_3]["net"], node_2)
            undercoordinated_nodes = np_remove_elements(undercoordinated_nodes, node_2)
            undercoordinated_nodes = np_remove_elements(undercoordinated_nodes, node_3)
            atoms = np.append(atoms, [node_2, node_3])
            refresh_new_cnxs(network_a, network_b, graph_a, graph_b,
                             atoms, undercoordinated_nodes, clone)
            cv2.imshow("image", background)
            cv2.imshow('image', draw_line_widget.show_image())
            cv2.waitKey(0)
            for ring in broken_rings:
                if node_2 in [i for i in network_b[ring]["dual"]] and node_3 in [i for i in network_b[ring]["dual"]]:
                    broken_rings = np_remove_elements(broken_rings, ring)
        else:
            print("Nodes Already Connected!")
        node_2 = node_3

    network_a, network_b, new_ring = remapped_node(broken_rings, network_a, network_b)
    return network_a, network_b, new_ring, broken_rings, atoms


def create_initial_path(node_1, node_2, atoms: np.ndarray, network_1: Network, network_2: Network, undercoordinated_nodes: np.ndarray,
                        broken_rings: np.ndarray) -> tuple:
    network_1[node_2]["net"] = np.append(network_1[node_2]["net"], node_1)
    network_1[node_1]["net"] = np.append(network_1[node_1]["net"], node_2)
    undercoordinated_nodes = np.setdiff1d(undercoordinated_nodes, np.array([node_1, node_2]))
    for ring in broken_rings:
        if node_1 in network_2[ring]["dual"] and node_2 in network_2[ring]["dual"]:
            broken_rings = np_remove_elements(broken_rings, ring)
    atoms = np.append(atoms, node_2)
    return network_1, undercoordinated_nodes, broken_rings


def find_shared_cnxs(node: int | np.int64, undercoordinated_nodes: np.ndarray, network: Network) -> np.ndarray:
    node_dual = network[node]["dual"]
    shared_cnxs = np.array([])
    for i, undercoordinated_node in enumerate(undercoordinated_nodes):
        if undercoordinated_node not in network:
            print('This atom is deleted, code breaks here')
            exit(1)
        undercoordinated_node_dual = network[undercoordinated_node]["dual"]
        shared_cnxs = np.append(shared_cnxs, len(np.intersect1d(node_dual, undercoordinated_node_dual)))
    # 0 connections - More than one ring apart
    # 1 connection  - One ring apart
    # 2 connections - share an edge
    # 3 connections - same node!
    paths = np.nonzero(shared_cnxs == 1)[0]
    num_ways = len(paths)
    STRINGS = {1: "One way around the ring...", 2: "Two ways around the ring...",
               3: "Three ways around the ring..."}
    if num_ways == 0:
        paths = np.nonzero(shared_cnxs == 2)[0]
        num_ways = len(np.nonzero(shared_cnxs == 2)[0])
        if num_ways == 1:
            print("One way around the ring via a shared edge...")
            return paths
        else:
            print("Error: no cross ring connections")
            exit(1)
    elif num_ways < 4:
        print(STRINGS[num_ways])
        return paths
    print("num_ways > 5, program likely to crash")


def check_undercoordinated(uncoordinated_nodes: list[int], network: Network) -> None:
    for uncoordinated_node in uncoordinated_nodes:
        num_cnxs = network[uncoordinated_node]["net"].shape[0]
        if num_cnxs <= 2:
            continue
        if num_cnxs == 3:
            print(f"{uncoordinated_node} is not undercoordinated, exiting...")
        elif num_cnxs > 3:
            print(f"{uncoordinated_node} is overcoordinated, exiting...")
        print(f"uncoordinated node's net: {network[uncoordinated_node]['net']}")
        exit(1)


def check_path(node_1: int, node_2: int, network: Network, undercoordinated: np.ndarray) -> bool:
    if node_1 not in undercoordinated or node_2 not in undercoordinated:
        print(f"node_1: {node_1} or node)2: {node_2} not found in undercoordinated: {undercoordinated}")
        exit(1)
    if np.intersect1d(network[node_1]["net"], network[node_2]["net"]).size > 0:
        return False
    return True


def remove_deleted_nodes(deleted_nodes: list, undercoordinated_nodes: list) -> list:
    overlap = set(deleted_nodes).intersection(undercoordinated_nodes)
    print(f"Overlap before: {overlap}")
    for node in overlap:
        undercoordinated_nodes.remove(node)
    print(f"Overlap after: {set(deleted_nodes).intersection(undercoordinated_nodes)}")
    return undercoordinated_nodes


def export_crds(path: Path, network: Network) -> None:
    crds_array = np.array([node["crds"] for node in network])
    np.savetxt(path, crds_array, fmt="%-19.6f")


def get_target_files(file_names: list, func: Callable[[str], bool]) -> list:
    return_list = []
    for name in file_names:
        if func(name):
            return_list.append(name)
    return return_list


def import_crds(path: Path) -> np.array:
    with open(path, "r") as crds_file:
        return np.genfromtxt(crds_file, dtype=np.float64)


def find_prefix(path):
    file_names = [path.name for path in Path.iterdir(path) if path.is_file()]
    aux_files = get_target_files(file_names, lambda name: name.endswith("aux.dat"))
    crds_files = get_target_files(file_names, lambda name: name.endswith("crds.dat"))
    net_files = get_target_files(file_names, lambda name: name.endswith("net.dat"))
    dual_files = get_target_files(file_names, lambda name: name.endswith("dual.dat"))
    all_prefixes = [name[:-10] for name in aux_files] + [name[:-11] for name in crds_files]
    all_prefixes += [name[:-10] for name in net_files] + [name[:-11] for name in dual_files]
    totals = {}
    for prefix in all_prefixes:
        if prefix not in totals:
            totals[prefix] = 1
        else:
            totals[prefix] += 1
    potential_prefixes = []
    for prefix, total in totals.items():
        if total == 8:
            potential_prefixes.append(prefix)
    if len(potential_prefixes) > 1:
        return_prefix = potential_prefixes[0]
        string = "Multiple file prefixes available: "
        for prefix in potential_prefixes:
            string += f"{prefix}\t"
        print(string)
    elif potential_prefixes:
        return_prefix = potential_prefixes[0]
        print(f"Selecting prefix: {prefix}")
    else:
        print("No valid prefixes found in {path}")
        exit(1)
    print(f"Selecting prefix: {return_prefix}")
    return return_prefix


def read_netmc_files(path, prefix):
    file_types = ("crds.dat", "net.dat", "dual.dat")
    return_array = []
    for file_type in file_types:
        print(f"{prefix}_{file_type}")
        with open(path.joinpath(f"{prefix}_{file_type}"), "r") as file:
            return_array.append(np.genfromtxt(file))
    return return_array


def get_graph(network: Network) -> nx.Graph:
    network_graph = nx.Graph()
    for selected_node in network:
        network_graph.add_node(selected_node, pos=network[selected_node]["crds"])
        for bonded_node in network[selected_node]["net"]:
            network_graph.add_edge(selected_node, bonded_node)
    return network_graph


def ordered_cnxs(network_1: Network, network_2: Network, cnx_type: str) -> dict:
    # ordered_cnxs_dual_to_dual = ordered_cnxs(network_b, network_b, "dual")
    # ordered_cnxs_dual_to_node = ordered_cnxs(network_b, network_a, "net")
    for node in network_1:
        node_cnxs = network_1[node][cnx_type]
        node_cnxs_new = np.array([node_cnxs[0]])
        node_cnxs_new_crds = np.array([network_2[node_cnxs[0]]["crds"]])
        for i in range(0, len(node_cnxs)):
            node_new = node_cnxs_new[i]
            node_new_net = network_2[node_new]["net"]
            options = np.intersect1d(node_new_net, node_cnxs)
            options = np.setdiff1d(options, node_cnxs_new)
            node_cnxs_new = np.append(node_cnxs_new, options[0])
            node_cnxs_new_crds = np.append(node_cnxs_new_crds, network_2[options[0]]["crds"])
        area = 0
        for i, crd_2 in enumerate(node_cnxs_new_crds):
            crd_1 = node_cnxs_new_crds[i - 1]
            area += crd_1[0] * crd_2[1] - crd_2[0] * crd_1[1]
        if area > 0:
            node_cnxs_new.reverse()
        network_1[node][cnx_type] = node_cnxs_new
    return network_1


def write_key(path: Path, deleted_nodes: np.ndarray, network: Network) -> None:
    with open(path, "w") as key_file:
        for deleted_node in deleted_nodes:
            key_file.write(f"{deleted_node:<5}")
        key_file.write('\n')
        for node in network:
            translated_node = node
            for deleted_node in deleted_nodes:
                if translated_node >= deleted_node:
                    translated_node -= 1
            key_file.write(f"{node:<10}{translated_node:<10}\n")


def write(output_path: Path, input_path: Path, network_a: Network, network_b: Network, deleted_nodes_a: np.ndarray,
                deleted_nodes_b: np.ndarray, new_ring: int | np.int64) -> None:
    num_nodes_a = len(network_a)
    num_nodes_b = len(network_b)
    max_cnxs_a = max([node["net"].shape[0] for node in network_a])
    min_cnxs_a = min([node["net"].shape[0] for node in network_a])
    max_cnxs_b = max([node["net"].shape[0] for node in network_b])
    min_cnxs_b = min([node["net"].shape[0] for node in network_b])

    Path.mkdir(output_path, parents=True, exist_ok=True)

    write_key(output_path.joinpath("key.dat"), deleted_nodes_a, network_a)

    network_b = ordered_cnxs(network_b, network_a, "net")
    network_b = ordered_cnxs(network_b, network_b, "dual")

    deleted_nodes_a.sort(reverse=True)

    input_aux_a = NetMCAux.from_file(input_path.joinpath("test_a_aux.dat"))
    output_aux_a = NetMCAux(num_nodes_a, max_cnxs_a, max_cnxs_b,
                                geom_code="2DE", xlo=input_aux_a.xlo, xhi=input_aux_a.xhi, ylo=input_aux_a.ylo,
                                yhi=input_aux_a.yhi, network_type=NetworkType.A, prefix="testA")
    output_aux_a.export(output_path, prefix="testA")
    export_crds(output_path.joinpath("testA_a_crds.dat"), network_a)

    network_a_net = NetMCNets.from_array(NetworkType.A, NetworkType.A, [node["net"] for node in network_a])
    network_a_net.shift_export(output_path, deleted_nodes_a, prefix="testA")

    network_a_dual = NetMCNets.from_array(NetworkType.A, NetworkType.B, [node["dual"] for node in network_a])
    deleted_nodes_b.sort(reverse=True)
    network_a_dual.shift_export(output_path, deleted_nodes_b, prefix="testA")
    input_aux_b = NetMCAux.from_file(input_path.joinpath("test_b_aux.dat"))
    output_aux_b = NetMCAux(num_nodes_b, max_cnxs_b, max_cnxs_a, geom_code="2DE",
                            xlo=input_aux_b.xlo, xhi=input_aux_b.xhi, ylo=input_aux_b.ylo,
                            yhi=input_aux_b.yhi, network_type=NetworkType.B, prefix="testA")
    output_aux_b.export(output_path, prefix="testA")

    export_crds(output_path.joinpath("testA_b_crds.dat"), network_b)

    network_b_net = NetMCNets.from_array(NetworkType.B, NetworkType.B, [node["net"] for node in network_b])
    network_b_net.shift_export(output_path, deleted_nodes_b, prefix="testA")
    network_b_dual = NetMCNets.from_array(NetworkType.B, NetworkType.A, [node["dual"] for node in network_b])
    network_b_dual.special_export(output_path, deleted_nodes_a, prefix="testA")

    with open(output_path.joinpath("fixed_rings.dat"), 'w') as fixed_rings_file:
        fixed_rings_file.write("1\n")
        fixed_rings_file.write(f"{new_ring}\n")


def get_folder_name(network: Network, lj: bool) -> str:
    max_cnxs = max([node["net"].shape[0] for node in network])
    num_nodes = len(network)
    output_folder_name = f"{max_cnxs}_{num_nodes}_"
    for node, info in network.values():
        num_node_cnxs = info["net"].shape[0]
        if num_node_cnxs < 6:
            output_folder_name += num_node_cnxs
    if lj:
        output_folder_name += "_LJ"
    return output_folder_name


class DrawLineWidget:
    def __init__(self, output_path):
        self.original_image = cv2.imread("bg.png")
        self.clone = copy.deepcopy(self.original_image)
        self.new_ring = 0
        self.lj = 1
        self.network_a, self.network_b, self.graph_a, self.graph_b = network_a, network_b, graph_a, graph_b
        print("DLW")
        print(self.graph_a)
        self.deleted_nodes, self.undercoordinated, self.broken_rings, self.rings_to_remove = [], [], [], []
        cv2.namedWindow("image")
        print("Before EC")
        print(graph_a)
        cv2.setMouseCallback("image", self.extract_coordinates)
        print("After EC")
        print(graph_a)
        # List to store start/end points
        self.image_coordinates = []

        # local variables to help recenter
        # dimensions of box = 1700x1000

        global scale, X_OFFSET, Y_OFFSET
        scale = int(1000 / np.sqrt(len(network_b)))
        refresh_new_cnxs(self.network_a, self.network_b, self.graph_a, self.graph_b,
                         [], self.undercoordinated, self.clone)

    def check(self):
        print("Checking for broken nodes...\n")
        broken_nodes = []
        for selected_node in self.network_a:
            if selected_node not in self.network_a:
                print(f"Node not found in network_a: {selected_node}")
            for bonded_node in self.network_a[selected_node]["net"]:
                if bonded_node not in self.network_a:
                    print(f"Bonded node not found in network_a: {bonded_node}")
                if selected_node not in self.network_a[bonded_node]["net"]:
                    print("#" * 40)
                    print("Selected node not found in bonded node's network")
                    print(f"selected_node: {selected_node}\tbonded_node: {bonded_node}")
                    print(f"selected_node['net']: {selected_node['net']}\tbonded_node['net']: {bonded_node['net']}\n")
                    broken_nodes.append(selected_node)
                    broken_nodes.append(bonded_node)
        if len(broken_nodes) == 0:
            print("No broken nodes detected!\n")
        else:
            print("Broken nodes detected, exiting...")
            exit(1)

    def extract_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("EC")
            print(self.graph_a)
            self.network_a, self.network_b, self.deleted_nodes, self.broken_rings, self.undercoordinated = left_click(np.array([x, y]), self.graph_a, self.network_a, self.network_b, self.deleted_nodes,
                                                                                                                      self.broken_rings, self.undercoordinated, self.clone)
        if event == cv2.EVENT_RBUTTONDOWN:
            print("Recombining...")
            # Check that these two lists don't overlap!
            self.undercoordinated = remove_deleted_nodes(self.deleted_nodes, self.undercoordinated)

            # there are always two connection patterns
            self.network_a_copy = copy.deepcopy(self.network_a)
            self.network_b_copy = copy.deepcopy(self.network_b)

            # There is more than one route round a ring !
            starting_node = self.undercoordinated[0]

            # Add a graphical aid to show new connections
            atoms = np.array([starting_node])

            # Visualise
            self.clone = self.original_image.copy()
            self.refresh_new_cnxs(atoms)
            cv2.imshow("image", self.clone)
            paths = find_shared_cnxs(starting_node, self.undercoordinated, self.network_a)
            atom_a, atom_b = -1, -1
            if check_path(starting_node, self.undercoordinated[paths[0]], self.network_a_copy, self.undercoordinated):
                atom_a = self.undercoordinated[paths[0]]
            if len(paths) > 1:
                if check_path(starting_node, self.undercoordinated[paths[1]], self.network_a_copy, self.undercoordinated):
                    if atom_a == -1:
                        atom_a = self.undercoordinated[paths[1]]
                    elif atom_a != -1:
                        atom_b = self.undercoordinated[paths[1]]
            if len(paths) > 2:
                if check_path(starting_node, self.undercoordinated[paths[2]], self.network_a_copy, self.undercoordinated):
                    if atom_a == -1:
                        atom_a = self.undercoordinated[paths[2]]
                    elif atom_b == -1:
                        atom_b = self.undercoordinated[paths[2]]

            self.clone = self.original_image.copy()
            self.refresh_new_cnxs(atoms)
            cv2.imshow("image", self.clone)
            cv2.imshow('image', draw_line_widget.show_image())
            cv2.waitKey(1)
            print('############ Initial Broken Rings : ', self.broken_rings)
            print('>>>>>>>>>>>> Initial Undercoordinated : ', self.undercoordinated)
            # SPLIT HERE TO N OPTIONS

            if check_path(starting_node, atom_a, self.network_a_copy, self.undercoordinated):
                local_nodes, local_undercoordinated, local_broken_rings = create_initial_path(starting_node, atom_a, atoms,
                                                                                              self.network_a, self.network_b,
                                                                                              self.undercoordinated, self.broken_rings)
                
                self.clone = self.original_image.copy()
                self.refresh_new_cnxs(atoms)
                cv2.imshow("image", self.clone)
                cv2.imshow('image', draw_line_widget.show_image())
                cv2.waitKey(1)
                print('############ One Connection Broken Rings : ', self.broken_rings)
                print('>>>>>>>>>>>> One Connection Undercoordinated : ', self.undercoordinated)
                self.network_a, self.network_b, self.new_ring, self.broken_rings, self.atoms = create_secondary_path(starting_node, atom_a, local_undercoordinated, local_broken_rings,
                                                                                                                     self.network_a, self.network_b_copy, self.original_image, self.atoms,
                                                                                                                     self.graph_a, self.graph_b, self.clone)
                create_secondary_path(starting_node, atom_a, local_undercoordinated, local_broken_rings, self.network_a, self.network_b_copy)
                self.refresh_new_cnxs(atoms)
                cv2.imshow("image", self.clone)
                cv2.imshow('image', draw_line_widget.show_image())
                cv2.waitKey(1)
                output_folder_name = get_folder_name(local_nodes, self.lj)
                write(output_path.joinpath(output_folder_name), input_path, local_nodes, self.network_b,
                            self.deleted_nodes, self.rings_to_remove, self.new_ring)
                make_crds_marks_bilayer(output_folder_name, self.lj, True, True)

    def show_image(self):
        return self.clone


def make_crds_marks_bilayer(folder, intercept_2, triangle_raft, bilayer, common_files_path, output_path):
    area = 1.00
    intercept_1 = intercept_2

    # NAMING

    AREA_SCALING = np.sqrt(area)
    UNITS_SCALING = 1 / 0.52917721090380
    si_si_distance = UNITS_SCALING * 1.609 * np.sqrt((32.0 / 9.0))
    si_o_length = UNITS_SCALING * 1.609 
    o_o_distance = UNITS_SCALING * 1.609 * np.sqrt((8.0 / 3.0))
    h = UNITS_SCALING * np.sin((19.5 / 180) * np.pi) * 1.609

    displacement_vectors_norm = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 3]])
    displacement_vectors_factored = displacement_vectors_norm * 0.5

    with open(folder + '/testA_a_aux.dat', 'r') as f:
        n_nodes = np.genfromtxt(f, max_rows=1)
        n_nodes = int(n_nodes)
    with open(folder + '/testA_a_aux.dat', 'r') as f:
        dims = np.genfromtxt(f, skip_header=3, skip_footer=1)
        dim_x, dim_y = dims[0], dims[1]

    dim = np.array([dim_x, dim_y, 30])
    with open(folder + '/testA_a_net.dat', 'r') as f:
        net = np.genfromtxt(f)

    with open(folder + '/testA_crds_a.dat', 'r') as f:
        node_crds = np.genfromtxt(f)

    with open(folder + '/testA_b_crds.dat', 'r') as f:
        dual_crds = np.genfromtxt(f)
    number_scaling = np.sqrt(dual_crds.shape[0] / num_nodes_b)
    print(dim_x, dim_y)
    dim_x, dim_y = number_scaling * dim_x, number_scaling * dim_y
    print(dim_x, dim_y)
    print(dual_crds.shape[0], num_nodes_b)
    print(number_scaling)
    dim = np.array([dim_x, dim_y, 30])

    node_crds = np.multiply(node_crds, number_scaling)

    def pbc_v(i, j):
        v = np.subtract(j, i)
        for dimension in range(2):
            if v[dimension] < -dim[dimension] / 2:
                v[dimension] += dim[dimension]
            elif v[dimension] > dim[dimension] / 2:
                v[dimension] -= dim[dimension]

        return v
    # Monolayer
    monolayer_crds = np.multiply(node_crds, 1)

    for i in range(n_nodes):
        atom_1 = i
        for j in range(3):
            atom_2 = net[i, j]

            if i == 0 and j == 0:
                monolayer_harmpairs = np.asarray([int(atom_1), int(atom_2)])
            else:
                if atom_2 > atom_1:
                    monolayer_harmpairs = np.vstack((monolayer_harmpairs, np.asarray([int(atom_1), int(atom_2)])))

    for i in range(n_nodes):
        atom_1 = i
        if atom_1 == 0:
            monolayer_angles = np.asarray([[net[i, 0], i, net[i, 1]],
                                           [net[i, 0], i, net[i, 2]],
                                           [net[i, 1], i, net[i, 2]]])
        else:
            monolayer_angles = np.vstack((monolayer_angles, np.asarray([[net[i, 0], i, net[i, 1]],
                                                                        [net[i, 0], i, net[i, 2]],
                                                                        [net[i, 1], i, net[i, 2]]])))

    print(f"Monolayer n {monolayer_crds.shape[0]}")
    print(f"Monolayer harmpairs {monolayer_harmpairs.shape[0]}")

    with open(folder + '/PARM_Si.lammps', 'w') as f:
        f.write('bond_style harmonic        \n')
        f.write('bond_coeff 1 0.800 1.000  \n')
        f.write('angle_style cosine/squared       \n')
        f.write('angle_coeff 1 0.200 120   \n')

    with open(folder + '/Si.data', 'w') as f:
        f.write('DATA FILE Produced from netmc results (cf David Morley)\n')
        f.write(f"{monolayer_crds.shape[0]} atoms\n")
        f.write(f"{monolayer_harmpairs.shape[0]} bonds\n")
        f.write(f"{monolayer_angles.shape[0]} angles\n")
        f.write('0 dihedrals\n')
        f.write('0 impropers\n')
        f.write('1 atom types\n')
        f.write('1 bond types\n')
        f.write('1 angle types\n')
        f.write('0 dihedral types\n')
        f.write('0 improper types\n')
        f.write('0.00000 {:<5} xlo xhi\n'.format(dim[0]))
        f.write('0.00000 {:<5} ylo yhi\n'.format(dim[1]))
        f.write('\n')
        f.write('# Pair Coeffs\n')
        f.write('#\n')
        f.write('# 1  Si\n')
        f.write('\n')
        f.write('# Bond Coeffs\n')
        f.write('# \n')
        f.write('# 1  Si-Si\n')
        f.write('\n')
        f.write('# Angle Coeffs\n')
        f.write('# \n')
        f.write('# 1  Si-Si-Si\n')
        f.write('\n')
        f.write(' Masses\n')
        f.write('\n')
        f.write('1 28.085500 # Si\n')
        f.write('\n')
        f.write(' Atoms # molecular\n')
        f.write('\n')
        for i in range(monolayer_crds.shape[0]):
            f.write('{:<4} {:<4} {:<4} {:<24} {:<24} {:<24}# Si\n'.format(int(i + 1), int(i + 1), 1, monolayer_crds[i, 0], monolayer_crds[i, 1], 0.0))
        f.write('\n')
        f.write(' Bonds\n')
        f.write('\n')
        for i in range(monolayer_harmpairs.shape[0]):
            f.write('{:} {:} {:} {:}\n'.format(int(i + 1), 1, int(monolayer_harmpairs[i, 0] + 1), int(monolayer_harmpairs[i, 1] + 1)))
        f.write('\n')
        f.write(' Angles\n')
        f.write('\n')
        for i in range(monolayer_angles.shape[0]):
            f.write('{:} {:} {:} {:} {:}\n'.format(int(i + 1), 1, int(monolayer_angles[i, 0] + 1), int(monolayer_angles[i, 1] + 1), int(monolayer_angles[i, 2] + 1)))
    shutil.copyfile(common_files_path.joinpath("Si.in"), output_path.joinpath("Si.in"))

    # Tersoff Graphene

    print("########### Tersoff Graphene ###############")
    tersoff_crds = np.multiply(node_crds, 1.42)
    with open(folder + '/PARM_C.lammps', 'w') as f:
        f.write('pair_style tersoff\n')
        f.write('pair_coeff * * Results/BNC.tersoff C\n')
    shutil.copyfile(common_files_path.joinpath("C.in"), output_path.joinpath("C.in"))

    with open(folder + '/C.data', 'w') as f:
        f.write('DATA FILE Produced from netmc results (cf David Morley)\n')
        f.write('{:} atoms\n'.format(tersoff_crds.shape[0]))
        f.write('1 atom types\n')
        f.write('0.00000 {:<5} xlo xhi\n'.format(dim[0] * 1.42))
        f.write('0.00000 {:<5} ylo yhi\n'.format(dim[1] * 1.42))
        f.write('\n')
        f.write(' Masses\n')
        f.write('\n')
        f.write('1 12.0000 # Si\n')
        f.write('\n')
        f.write(' Atoms # molecular\n')
        f.write('\n')
        for i in range(tersoff_crds.shape[0]):
            f.write('{:<4} {:<4} {:<24} {:<24} {:<24}# C\n'.format(int(i + 1), 1,
                                                                   tersoff_crds[i, 0],
                                                                   tersoff_crds[i, 1], 0.0))
        f.write('\n')

    # Triangle Raft

    if triangle_raft:

        print("########### Triangle Raft ##############")
        dim[0] *= si_si_distance * AREA_SCALING
        dim[1] *= si_si_distance * AREA_SCALING

        dim_x *= si_si_distance * AREA_SCALING
        dim_y *= si_si_distance * AREA_SCALING
        triangle_raft_si_crds = np.multiply(monolayer_crds, si_si_distance * AREA_SCALING)
        dict_sio = {}
        for i in range(int(n_nodes * 3 / 2), int(n_nodes * 5 / 2)):
            dict_sio['{:}'.format(i)] = []
        for i in range(monolayer_harmpairs.shape[0]):
            atom_1 = int(monolayer_harmpairs[i, 0])
            atom_2 = int(monolayer_harmpairs[i, 1])
            atom_1_crds = triangle_raft_si_crds[atom_1, :]
            atom_2_crds = triangle_raft_si_crds[atom_2, :]

            v = pbc_v(atom_1_crds, atom_2_crds)
            norm_v = np.divide(v, np.linalg.norm(v))

            grading = [abs(np.dot(norm_v, displacement_vectors_norm[i, :])) for i in range(displacement_vectors_norm.shape[0])]
            selection = grading.index(min(grading))
            if abs(grading[selection]) < 0.1:

                unperturbed_oxygen_0_crds = np.add(atom_1_crds, np.divide(v, 2))
                oxygen_0_crds = np.add(unperturbed_oxygen_0_crds, displacement_vectors_factored[selection])

            else:
                oxygen_0_crds = np.add(atom_1_crds, np.divide(v, 2))

            if oxygen_0_crds[0] > dim_x:
                oxygen_0_crds[0] -= dim_x
            elif oxygen_0_crds[0] < 0:
                oxygen_0_crds[0] += dim_x
            if oxygen_0_crds[1] > dim_y:
                oxygen_0_crds[1] -= dim_y
            elif oxygen_0_crds[1] < 0:
                oxygen_0_crds[1] += dim_y

            if i == 0:
                triangle_raft_o_crds = np.asarray(oxygen_0_crds)
                triangle_raft_harmpairs = np.asarray([[i, atom_1 + n_nodes * 3 / 2],
                                                      [i, atom_2 + n_nodes * 3 / 2]])
                dict_sio['{:}'.format(int(atom_1 + n_nodes * 3 / 2))].append(i)
                dict_sio['{:}'.format(int(atom_2 + n_nodes * 3 / 2))].append(i)
            else:
                triangle_raft_o_crds = np.vstack((triangle_raft_o_crds, oxygen_0_crds))
                triangle_raft_harmpairs = np.vstack((triangle_raft_harmpairs, np.asarray([[i, atom_1 + n_nodes * 3 / 2],
                                                                                          [i, atom_2 + n_nodes * 3 / 2]])))
                dict_sio['{:}'.format(int(atom_1 + n_nodes * 3 / 2))].append(i)
                dict_sio['{:}'.format(int(atom_2 + n_nodes * 3 / 2))].append(i)

        for i in range(int(n_nodes * 3 / 2), int(n_nodes * 5 / 2)):
            for j in range(2):
                for k in range(j + 1, 3):
                    if i == int(n_nodes * 3 / 2) and j == 0 and k == 1:
                        triangle_raft_o_harmpairs = np.array([dict_sio['{:}'.format(i)][j], dict_sio['{:}'.format(i)][k]])
                    else:
                        triangle_raft_o_harmpairs = np.vstack((triangle_raft_o_harmpairs, np.array([dict_sio['{:}'.format(i)][j], dict_sio['{:}'.format(i)][k]])))
                    triangle_raft_harmpairs = np.vstack((triangle_raft_harmpairs, np.array([dict_sio['{:}'.format(i)][j], dict_sio['{:}'.format(i)][k]])))

        triangle_raft_crds = np.vstack((triangle_raft_o_crds, triangle_raft_si_crds))

        for i in range(triangle_raft_crds.shape[0]):
            for j in range(2):
                if triangle_raft_crds[i, j] > dim[j] or triangle_raft_crds[i, j] < 0:
                    print('FUCK')

        print('Triangle Raft n {:}    si {:}    o {:}'.format(triangle_raft_crds.shape[0], triangle_raft_si_crds.shape[0], triangle_raft_o_crds.shape[0]))
        print('Triangle Raft harmpairs : {:}'.format(triangle_raft_harmpairs.shape[0]))

        def plot_triangle_raft():
            plt.scatter(triangle_raft_si_crds[:, 0], triangle_raft_si_crds[:, 1], color='y', s=0.4)
            plt.scatter(triangle_raft_o_crds[:, 0], triangle_raft_o_crds[:, 1], color='r', s=0.4)
            plt.savefig('triangle_raft atoms')
            plt.clf()
            plt.scatter(triangle_raft_si_crds[:, 0], triangle_raft_si_crds[:, 1], color='y', s=0.6)
            plt.scatter(triangle_raft_o_crds[:, 0], triangle_raft_o_crds[:, 1], color='r', s=0.6)
            print(triangle_raft_harmpairs.shape)
            for i in range(triangle_raft_harmpairs.shape[0]):

                atom_1 = int(triangle_raft_harmpairs[i, 0])
                atom_2 = int(triangle_raft_harmpairs[i, 1])
                if atom_1 < triangle_raft_o_crds.shape[0] and atom_2 < triangle_raft_o_crds.shape[0]:
                    atom_1_crds = triangle_raft_crds[atom_1, :]
                    atom_2_crds = triangle_raft_crds[atom_2, :]
                    atom_2_crds = np.add(atom_1_crds, pbc_v(atom_1_crds, atom_2_crds))
                    plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[1], atom_2_crds[1]], color='k')
            for i in range(triangle_raft_harmpairs.shape[0]):

                atom_1 = int(triangle_raft_harmpairs[i, 0])
                atom_2 = int(triangle_raft_harmpairs[i, 1])
                if atom_1 < triangle_raft_o_crds.shape[0] and atom_2 < triangle_raft_o_crds.shape[0]:
                    atom_1_crds = np.add(triangle_raft_crds[atom_1, :], np.array([0, dim[1]]))
                    atom_2_crds = np.add(triangle_raft_crds[atom_2, :], np.array([0, dim[1]]))
                    atom_2_crds = np.add(atom_1_crds, pbc_v(atom_1_crds, atom_2_crds))
                    plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[1], atom_2_crds[1]], color='k')

            plt.savefig('triangle raft bonds')
            plt.clf()
        n_bonds = triangle_raft_harmpairs.shape[0]

        n_bond_types = 2

        with open(folder + '/PARM_Si2O3.lammps', 'w') as output_file:

            output_file.write('pair_style lj/cut {:}\n'.format(o_o_distance * intercept_1))
            output_file.write('pair_coeff * * 0.1 {:} {:}\n'.format(o_o_distance * intercept_1 / 2**(1 / 6), o_o_distance * intercept_1))
            output_file.write('pair_modify shift yes\n'.format())
            output_file.write('special_bonds lj 0.0 1.0 1.0\n'.format())

            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 2.86667626014\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')
        shutil.copyfile(common_files_path.joinpath("Si2O3.in"), output_path.joinpath("Si2O3.in"))

        with open(folder + '/Si2O3.data', 'w') as f:
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
            f.write('0.00000 {:<5} xlo xhi\n'.format(dim[0]))
            f.write('0.00000 {:<5} ylo yhi\n'.format(dim[1]))
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
                                                                              int(i + 1 + triangle_raft_si_crds.shape[0]),
                                                                              1, triangle_raft_o_crds[i, 0],
                                                                              triangle_raft_o_crds[i, 1], 5.0))

            f.write('\n')
            f.write(' Bonds\n')
            f.write('\n')
            for i in range(triangle_raft_harmpairs.shape[0]):
                pair1 = triangle_raft_harmpairs[i, 0]
                if pair1 < triangle_raft_o_crds.shape[0]:
                    pair1_ref = pair1 + 1 + triangle_raft_si_crds.shape[0]
                else:
                    pair1_ref = pair1 + 1 - triangle_raft_o_crds.shape[0]
                pair2 = triangle_raft_harmpairs[i, 1]
                if pair2 < triangle_raft_o_crds.shape[0]:
                    pair2_ref = pair2 + 1 + triangle_raft_si_crds.shape[0]
                else:
                    pair2_ref = pair2 + 1 - triangle_raft_o_crds.shape[0]

                if triangle_raft_harmpairs[i, 0] < triangle_raft_o_crds.shape[0] and triangle_raft_harmpairs[i, 1] < triangle_raft_o_crds.shape[0]:

                    f.write('{:} {:} {:} {:} \n'.format(int(i + 1), 1, int(pair1_ref),
                                                        int(pair2_ref)))
                else:

                    f.write('{:} {:} {:} {:} \n'.format(int(i + 1), 2, int(pair1_ref),
                                                        int(pair2_ref)))

        with open(folder + '/Si2O3_harmpairs.dat', 'w') as f:
            f.write('{:}\n'.format(triangle_raft_harmpairs.shape[0]))
            for i in range(triangle_raft_harmpairs.shape[0]):
                if triangle_raft_harmpairs[i, 0] < triangle_raft_o_crds.shape[0] and triangle_raft_harmpairs[i, 1] < triangle_raft_o_crds.shape[0]:
                    f.write('{:<10} {:<10} \n'.format(int(triangle_raft_harmpairs[i, 0] + 1),
                                                      int(triangle_raft_harmpairs[i, 1] + 1)))
                else:
                    f.write('{:<10} {:<10} \n'.format(int(triangle_raft_harmpairs[i, 0] + 1),
                                                      int(triangle_raft_harmpairs[i, 1] + 1)))

    if bilayer:
        def triangle_raft_to_bilayer(i):
            if i > 3 * n_nodes / 2:
                # Si atom
                si_ref = i - 3 * n_nodes / 2
                return [4 * n_nodes + 2 * si_ref, 4 * n_nodes + 2 * si_ref + 1]
            else:
                # O atom
                o_ref = i
                return [n_nodes + 2 * o_ref, n_nodes + 2 * o_ref + 1]

        # Bilayer
        print("############ Bilayer ###############")

        # Si Atoms
        for i in range(triangle_raft_si_crds.shape[0]):
            if i == 0:
                bilayer_si_crds = np.asarray([[triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5],
                                              [triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + 2 * si_o_length]])
            else:
                bilayer_si_crds = np.vstack((bilayer_si_crds, np.asarray([[triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5],
                                                                          [triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + 2 * si_o_length]])))
        # O ax Atoms
        for i in range(triangle_raft_si_crds.shape[0]):
            if i == 0:
                bilayer_o_crds = np.asarray([triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + si_o_length])
            else:
                bilayer_o_crds = np.vstack((bilayer_o_crds, np.asarray([triangle_raft_si_crds[i, 0], triangle_raft_si_crds[i, 1], 5 + si_o_length])))
        # O eq
        for i in range(triangle_raft_o_crds.shape[0]):
            bilayer_o_crds = np.vstack((bilayer_o_crds, np.asarray([triangle_raft_o_crds[i, 0], triangle_raft_o_crds[i, 1], 5 - h])))
            bilayer_o_crds = np.vstack((bilayer_o_crds, np.asarray([triangle_raft_o_crds[i, 0], triangle_raft_o_crds[i, 1], 5 + h + 2 * si_o_length])))

        bilayer_crds = np.vstack((bilayer_o_crds, bilayer_si_crds))

        dict_sio2 = {}

        # Harmpairs
        # O ax
        for i in range(triangle_raft_si_crds.shape[0]):
            if i == 0:
                bilayer_harmpairs = np.asarray([[i, 4 * n_nodes + 2 * i],  # 3200
                                                [i, 4 * n_nodes + 1 + 2 * i],  # 3201
                                                [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][0])[0]],
                                                [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][0])[1]],
                                                [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][1])[0]],
                                                [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][1])[1]],
                                                [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][2])[0]],
                                                [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][2])[1]]]
                                               )
            else:
                bilayer_harmpairs = np.vstack((bilayer_harmpairs, np.asarray([[i, 4 * n_nodes + 2 * i],  # 3200
                                                                              [i, 4 * n_nodes + 1 + 2 * i],  # 3201
                                                                              [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][0])[0]],
                                                                              [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][0])[1]],
                                                                              [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][1])[0]],
                                                                              [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][1])[1]],
                                                                              [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][2])[0]],
                                                                              [i, triangle_raft_to_bilayer(dict_sio['{:}'.format(int(3 * n_nodes / 2 + i))][2])[1]]])))
        # Si - O cnxs
        for i in range(triangle_raft_harmpairs.shape[0]):
            atom_1 = triangle_raft_to_bilayer(triangle_raft_harmpairs[i, 0])
            atom_2 = triangle_raft_to_bilayer(triangle_raft_harmpairs[i, 1])

            bilayer_harmpairs = np.vstack((bilayer_harmpairs, np.asarray([[atom_1[0], atom_2[0]], [atom_1[1], atom_2[1]]])))

        for vals in dict_sio.keys():
            dict_sio2['{:}'.format(int(vals) - 3 * n_nodes / 2 + 4 * n_nodes)] = [triangle_raft_to_bilayer(dict_sio["{:}".format(vals)][i]) for i in range(3)]

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
                atom_2_crds = np.add(atom_1_crds, pbc_v(atom_1_crds, atom_2_crds))
                if int(bilayer_harmpairs[i, 0]) >= 4 * n_nodes or int(bilayer_harmpairs[i, 1]) >= 4 * n_nodes:
                    plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[1], atom_2_crds[1]], color='k')
            plt.title('Si-O')
            plt.savefig('bilayer SiO bond')
            plt.clf()
            plt.scatter(bilayer_si_crds[:, 0], bilayer_si_crds[:, 1], color='y', s=0.4)
            plt.scatter(bilayer_o_crds[:, 0], bilayer_o_crds[:, 1], color='r', s=0.4)
            for i in range(bilayer_harmpairs.shape[0]):
                atom_1_crds = bilayer_crds[int(bilayer_harmpairs[i, 0]), :]
                atom_2_crds = bilayer_crds[int(bilayer_harmpairs[i, 1]), :]
                atom_2_crds = np.add(atom_1_crds, pbc_v(atom_1_crds, atom_2_crds))
                if int(bilayer_harmpairs[i, 0]) < 4 * n_nodes and int(bilayer_harmpairs[i, 1]) < 4 * n_nodes:
                    plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[1], atom_2_crds[1]], color='k')
            plt.title('O-O')
            plt.savefig('bilayer OO bond')
            plt.clf()

            plt.scatter(bilayer_si_crds[:, 0], bilayer_si_crds[:, 2], color='y', s=0.4)
            plt.scatter(bilayer_o_crds[:, 0], bilayer_o_crds[:, 2], color='r', s=0.4)
            for i in range(bilayer_harmpairs.shape[0]):
                atom_1_crds = bilayer_crds[int(bilayer_harmpairs[i, 0]), :]
                atom_2_crds = bilayer_crds[int(bilayer_harmpairs[i, 1]), :]
                atom_2_crds = np.add(atom_1_crds, pbc_v(atom_1_crds, atom_2_crds))
                plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[2], atom_2_crds[2]], color='k')

            plt.savefig('bilayer all')
            plt.clf()

        plot_bilayer()

        n_bonds = bilayer_harmpairs.shape[0]

        with open(folder + '/PARM_SiO2.lammps', 'w') as output_file:
            output_file.write('pair_style lj/cut {:}\n'.format(o_o_distance * intercept_1))
            output_file.write('pair_coeff * * 0.1 {:} {:}\n'.format(o_o_distance * intercept_1 / 2**(1 / 6), o_o_distance * intercept_1))
            output_file.write('pair_modify shift yes\n'.format())
            output_file.write('special_bonds lj 0.0 1.0 1.0\n'.format())

            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 3.0405693345182674\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

        with open(folder + '/SiO2.data', 'w') as f:
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
            f.write('0.00000 {:<5} xlo xhi\n'.format(dim[0]))
            f.write('0.00000 {:<5} ylo yhi\n'.format(dim[1]))
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
                                                                              int(i + 1 + bilayer_si_crds.shape[0]),
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
                    f.write('{:} {:} {:} {:}\n'.format(int(i + 1), 1, int(pair1_ref), int(pair2_ref)))
                else:
                    f.write('{:} {:} {:} {:}\n'.format(int(i + 1), 2, int(pair1_ref), int(pair2_ref)))

        with open(folder + '/SiO2_harmpairs.dat', 'w') as f:
            f.write('{:}\n'.format(bilayer_harmpairs.shape[0]))
            for i in range(bilayer_harmpairs.shape[0]):
                if bilayer_harmpairs[i, 0] < bilayer_o_crds.shape[0] and bilayer_harmpairs[i, 1] < bilayer_o_crds.shape[0]:
                    f.write('{:<10} {:<10}\n'.format(int(bilayer_harmpairs[i, 0] + 1), int(bilayer_harmpairs[i, 1] + 1)))
                else:
                    f.write('{:<10} {:<10}\n'.format(int(bilayer_harmpairs[i, 0] + 1), int(bilayer_harmpairs[i, 1] + 1)))

        shutil.copyfile(common_files_path.joinpath("SiO2.in"), output_path.joinpath("SiO2.in"))
    print('Finished')


if __name__ == '__main__':
    cwd = Path(__file__).parent
    common_files_path = cwd.joinpath(cwd, "common_files")
    input_folder = "t-3200_s0_same"
    input_path = cwd.joinpath(input_folder)
    output_path = cwd.joinpath("Results")
    prefix = find_prefix(input_path)
    crds_a, net_a, dual_a = read_netmc_files(input_path, f"{prefix}_A")
    crds_b, net_b, dual_b = read_netmc_files(input_path, f"{prefix}_B")

    num_nodes_a = crds_a.shape[0]
    num_nodes_b = crds_b.shape[0]
    network_a = {i: {"crds": crds_a[i], "net": net_a[i], "dual": dual_a[i]} for i in np.arange(num_nodes_a)}
    network_b = {i: {"crds": crds_b[i], "net": net_b[i], "dual": dual_b[i]} for i in np.arange(num_nodes_b)}
    graph_a = get_graph(network_a)
    graph_b = get_graph(network_b)

    draw_line_widget = DrawLineWidget(output_path)

    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)
