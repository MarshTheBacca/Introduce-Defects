import copy
import shutil
from enum import Enum
from pathlib import Path
from sys import exit
from typing import Callable, TypeAlias

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import (LAMMPSAngle, LAMMPSAtom, LAMMPSBond, LAMMPSData, NetMCAux,
                   NetMCData, NetMCNets, NetworkType)

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

# Constants
ANGSTROM_TO_BOHR = 1 / 0.52917721090380
O_O_DISTANCE_FACTOR = np.sqrt(8.0 / 3.0)
SI_O_LENGTH_ANGSTROM = 1.609
H_ANGLE_DEGREES = 19.5

# Scaling factors
SI_O_LENGTH_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM
O_O_DISTANCE_BOHR = ANGSTROM_TO_BOHR * SI_O_LENGTH_ANGSTROM * O_O_DISTANCE_FACTOR
H_BOHR = ANGSTROM_TO_BOHR * np.sin(np.radians(H_ANGLE_DEGREES)) * SI_O_LENGTH_ANGSTROM


class ConnectionType(Enum):
    net: str = "net"
    dual: str = "dual"


CONNECTION_TYPE_MAP = {ConnectionType.net: "net", ConnectionType.dual: "dual"}
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
        undercoordinated_nodes = np.append(undercoordinated_nodes, node)
        network_a[node]["net"] = np_remove_elements(network_a[node]["net"], close_node)
        if node in deleted_nodes:
            undercoordinated_nodes = np_remove_elements(undercoordinated_nodes, node)
    del network_a[close_node]
    refresh_new_cnxs(network_a, network_b, graph_a, graph_b,
                     np.array([]), undercoordinated_nodes, image)
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
              color: tuple[int, int, int], thickness: float) -> None:
    x_coord_0 = scale * coord_1[0] + X_OFFSET
    y_coord_0 = scale * coord_1[1] + Y_OFFSET
    x_coord_1 = scale * coord_2[0] + X_OFFSET
    y_coord_1 = scale * coord_2[1] + Y_OFFSET
    cv2.line(image, (int(x_coord_0), int(y_coord_0)),
             (int(x_coord_1), int(y_coord_1)), color, thickness)


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
            draw_circle(image, network_a[node]["crds"], 15, (36, 255, 12), 1)
        else:
            print("Uncoordinated Atoms Doesnt Exist")
    for atom in atoms:
        if atom in network_a:
            atom_index = np.nonzero(atoms == atom)[0][0]
            r, g, b = (int(5 + atom_index * 20)) % 255, 36, int(abs(255 - atom_index * 20))
            draw_circle(image, network_a[atom]["crds"], 10, (r, g, b), 1)
        else:
            print("Starting From atom doesn't Exist!")

    for i in range(0, len(atoms) - 1, 2):
        draw_line(image, network_a[atoms[i]]["crds"],
                  network_a[atoms[i + 1]]["crds"], (1, 1, 1), 5)


def np_remove_elements(array: np.ndarray, elements_to_remove: np.ndarray) -> np.ndarray:
    elements_to_remove = np.array([elements_to_remove]) if np.isscalar(elements_to_remove) else elements_to_remove
    print(array, elements_to_remove)
    mask = np.logical_not(np.isin(array, elements_to_remove))
    print(mask)
    return array[mask]


def find_nodes_connections(nodes: np.ndarray, network: Network, connection_type: ConnectionType) -> np.ndarray:
    connected_nodes = {connected_node for node in nodes for connected_node in network[node][CONNECTION_TYPE_MAP[connection_type]]}
    return np.array(list(connected_nodes))


def remove_connections(network: Network, nodes: np.ndarray,
                       connections_to_remove: np.ndarray, connection_type: ConnectionType) -> Network:
    for node in nodes:
        network[node][CONNECTION_TYPE_MAP[connection_type]] = np_remove_elements(network[node][CONNECTION_TYPE_MAP[connection_type]],
                                                                                 connections_to_remove)
    return network


def add_connections(network: Network, nodes: np.ndarray, connections_to_add: np.ndarray,
                    connection_type: ConnectionType) -> Network:
    for node in nodes:
        network[node][CONNECTION_TYPE_MAP[connection_type]] = np.append(network[node][CONNECTION_TYPE_MAP[connection_type]],
                                                                        connections_to_add)
        network[node][CONNECTION_TYPE_MAP[connection_type]] = np.unique(network[node][CONNECTION_TYPE_MAP[connection_type]])
    return network


def replace_node_ring_connections(network: Network, rings_to_merge: np.ndarray,
                                  merged_ring: int) -> Network:
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


def find_paths(node: int, potential_node_cnxs: np.ndarray, network: Network) -> int:
    paths = find_shared_cnxs(node, potential_node_cnxs, network)
    if not paths:
        raise ValueError("No paths found")
    return potential_node_cnxs[paths[0]]


def create_secondary_path(node: int, undercoordinated_nodes: np.ndarray,
                          broken_rings: np.ndarray, network_a: Network, network_b: Network,
                          background: np.ndarray, atoms: np.ndarray,
                          graph_a: nx.graph, graph_b: nx.graph, clone: np.ndarray) -> tuple[Network, Network, int, np.ndarray, np.ndarray]:
    while undercoordinated_nodes:
        # Check if the newly formed connection is to a site with no further connections
        if node not in undercoordinated_nodes:
            # If so, we travel around to the next undercoordinated atom ...
            node = find_paths(node, undercoordinated_nodes, network_a)
        check_undercoordinated(undercoordinated_nodes, network_a)
        node_2 = find_paths(node, undercoordinated_nodes, network_a)
        if node_2 not in network_a[node]["net"] and node not in network_a[node_2]["net"]:
            network_a[node]["net"] = np.append(network_a[node]["net"], node_2)
            network_a[node_2]["net"] = np.append(network_a[node_2]["net"], node)
            undercoordinated_nodes = np_remove_elements(undercoordinated_nodes, node)
            undercoordinated_nodes = np_remove_elements(undercoordinated_nodes, node_2)
            atoms = np.append(atoms, [node, node_2])
            refresh_new_cnxs(network_a, network_b, graph_a, graph_b, atoms, undercoordinated_nodes, clone)
            cv2.imshow("image", background)
            cv2.imshow('image', draw_line_widget.show_image())
            cv2.waitKey(0)
            for ring in broken_rings:
                if node in network_b[ring]["dual"] and node_2 in network_b[ring]["dual"]:
                    broken_rings = np_remove_elements(broken_rings, ring)
        else:
            print("Nodes Already Connected!")
        node = node_2

    new_ring, network_a, network_b = remapped_node(broken_rings, network_a, network_b)
    return network_a, network_b, new_ring, broken_rings, atoms


def create_initial_path(node_1, node_2, atoms: np.ndarray, network_1: Network, network_2: Network,
                        undercoordinated_nodes: np.ndarray,
                        broken_rings: np.ndarray) -> tuple:
    network_1[node_2]["net"] = np.append(network_1[node_2]["net"], node_1)
    network_1[node_1]["net"] = np.append(network_1[node_1]["net"], node_2)
    undercoordinated_nodes = np.setdiff1d(undercoordinated_nodes, np.array([node_1, node_2]))
    for ring in broken_rings:
        if node_1 in network_2[ring]["dual"] and node_2 in network_2[ring]["dual"]:
            broken_rings = np_remove_elements(broken_rings, ring)
    atoms = np.append(atoms, node_2)
    return network_1, undercoordinated_nodes, broken_rings


def find_shared_cnxs(node: int, undercoordinated_nodes: np.ndarray, network: Network) -> np.ndarray:
    node_dual = network[node]["dual"]
    shared_cnxs = np.array([])
    for undercoordinated_node in undercoordinated_nodes:
        if undercoordinated_node not in network:
            raise ValueError(f"Undercoordinated node {undercoordinated_node} not found in network")

        undercoordinated_node_dual = network[undercoordinated_node]["dual"]
        shared_cnxs = np.append(shared_cnxs, len(np.intersect1d(node_dual, undercoordinated_node_dual)))
    # 0 connections - More than one ring apart
    # 1 connection  - One ring apart
    # 2 connections - share an edge
    # 3 connections - same node!

    ######## Revisit this code ########
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
    return shared_cnxs


def check_undercoordinated(uncoordinated_nodes: np.ndarray, network: Network) -> None:
    print("Checking undercoordinated nodes...")
    for uncoordinated_node in uncoordinated_nodes:
        num_cnxs = network[uncoordinated_node]["net"].shape[0]
        if num_cnxs <= 2:
            continue
        if num_cnxs == 3:
            print(f"uncoordinated node's net: {network[uncoordinated_node]['net']}")
            raise ValueError(f"{uncoordinated_node} is tricoordinated, exiting...")
        elif num_cnxs > 3:
            print(f"uncoordinated node's net: {network[uncoordinated_node]['net']}")
            raise ValueError(f"{uncoordinated_node} is overcoordinated, exiting...")
    print("Undercoordinated nodes valid.")


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
    crds_array = np.array([network[node]["crds"] for node in network])
    np.savetxt(path, crds_array, fmt="%-19.6f")


def get_target_files(file_names: list, func: Callable[[str], bool]) -> list:
    return_list = []
    for name in file_names:
        if func(name):
            return_list.append(name)
    return return_list


def import_crds(path: Path) -> np.ndarray:
    with open(path, "r") as crds_file:
        return np.genfromtxt(crds_file, dtype=np.float64)


def find_prefix(path):
    file_names = [path.name for path in Path.iterdir(path) if path.is_file()]
    aux_files = get_target_files(
        file_names, lambda name: name.endswith("aux.dat"))
    crds_files = get_target_files(
        file_names, lambda name: name.endswith("crds.dat"))
    net_files = get_target_files(
        file_names, lambda name: name.endswith("net.dat"))
    dual_files = get_target_files(
        file_names, lambda name: name.endswith("dual.dat"))
    all_prefixes = [name[:-10]
                    for name in aux_files] + [name[:-11] for name in crds_files]
    all_prefixes += [name[:-10]
                     for name in net_files] + [name[:-11] for name in dual_files]
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
        network_graph.add_node(
            selected_node, pos=network[selected_node]["crds"])
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
            node_cnxs_new_crds = np.append(
                node_cnxs_new_crds, network_2[options[0]]["crds"])
        area = 0
        for i, crd_2 in enumerate(node_cnxs_new_crds):
            crd_1 = node_cnxs_new_crds[i - 1]
            area += crd_1[0] * crd_2[1] - crd_2[0] * crd_1[1]
        if area > 0:
            node_cnxs_new.flip(node_cnxs_new)
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


def write_netmc_data(output_path: Path, input_path: Path, network_a: Network, network_b: Network, deleted_nodes_a: np.ndarray,
                     deleted_nodes_b: np.ndarray, new_ring: int) -> None:
    num_nodes_a = len(network_a)
    num_nodes_b = len(network_b)
    max_cnxs_a = max([network_a[node]["net"].shape[0] for node in network_a])
    max_cnxs_b = max([network_b[node]["net"].shape[0] for node in network_b])

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

    network_a_net = NetMCNets.from_array(NetworkType.A, NetworkType.A, [network_a[node]["net"] for node in network_a])
    network_a_net.shift_export(output_path, deleted_nodes_a, prefix="testA")

    network_a_dual = NetMCNets.from_array(NetworkType.A, NetworkType.B, [network_a[node]["dual"] for node in network_a])
    deleted_nodes_b.sort(reverse=True)
    network_a_dual.shift_export(output_path, deleted_nodes_b, prefix="testA")
    input_aux_b = NetMCAux.from_file(input_path.joinpath("test_b_aux.dat"))
    output_aux_b = NetMCAux(num_nodes_b, max_cnxs_b, max_cnxs_a, geom_code="2DE",
                            xlo=input_aux_b.xlo, xhi=input_aux_b.xhi, ylo=input_aux_b.ylo,
                            yhi=input_aux_b.yhi, network_type=NetworkType.B, prefix="testA")
    output_aux_b.export(output_path, prefix="testA")

    export_crds(output_path.joinpath("testA_b_crds.dat"), network_b)

    network_b_net = NetMCNets.from_array(NetworkType.B, NetworkType.B, [network_b[node]["net"] for node in network_b])
    network_b_net.shift_export(output_path, deleted_nodes_b, prefix="testA")
    network_b_dual = NetMCNets.from_array(NetworkType.B, NetworkType.A, [network_b[node]["dual"] for node in network_b])
    network_b_dual.special_export(output_path, deleted_nodes_a, prefix="testA")

    with open(output_path.joinpath("fixed_rings.dat"), 'w') as fixed_rings_file:
        fixed_rings_file.write(f"1\n{new_ring}\n")


def get_folder_name(network: Network, lj: bool) -> str:
    max_cnxs = max([node["net"].shape[0] for node in network])
    num_nodes = len(network)
    output_folder_name = f"{max_cnxs}_{num_nodes}_"
    for info in network.values():
        num_node_cnxs = info["net"].shape[0]
        if num_node_cnxs < 6:
            output_folder_name += num_node_cnxs
    if lj:
        output_folder_name += "_LJ"
    return output_folder_name


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
    # This will only work for exactly hexagonal networks
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
            vector_between_si_atoms = pbc_vector(si_atom.coord, bonded_si_atom.coord, dimension_ranges)
            normalized_vector = vector_between_si_atoms / np.linalg.norm(vector_between_si_atoms)
            dot_product_grades = np.abs(np.dot(normalized_vector, DISPLACEMENT_VECTORS_NORM.T))
            selected_vector_index = np.argmin(dot_product_grades)
            midpoint = (si_atom.coord + vector_between_si_atoms / 2) % dimension_ranges

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
            output_file.write('pair_style lj/cut {:}\n'.format(O_O_DISTANCE_BOHR * intercept))
            output_file.write(f"pair_coeff * * 0.1 {O_O_DISTANCE_BOHR * intercept / 2**(1 / 6)} {O_O_DISTANCE_BOHR * intercept}\n")
            output_file.write("pair_modify shift yes\n")
            output_file.write("special_bonds lj 0.0 1.0 1.0\n")
            output_file.write('bond_style harmonic\n')
            output_file.write('bond_coeff 2 1.001 3.0405693345182674\n')
            output_file.write('bond_coeff 1 1.001 4.965228931415713\n')

    print("Finished writing LAMMPS files.")


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
            self.undercoordinated = remove_deleted_nodes(
                self.deleted_nodes, self.undercoordinated)

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
            paths = find_shared_cnxs(
                starting_node, self.undercoordinated, self.network_a)
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
            print('>>>>>>>>>>>> Initial Undercoordinated : ',
                  self.undercoordinated)
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
                print('############ One Connection Broken Rings : ',
                      self.broken_rings)
                print('>>>>>>>>>>>> One Connection Undercoordinated : ',
                      self.undercoordinated)
                self.network_a, self.network_b, self.new_ring, self.broken_rings, self.atoms = create_secondary_path(starting_node, atom_a, local_undercoordinated, local_broken_rings,
                                                                                                                     self.network_a, self.network_b_copy, self.original_image, self.atoms,
                                                                                                                     self.graph_a, self.graph_b, self.clone)
                create_secondary_path(starting_node, atom_a, local_undercoordinated,
                                      local_broken_rings, self.network_a, self.network_b_copy)
                self.refresh_new_cnxs(atoms)
                cv2.imshow("image", self.clone)
                cv2.imshow('image', draw_line_widget.show_image())
                cv2.waitKey(1)
                output_folder_name = get_folder_name(local_nodes, self.lj)
                write_netmc_data(output_path.joinpath(output_folder_name), input_path, local_nodes, self.network_b,
                                 self.deleted_nodes, self.rings_to_remove, self.new_ring)
                write_lammps_files(output_folder_name, self.lj, True, True)

    def show_image(self):
        return self.clone


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
    network_a = {i: {"crds": crds_a[i], "net": net_a[i],
                     "dual": dual_a[i]} for i in np.arange(num_nodes_a)}
    network_b = {i: {"crds": crds_b[i], "net": net_b[i],
                     "dual": dual_b[i]} for i in np.arange(num_nodes_b)}
    graph_a = get_graph(network_a)
    graph_b = get_graph(network_b)

    draw_line_widget = DrawLineWidget(output_path)

    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)
