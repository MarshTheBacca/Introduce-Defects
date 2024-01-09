from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import networkx as nx
import numpy as np


class NetworkType(Enum):
    A: str = "A"
    B: str = "B"


CONNECTED_MAP = {NetworkType.A: {NetworkType.A: "net", NetworkType.B: "dual"},
                 NetworkType.B: {NetworkType.A: "dual", NetworkType.B: "net"}}
NETMC_FILE_SUFFIXES = ("_A_aux.dat", "_A_crds.dat", "_A_net.dat", "_A_dual.dat",
                       "_B_aux.dat", "_B_crds.dat", "_B_net.dat", "_A_dual.dat")


def np_delete_element(array: np.ndarray, target) -> np.ndarray:
    index = np.nonzero(array == target)
    return np.delete(array, index)


def get_dim(array: np.ndarray, col_index: int, minimum: bool = False) -> float | np.float64:
    """Return the minimum or maximum value of a column of a 2D array.
    minimum = True means return minimum, else, return maximum
    """
    if minimum:
        return min(array[:, col_index])
    return max(array[:, col_index])


def shift_nodes(array: np.ndarray, deleted_node: int | np.int64) -> np.ndarray:
    for i, node in enumerate(array):
        if node > deleted_node:
            array[i] -= 1
    return array


def check_nets(net_1: NetMCNets, net_2: NetMCNets) -> bool:
    valid = True
    for selected_node, net in enumerate(net_1.nets):
        for bonded_node in net:
            if selected_node not in net_2.nets[bonded_node]:
                print(f"Selected node not in bonded node's net: {selected_node}\n"
                      "Bonded node's net: {self.net_a.nets[bonded_node]}")
                valid = False
    return valid


def compare_aux_dim_crd_dim(aux_dim: float | np.float64, crd_dim: float | np.float64,
                            network_type: NetworkType, dim_string: str) -> bool:
    if "lo" in dim_string:
        if aux_dim > crd_dim:
            print(f"aux_{network_type.value} {dim_string} greater than crds_{network_type.value} {dim_string}: {aux_dim:<10}\t{crd_dim:<10}")
            return False
    elif "hi" in dim_string:
        if aux_dim < crd_dim:
            print(f"aux_{network_type.value} {dim_string} less than crds_{network_type.value} {dim_string}: {aux_dim:<10}\t{crd_dim:<10}")
            return False
    elif aux_dim != crd_dim:
        print(f"Inconsistent dimensions in aux_{network_type.value} and crds_{network_type.value} (safe to ignore): {aux_dim:<10}\t{crd_dim:<10}")
    return True


def get_graph(coords: np.ndarray, nets: NetMCNets) -> nx.Graph:
    graph = nx.Graph()
    for selected_node, coord in enumerate(coords):
        graph.add_node(selected_node, pos=coord)
        for bonded_node in nets.nets[selected_node]:
            graph.add_edge(selected_node, bonded_node)
    return graph


@dataclass
class NetMCData:
    aux_a: NetMCAux
    aux_b: NetMCAux
    crds_a: np.ndarray
    crds_b: np.ndarray
    net_a: NetMCNets
    net_b: NetMCNets
    dual_a: NetMCNets
    dual_b: NetMCNets

    def __post_init__(self):
        self.num_nodes_a = len(self.crds_a)
        self.num_nodes_b = len(self.crds_b)

    def delete_node(self, deleted_node: int | np.int64, deleted_node_network: str):
        if NetworkType(deleted_node_network) == NetworkType.A:
            self.crds_a = np.delete(self.crds_a, deleted_node, axis=0)
            self.num_nodes_a -= 1
        elif NetworkType(deleted_node_network) == NetworkType.B:
            self.crds_b = np.delete(self.crds_b, deleted_node, axis=0)
            self.num_nodes_b -= 1
        self.aux_a.delete_node(NetworkType(deleted_node_network))
        self.aux_b.delete_node(NetworkType(deleted_node_network))
        self.net_a.delete_node(deleted_node, NetworkType(deleted_node_network))
        self.net_b.delete_node(deleted_node, NetworkType(deleted_node_network))
        self.dual_a.delete_node(deleted_node, NetworkType(deleted_node_network))
        self.dual_b.delete_node(deleted_node, NetworkType(deleted_node_network))
        self.refresh_auxs()

    def refresh_auxs(self):
        self.aux_a.refresh(self.num_nodes_a, self.net_a.get_max_cnxs(), self.dual_a.get_max_cnxs(),
                           get_dim(self.crds_a, 0, minimum=True), get_dim(
                               self.crds_a, 0),
                           get_dim(self.crds_a, 1, minimum=True), get_dim(self.crds_a, 1))
        self.aux_b.refresh(self.num_nodes_b, self.net_b.get_max_cnxs(), self.dual_b.get_max_cnxs(),
                           get_dim(self.crds_b, 0, minimum=True), get_dim(
                               self.crds_b, 0),
                           get_dim(self.crds_b, 1, minimum=True), get_dim(self.crds_b, 1))

    def check(self):
        # I continue the function despite invalid data so that I can see all the issues
        valid = True
        if not self.check_nets():
            print("Inconsistent nets")
            valid = False
        else:
            print("Nets valid!")
        if not self.check_num_nodes():
            print("Inconsistent number of nodes")
            valid = False
        else:
            print("Number of nodes valid!")
        if not self.check_dims():
            print("Erroneous dimensions")
            valid = False
        else:
            print("Dimensions valid!")
        return valid

    def check_num_nodes(self):
        print("Checking number of nodes...")
        valid = True
        if self.aux_a.num_nodes != len(self.crds_a):
            print("Inconsistent number of nodes in aux_a and crds_a")
            valid = False
        if self.aux_b.num_nodes != len(self.crds_b):
            print("Inconsistent number of nodes in aux_b and crds_b")
            valid = False
        return valid

    def check_dims(self):
        print("Checking dimensions...")
        valid = True
        if not compare_aux_dim_crd_dim(self.aux_a.xlo, get_dim(self.crds_a, 0, minimum=True), NetworkType.A, "xlo"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_a.xhi, get_dim(self.crds_a, 0), NetworkType.A, "xhi"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_a.ylo, get_dim(self.crds_a, 1, minimum=True), NetworkType.A, "ylo"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_a.yhi, get_dim(self.crds_a, 1), NetworkType.A, "yhi"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_b.xlo, get_dim(self.crds_b, 0, minimum=True), NetworkType.B, "xlo"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_b.xhi, get_dim(self.crds_b, 0), NetworkType.B, "xhi"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_b.ylo, get_dim(self.crds_b, 1, minimum=True), NetworkType.B, "ylo"):
            valid = False
        if not compare_aux_dim_crd_dim(self.aux_b.yhi, get_dim(self.crds_b, 1), NetworkType.B, "yhi"):
            valid = False
        return valid

    def check_nets(self):
        valid = True
        print("Checking nets...")
        if not check_nets(self.net_a, self.net_a):
            print("A-A connections invalid")
            valid = False
        print("A-A connections valid")
        if not check_nets(self.net_b, self.net_b):
            print("B-B connections invalid")
            valid = False
        print("B-B connections valid")
        if not check_nets(self.dual_a, self.dual_b):
            print("A-B dual connections valid")
            valid = False
        print("A-B dual connections valid")
        if not check_nets(self.dual_b, self.dual_a):
            print("B-A dual connections valid")
            valid = False
        print("B-A dual connections valid")
        if not valid:
            raise ValueError("Nets not valid")
        else:
            return valid

    @staticmethod
    def import_data(path: Path, prefix: str = "default") -> NetMCData:
        aux_a = NetMCAux.from_file(path.joinpath(f"{prefix}_A_aux.dat"))
        aux_b = NetMCAux.from_file(path.joinpath(f"{prefix}_B_aux.dat"))
        crds_a = np.genfromtxt(path.joinpath(f"{prefix}_A_crds.dat"))
        crds_b = np.genfromtxt(path.joinpath(f"{prefix}_B_crds.dat"))
        net_a = NetMCNets.from_file(path.joinpath(f"{prefix}_A_net.dat"))
        net_b = NetMCNets.from_file(path.joinpath(f"{prefix}_B_net.dat"))
        dual_a = NetMCNets.from_file(path.joinpath(f"{prefix}_A_dual.dat"))
        dual_b = NetMCNets.from_file(path.joinpath(f"{prefix}_B_dual.dat"))
        return NetMCData(aux_a, aux_b, crds_a, crds_b, net_a, net_b, dual_a, dual_b)

    def export_all(self, output_path: Path, prefix: str = "default"):
        self.aux_a.export(output_path, prefix)
        self.aux_b.export(output_path, prefix)
        np.savetxt(output_path.joinpath(
            f"{prefix}_A_crds.dat"), self.crds_a, fmt="%-19.6f")
        np.savetxt(output_path.joinpath(
            f"{prefix}_B_crds.dat"), self.crds_b, fmt="%-19.6f")
        self.net_a.export(output_path, prefix)
        self.net_b.export(output_path, prefix)
        self.dual_a.export(output_path, prefix)
        self.dual_b.export(output_path, prefix)

    def get_graph(self, network_type: NetworkType) -> nx.graph:
        if network_type == NetworkType.A:
            return get_graph(self.crds_a, self.net_a)
        elif network_type == NetworkType.B:
            return get_graph(self.crds_b, self.net_b)
        raise ValueError("Invalid network type")

    def get_dimensions(self, network_type: NetworkType) -> np.ndarray:
        if network_type == NetworkType.A:
            return np.array([[self.aux_a.xlo, self.aux_a.xhi], [self.aux_a.ylo, self.aux_a.yhi]])
        elif network_type == NetworkType.B:
            return np.array([[self.aux_b.xlo, self.aux_b.xhi], [self.aux_b.ylo, self.aux_b.yhi]])
        raise ValueError("Invalid network type")


@dataclass
class NetMCNets:
    network_type: NetworkType
    connected_to: NetworkType
    nets: list[np.ndarray] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        self.num_nodes = len(self.nets)

    def add_net(self, net: np.ndarray) -> None:
        self.nets.append(net)
        self.num_nodes += 1

    def delete_node(self, deleted_node: int | np.int64, deleted_node_network: NetworkType):
        if NetworkType(deleted_node_network) == self.network_type:
            del self.nets[deleted_node]
        if NetworkType(deleted_node_network) == self.connected_to:
            for i, net in enumerate(self.nets):
                temp = self.nets[i]
                if deleted_node in net:
                    temp = np_delete_element(temp, deleted_node)
                self.nets[i] = shift_nodes(temp, deleted_node)

    @staticmethod
    def from_file(path: Path, skip_header: int = 0) -> NetMCNets:
        with open(path, "r") as net_file:
            nets = [np.array(line.strip().split(), dtype=np.int64)
                    for line in net_file]
        if path.name[-7:-4] == "net":
            network_type = NetworkType(path.name[-9])
            if network_type == NetworkType.A:
                connected_to = NetworkType.A
            else:
                connected_to = NetworkType.B
        elif path.name[-8:-4] == "dual":
            network_type = NetworkType(path.name[-10])
            if network_type == NetworkType.A:
                connected_to = NetworkType.B
            else:
                connected_to = NetworkType.A
        else:
            raise ValueError("file name does not contain 'net' or 'dual', so cannot identify"
                             "what the nodes are connected to")
        return NetMCNets(network_type=network_type, connected_to=connected_to, nets=nets)

    @staticmethod
    def from_array(array: list[np.ndarray], network_type: NetworkType, connected_to: NetworkType):
        return NetMCNets(network_type, connected_to, array)

    def export(self, output_path: Path = Path.cwd(), prefix: str = "default") -> None:
        with open(output_path.joinpath(f"{prefix}_{self.network_type.value}_{CONNECTED_MAP[self.network_type][self.connected_to]}.dat"), "w") as net_file:
            for net in self.nets:
                for connected_node in net:
                    net_file.write(f"{connected_node:<20}")
                net_file.write("\n")

    def shift_export(self, output_path: Path = Path.cwd(), deleted_nodes: np.ndarray = np.array([]),
                     prefix: str = "default",) -> None:
        with open(output_path.joinpath(f"{prefix}_{self.network_type.value}_{CONNECTED_MAP[self.network_type][self.connected_to]}.dat"), 'w') as net_file:
            for net in self.nets:
                for connected_node in net:
                    for deleted_node in deleted_nodes:
                        if connected_node >= deleted_node:
                            connected_node -= 1
                    if connected_node < 0:
                        print("Error in atom repositioning")
                    elif connected_node > self.num_nodes:
                        print("Including illegal connections")
                    net_file.write(f"{connected_node:<10}")
                net_file.write("\n")

    def get_max_cnxs(self):
        return max([net.shape[0] for net in self.nets])

    def get_graph(self) -> nx.graph:
        graph = nx.graph()
        for selected_node, net in enumerate(self.nets):
            graph.add_node(selected_node)
            for bonded_node in net:
                graph.add_edge(selected_node, bonded_node)
        return graph


@dataclass
class NetMCAux:
    num_nodes: int | np.int64
    max_cnxs: int | np.int64
    max_cnxs_dual: int | np.int64
    geom_code: str
    xlo: float | np.float64
    xhi: float | np.float64
    ylo: float | np.float64
    yhi: float | np.float64
    network_type: NetworkType
    prefix: str

    def export(self, path: Path = Path.cwd(), prefix: str = "default") -> None:
        with open(path.joinpath(f"{prefix}_{self.network_type.value}_aux.dat"), "w") as aux_file:
            aux_file.write(f"{self.num_nodes}\n")
            aux_file.write(f"{self.max_cnxs:<10}{self.max_cnxs_dual:<10}\n")
            aux_file.write(f"{self.geom_code}\n")
            aux_file.write(f"{self.xlo:<20.6f}{self.xhi:<20.6f}\n")
            aux_file.write(f"{self.ylo:<20.6f}{self.yhi:<20.6f}\n")

    def delete_node(self, deleted_node_network):
        if deleted_node_network == self.network_type:
            self.num_nodes -= 1

    @staticmethod
    def from_file(path: Path):
        with open(path, "r") as aux_file:
            num_nodes = int(aux_file.readline().strip())
            cnxs = aux_file.readline().strip().split()
            max_cnxs, max_cnxs_dual = int(cnxs[0]), int(cnxs[1])
            geom_code = aux_file.readline().strip()
        dims = np.genfromtxt(path, skip_header=3, dtype=np.float64)
        # This is the correct format for the aux dimensions!!
        #           xhi         yhi
        #           xlo         ylo
        xhi, yhi, xlo, ylo = dims[0][0], dims[0][1], dims[1][0], dims[1][1]
        network_type = NetworkType(path.name[-9])
        prefix = path.name[:11]
        return NetMCAux(num_nodes=num_nodes, max_cnxs=max_cnxs,
                        max_cnxs_dual=max_cnxs_dual,
                        geom_code=geom_code,
                        xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                        network_type=network_type, prefix=prefix)

    def refresh(self, num_nodes: int | np.int64, max_cnxs_a: int | np.int64,
                max_cnxs_b: int | np.int64, xlo: float | np.float64,
                xhi: float | np.float64, ylo: float | np.float64,
                yhi: float | np.float64) -> None:
        self.num_nodes = num_nodes
        self.max_cnxs_a = max_cnxs_a
        self.max_cnxs_b = max_cnxs_b
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
