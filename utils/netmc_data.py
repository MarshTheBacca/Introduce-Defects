from __future__ import annotations
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import networkx as nx


class NetworkType(Enum):
    A: str = "A"
    B: str = "B"


CONNECTED_MAP = {NetworkType.A: {NetworkType.A: "net", NetworkType.B: "dual"},
                 NetworkType.B: {NetworkType.A: "dual", NetworkType.B: "net"}}
NETMC_FILE_SUFFIXES = ("_A_aux.dat", "_A_crds.dat", "_A_net.dat", "_A_dual.dat",
                       "_B_aux.dat", "_B_crds.dat", "_B_net.dat", "_A_dual.dat")


def np_delete_element(array: np.array, target) -> np.array:
    index = np.where(array == target)
    return np.delete(array, index)


def get_dim(array: np.array, col_index: int, minimum: bool = False) -> float | np.float64:
    """Return the minimum or maximum value of a column of a 2D array.
    minimum = True means return minimum, else, return maximum
    """
    if minimum:
        return min(array[:, col_index])
    return max(array[:, col_index])


def shift_nodes(array: np.array, deleted_node: int | np.int64) -> np.array:
    for i, node in enumerate(array):
        if node > deleted_node:
            array[i] -= 1
    return array


def check_nets(net_1: list, net_2: list) -> bool:
    valid = True
    for selected_node, net in enumerate(net_1):
        for bonded_node in net:
            if selected_node not in net_2[bonded_node]:
                print(f"Selected node not in bonded node's net: {selected_node}\n"
                      "Bonded node's net: {self.net_A[bonded_node]}")
                valid = False
    return valid


@dataclass
class NetMCData:
    aux_A: NetMCAux
    aux_B: NetMCAux
    crds_A: np.array
    crds_B: np.array
    net_A: NetMCNets
    net_B: NetMCNets
    dual_A: NetMCNets
    dual_B: NetMCNets

    def __post_init__(self):
        self.num_nodes_A = len(self.crds_A)
        self.num_nodes_B = len(self.crds_B)

    def delete_node(self, deleted_node: int | np.int64, deleted_node_network: str):
        deleted_node_network = NetworkType(deleted_node_network)
        if deleted_node_network == NetworkType.A:
            self.crds_A = np.delete(self.crds_A, deleted_node, axis=0)
            self.num_nodes_A -= 1
        elif deleted_node_network == NetworkType.B:
            self.crds_B = np.delete(self.crds_B, deleted_node, axis=0)
            self.num_nodes_B -= 1
        self.aux_A.delete_node(deleted_node, deleted_node_network)
        self.aux_B.delete_node(deleted_node, deleted_node_network)
        self.net_A.delete_node(deleted_node, deleted_node_network)
        self.net_B.delete_node(deleted_node, deleted_node_network)
        self.dual_A.delete_node(deleted_node, deleted_node_network)
        self.dual_B.delete_node(deleted_node, deleted_node_network)
        self.refresh_auxs()

    def refresh_auxs(self):
        self.aux_A.refresh(self.num_nodes_A, self.net_A.get_max_cnxs(), self.dual_A.get_max_cnxs(),
                           get_dim(self.crds_A, 0, minimum=True), get_dim(self.crds_A, 0),
                           get_dim(self.crds_A, 1, minimum=True), get_dim(self.crds_A, 1))
        self.aux_B.refresh(self.num_nodes_B, self.net_B.get_max_cnxs(), self.dual_B.get_max_cnxs(),
                           get_dim(self.crds_B, 0, minimum=True), get_dim(self.crds_B, 0),
                           get_dim(self.crds_B, 1, minimum=True), get_dim(self.crds_B, 1))

    def check_nets(self):
        valid = True
        print("Checking nets...")
        if not check_nets(self.net_A, self.net_A):
            print("A-A connections invalid")
            valid = False
        if check_nets(self.net_B, self.net_B):
            print("B-B connections valid")
            valid = False
        if check_nets(self.dual_A, self.dual_B):
            print("A-B dual connections valid")
            valid = False
        if check_nets(self.dual_B, self.dual_A):
            print("A-B dual connections valid")
            valid = False
        if not valid:
            raise ValueError("Nets not valid")
        else:
            return valid

    @staticmethod
    def import_data(path: Path, prefix: str = "default") -> NetMCData:
        aux_A = NetMCAux.from_file(path.joinpath(f"{prefix}_A_aux.dat"))
        aux_B = NetMCAux.from_file(path.joinpath(f"{prefix}_B_aux.dat"))
        crds_A = np.genfromtxt(path.joinpath(f"{prefix}_A_crds.dat"))
        crds_B = np.genfromtxt(path.joinpath(f"{prefix}_B_crds.dat"))
        net_A = NetMCNets.from_file(path.joinpath(f"{prefix}_A_net.dat"))
        net_B = NetMCNets.from_file(path.joinpath(f"{prefix}_B_net.dat"))
        dual_A = NetMCNets.from_file(path.joinpath(f"{prefix}_A_dual.dat"))
        dual_B = NetMCNets.from_file(path.joinpath(f"{prefix}_B_dual.dat"))
        return NetMCData(aux_A, aux_B, crds_A, crds_B, net_A, net_B, dual_A, dual_B)

    def export_all(self, output_path: Path, prefix: str = "default"):
        self.aux_A.export(output_path, prefix)
        self.aux_B.export(output_path, prefix)
        np.savetxt(output_path.joinpath(f"{prefix}_A_crds.dat"), self.crds_A, fmt="%-19.6f")
        np.savetxt(output_path.joinpath(f"{prefix}_B_crds.dat"), self.crds_B, fmt="%-19.6f")
        self.net_A.export(output_path, prefix)
        self.net_B.export(output_path, prefix)
        self.dual_A.export(output_path, prefix)
        self.dual_B.export(output_path, prefix)


@dataclass
class NetMCNets:
    network_type: NetworkType
    connected_to: NetworkType
    nets: list[np.array] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        self.num_nodes = len(self.nets)

    def add_net(self, net: np.array) -> None:
        self.nets.append(net)
        self.num_nodes += 1

    def delete_node(self, deleted_node: int | np.int64, deleted_node_network: str):
        deleted_node_network = NetworkType(deleted_node_network)
        if deleted_node_network == self.network_type:
            del self.nets[deleted_node]
        if deleted_node_network == self.connected_to:
            for i, net in enumerate(self.nets):
                temp = self.nets[i]
                if deleted_node in net:
                    temp = np_delete_element(temp, deleted_node)
                self.nets[i] = shift_nodes(temp, deleted_node)

    @staticmethod
    def from_file(path: Path, skip_header: int = 0) -> NetMCNets:
        with open(path, "r") as net_file:
            nets = [np.array(line.strip().split(), dtype=np.int64) for line in net_file]
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
    def from_array(array: list[np.array], network_type: NetworkType, connected_to: NetworkType):
        return NetMCNets(network_type, connected_to, array)

    def export(self, output_path: Path = Path.cwd(), prefix: str = "default") -> None:
        with open(output_path.joinpath(f"{prefix}_{self.network_type.value}_{CONNECTED_MAP[self.network_type][self.connected_to]}.dat"), "w") as net_file:
            for net in self.nets:
                for connected_node in net:
                    net_file.write(f"{connected_node:<20}")
                net_file.write("\n")

    def shift_export(self, output_path: Path = Path.cwd(), deleted_nodes: np.array = np.array([]),
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
    num_nodes: int
    max_cnxs: int
    max_cnxs_dual: int
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

    def delete_node(self, deleted_node, deleted_node_network):
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
        xlo, xhi, ylo, yhi = dims[0][0], dims[0][1], dims[1][0], dims[1][1]
        network_type = NetworkType(path.name[-9])
        prefix = path.name[:11]
        return NetMCAux(num_nodes=num_nodes, max_cnxs=max_cnxs,
                        max_cnxs_dual=max_cnxs_dual,
                        geom_code=geom_code,
                        xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                        network_type=network_type, prefix=prefix)

    def refresh(self, num_nodes: int | np.int64, max_cnxs_A: int | np.int64,
                max_cnxs_B: int | np.int64, xlo: float | np.float64,
                xhi: float | np.float64, ylo: float | np.float64,
                yhi: float | np.float64):
        self.num_nodes = num_nodes
        self.max_cnxs_A = max_cnxs_A
        self.max_cnxs_B = max_cnxs_B
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
