from dataclasses import dataclass
from .netmc_data import NetMCAux, NetMCNets
import numpy as np


@dataclass
class NetMCNetwork:
    aux: NetMCAux
    crds: np.array
    nets: NetMCNets
    dual: NetMCNets
