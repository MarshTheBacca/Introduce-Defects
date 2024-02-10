# Code to produce coordinates of a lattice of hexagonal nodes and the connections between them
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import NetMCData
import copy

cwd = Path(__file__).parent
netmc_data = NetMCData.from_files(cwd.joinpath("to_lammps"), "gi_test")
# netmc_data = NetMCData.from_files(cwd.joinpath("Results_12_3194_444_LJ"), "testA")
# netmc_data = NetMCData.from_scratch(np.array([[0, 0], [10, 10]]), 1)
netmc_data.draw_graph(True, False, True, False, False, True, False)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
netmc_data.check()
