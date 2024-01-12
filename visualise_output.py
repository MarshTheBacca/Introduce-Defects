# Code to produce coordinates of a lattice of hexagonal nodes and the connections between them
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import NetMCData
import copy

cwd = Path.cwd()
# netmc_data = NetMCData.from_files(cwd.joinpath("output_files"), "test")
netmc_data = NetMCData.from_files(cwd.joinpath("output_files"), "test")
netmc_data.draw_graph(True, True, True, True, True, False, False)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
netmc_data.check()
