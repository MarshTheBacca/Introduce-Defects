from __future__ import annotations


from pathlib import Path

import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib import pyplot as plt

from .bss_data import BSSData, InvalidUndercoordinatedNodesException


class DefectIntroducer:
    def __init__(self, bss_data: BSSData, output_path: Path):
        self.bss_data = bss_data
        self.output_path = output_path
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 10)
        self.fig.canvas.manager.window.title("Bond Switch Simulator Defect Editor")
        self.nodes = np.array([node.coord for node in self.bss_data.base_network.nodes])
        self.click = False

    def on_key_press(self, event: KeyEvent) -> None:
        if event.key in ["q", "escape"]:
            plt.close(self.fig)

    def on_press(self, event: MouseEvent):
        self.click = True

    def on_release(self, event: MouseEvent):
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode != '':
            return
        if self.click:
            self.onclick(event)
        self.click = False

    def onclick(self, event: MouseEvent):
        if event.dblclick:
            return
        try:
            if event.button == 1:  # Left click
                node, _ = self.bss_data.base_network.get_nearest_node(np.array([event.xdata, event.ydata]))
                self.bss_data.delete_node_and_merge_rings(node)
                print(f"Max ring size: {self.bss_data.ring_network.max_ring_connections}")
                self.refresh_plot()

            elif event.button == 3:  # Right click
                print("Bonding undercoordinated nodes...")
                try:
                    self.bss_data = BSSData.bond_undercoordinated_nodes(self.bss_data)
                    plt.close(self.fig)
                except InvalidUndercoordinatedNodesException as e:
                    if str(e) == "Number of undercoordinated nodes is odd, so cannot bond them.":
                        print("Number of undercoordinated nodes is odd, so cannot bond them.\n Please select another node to delete.")
                    elif str(e) == "There are three consecutive undercoordinated nodes in the ring walk.":
                        print("There are three consecutive undercoordinated nodes in the ring walk.\n Please select another node to delete.")
                    elif str(e) == "There are an odd number of undercoordinated nodes between two pairs of adjacent undercoordinated nodes.":
                        print("There are an odd number of undercoordinated nodes between two adjacent undercoordinated nodes.\n"
                              "This means we would have to bond an undercoordinated node to one of its own neighbours, which is not allowed.\n"
                              "Please select another node to delete.")
                    else:
                        raise
        except ValueError as e:
            if str(e) != "'x' must be finite, check for nan or inf values":
                raise

    def refresh_plot(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()

        plt.gca().set_aspect('equal', adjustable='box')
        self.bss_data.draw_graph()
        # Only set xlim and ylim if they have been changed from their default values
        if xlim != (0.0, 1.0):
            self.ax.set_xlim(xlim)
        if ylim != (0.0, 1.0):
            self.ax.set_ylim(ylim)
        plt.pause(0.001)

    def plot(self):
        self.refresh_plot()
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.bss_data.check()
        plt.title("Network Editor")
        plt.show()
