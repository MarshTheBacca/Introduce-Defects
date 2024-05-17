from __future__ import annotations

import copy
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent, ResizeEvent
from matplotlib.figure import Figure

from .bss_data import (BSSData, CouldNotBondUndercoordinatedNodesException,
                       InvalidUndercoordinatedNodesException)
from .bss_node import BSSNode
from .validation_utils import get_valid_str

MAX_UNDOS = 20
FIGURES_PATH = Path(__file__).parent.joinpath("figures")


class UserClosedError(Exception):
    pass


class DefectIntroducerFigures:
    def __init__(self, bss_data: BSSData, output_path: Path):
        self.bss_data: BSSData = bss_data
        self.coloured_bonds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"blue": [], "red": [], "limegreen": []}
        self.marked_nodes: list[BSSNode] = []
        self.numbered_rings: list[int] = []
        self.backups: list[DefectIntroducerState] = []
        self.output_path: Path = output_path
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.window.title("Bond Switch Simulator Defect Editor")
        self.closed_properly = False
        self.dragging: bool = False
        self.drag_start_time: Optional[float] = None
        self.initial_anchor_position: Optional[tuple[float, float]] = None
        self.initial_click_position: Optional[tuple[float, float]] = None
        self.dragging_node: Optional[bool] = None
        self.skip_release: bool = False

        # Disable default Matplotlib key bindings
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_key_press(self, event: KeyEvent) -> None:
        bond_key_map = {"u": "blue", "r": "red", "g": "limegreen"}

        def handle_coloured_bonds(bonds_list: list[tuple[np.ndarray, np.ndarray]]) -> None:
            if self.initial_anchor_position is not None:
                self.create_backup()
                bonds_list.append((self.bss_data.base_network.get_nearest_node(self.initial_anchor_position)[0].coord,
                                   self.bss_data.base_network.get_nearest_node(np.array([event.xdata, event.ydata]))[0].coord))
                self.initial_anchor_position = None
                self.refresh_plot()
                return
            self.initial_anchor_position = np.array([event.xdata, event.ydata])

        if event.key in ["q", "escape"]:
            print("Bonding undercoordinated nodes...")
            try:
                self.bss_data = BSSData.bond_undercoordinated_nodes(self.bss_data)
                print("Undercoordinated nodes bonded successfully.")
                self.closed_properly = True
                plt.close(self.fig)
            except InvalidUndercoordinatedNodesException as e:
                if str(e) == "Number of undercoordinated nodes is odd, so cannot bond them.":
                    print(f"{e}\nPlease select another node to delete.")
                elif str(e) == "There are three consecutive undercoordinated nodes in the ring walk.":
                    print(f"{e}\nPlease select another node to delete.")
                elif str(e) == "There are an odd number of undercoordinated nodes between 'islands'.":
                    print(f"{e}\nThis means we would have to bond an undercoordinated node to one of its own neighbours, which is not allowed.\n"
                          "Please select another node to delete.")
            except CouldNotBondUndercoordinatedNodesException as e:
                print(e)
        elif event.key == "b":
            if self.initial_anchor_position is not None:
                self.create_backup()
                self.manual_bond(np.array([event.xdata, event.ydata]))
                self.initial_anchor_position = None
                return
            self.initial_anchor_position = np.array([event.xdata, event.ydata])
        elif event.key in bond_key_map:
            handle_coloured_bonds(self.coloured_bonds[bond_key_map[event.key]])
        elif event.key == "m":
            if event.xdata == None or event.ydata == None:
                return
            self.create_backup()
            node, _ = self.bss_data.base_network.get_nearest_node([event.xdata, event.ydata])
            if node not in self.marked_nodes:
                self.marked_nodes.append(node)
            else:
                self.marked_nodes.remove(node)
            self.refresh_plot()
        elif event.key == "n":
            if event.xdata == None or event.ydata == None:
                return
            self.create_backup()
            node, _ = self.bss_data.ring_network.get_nearest_node([event.xdata, event.ydata])
            if node.id in self.numbered_rings:
                self.numbered_rings.remove(node.id)
            else:
                self.numbered_rings.append(node.id)
            self.refresh_plot()
        elif event.key == "s":
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(initialdir=FIGURES_PATH, defaultextension=".png")

            if file_path:
                self.fig.savefig(file_path, dpi=500, bbox_inches='tight', pad_inches=0.1)

    def on_press(self, event: MouseEvent):
        if event.xdata == None or event.ydata == None:
            self.skip_release = True
            return
        self.skip_release = False
        if event.button == 1:
            self.initial_click_position = np.array([event.xdata, event.ydata])
            self.drag_start_time = time.time()
        elif event.button == 3:
            self.drag_start_time = None

    def on_click(self, event: MouseEvent):
        if event.button == 1:  # Left click
            self.create_backup()
            node, _ = self.bss_data.base_network.get_nearest_node([event.xdata, event.ydata])
            if node in self.marked_nodes:
                self.marked_nodes.remove(node)
            self.bss_data.delete_node_and_merge_rings(node)
            self.refresh_plot()
        elif event.button == 3:  # Right click
            self.restore_backup()

    def on_release(self, event: MouseEvent):
        if event.xdata == None or event.ydata == None or self.skip_release:
            return
        self.drag_start_time = None
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode != '':
            return
        if not self.dragging:
            self.on_click(event)
            return
        self.dragging = False
        self.dragging_node.coord = np.array([event.xdata, event.ydata])
        self.refresh_plot()
        self.dragging_node = None

    def create_backup(self) -> None:
        if len(self.backups) >= MAX_UNDOS:
            self.backups.pop(0)
        self.backups.append(DefectIntroducerState(copy.deepcopy(self.bss_data),
                                                  copy.deepcopy(self.coloured_bonds),
                                                  copy.deepcopy(self.marked_nodes),
                                                  copy.deepcopy(self.numbered_rings)))

    def restore_backup(self) -> None:
        if self.backups:
            recovered_state = self.backups.pop()
            self.bss_data = recovered_state.bss_data
            self.coloured_bonds = recovered_state.coloured_bonds
            self.marked_nodes = recovered_state.marked_nodes
            self.numbered_rings = recovered_state.numbered_rings
            self.refresh_plot()
            return
        print("Cannot undo any further")

    def manual_bond(self, final_click_location: np.array) -> None:
        node_1, _ = self.bss_data.base_network.get_nearest_node(self.initial_anchor_position)
        node_2, _ = self.bss_data.base_network.get_nearest_node(final_click_location)
        if node_1 == node_2:
            return
        self.bss_data.manual_bond_addition_deletion(node_1, node_2)
        self.refresh_plot()

    def on_motion(self, event: MouseEvent):
        if self.dragging_node:
            return
        if self.drag_time > 0.3:
            self.dragging = True
            close_base_node, base_distance = self.bss_data.base_network.get_nearest_node(self.initial_click_position)
            close_ring_node, ring_distance = self.bss_data.ring_network.get_nearest_node(self.initial_click_position)
            if base_distance < ring_distance:
                self.dragging_node = close_base_node
                return
            self.dragging_node = close_ring_node

    @ property
    def drag_time(self) -> float:
        if self.drag_start_time is not None:
            return time.time() - self.drag_start_time
        return 0

    def redraw_node_circles(self):
        for bonds in self.coloured_bonds.values():
            for coord_1, coord_2 in bonds:
                self.ax.plot(coord_1[0], coord_1[1], 'ro', markersize=5)
                self.ax.plot(coord_2[0], coord_2[1], 'ro', markersize=5)

    def refresh_plot(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Latin Modern Roman']
        # self.bss_data.draw_graph(True, False, True, False, False, False, False)
        self.bss_data.draw_graph_pretty_figures(draw_dimensions=False)

        for color, bonds in self.coloured_bonds.items():
            for coord_1, coord_2 in bonds:
                self.ax.plot([coord_1[0], coord_2[0]], [coord_1[1], coord_2[1]], color=color, linewidth=2)

        # self.redraw_node_circles()

        for marked_node in self.marked_nodes:
            circle = patches.Circle(marked_node.coord, radius=0.2, edgecolor="red", facecolor="none", linewidth=2)
            self.ax.add_patch(circle)

        for ring_id in self.numbered_rings:
            # draw the ring lengths in the centre of each ring
            ring_node = self.bss_data.ring_network.nodes[ring_id]
            self.ax.text(ring_node.coord[0], ring_node.coord[1], str(len(ring_node.ring_neighbours)),
                         fontsize=30, color="black", ha='center', va='center')
        # Only set xlim and ylim if they have been changed from their default values
        if xlim != (0.0, 1.0):
            self.ax.set_xlim(xlim)
        if ylim != (0.0, 1.0):
            self.ax.set_ylim(ylim)
        plt.pause(0.001)

    def plot(self):
        self.refresh_plot()
        self.bss_data.check()
        plt.show()


@ dataclass
class DefectIntroducerState:
    bss_data: BSSData
    coloured_bonds: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    marked_nodes: list[BSSNode]
    numbered_rings: list[int]
