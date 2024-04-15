from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent, ResizeEvent

from .bss_data import (BSSData, CouldNotBondUndercoordinatedNodesException,
                       InvalidUndercoordinatedNodesException)

MAX_UNDOS = 10


class UserClosedError(Exception):
    pass


class DefectIntroducer:
    def __init__(self, bss_data: BSSData, output_path: Path):
        self.bss_data = bss_data
        self.bss_data_backups = []
        self.output_path = output_path
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 10)
        self.fig.canvas.manager.window.title("Bond Switch Simulator Defect Editor")
        self.closed_properly = False
        self.dragging = False
        self.drag_start_time = None
        self.initial_anchor_position = None
        self.initial_click_position = None
        self.dragging_node = None

    def on_key_press(self, event: KeyEvent) -> None:
        if event.key in ["q", "escape"]:
            print("Bonding undercoordinated nodes...")
            try:
                self.bss_data = BSSData.bond_undercoordinated_nodes(self.bss_data)
                print("Undercoordinated nodes bonded successfully.")
                self.closed_properly = True
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
            except CouldNotBondUndercoordinatedNodesException as e:
                print(e)
        elif event.key == "b":
            if self.initial_anchor_position is not None:
                self.manual_bond(np.array([event.xdata, event.ydata]))
                self.initial_anchor_position = None
                return
            self.initial_anchor_position = np.array([event.xdata, event.ydata])

    def on_press(self, event: MouseEvent):
        if event.button == 1:
            self.create_backup()
            self.initial_click_position = np.array([event.xdata, event.ydata])
            self.drag_start_time = time.time()
        elif event.button == 3:
            self.drag_start_time = None

    def on_release(self, event: MouseEvent):
        self.drag_start_time = None
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode != '':
            return
        if not self.dragging:
            self.onclick(event)
            return
        self.dragging = False
        self.dragging_node.coord = np.array([event.xdata, event.ydata])
        self.refresh_plot()
        self.dragging_node = None

    def create_backup(self) -> None:
        if len(self.bss_data_backups) >= MAX_UNDOS:
            self.bss_data_backups.pop(0)
        self.bss_data_backups.append(copy.deepcopy(self.bss_data))

    def manual_bond(self, final_click_location: np.array) -> None:
        self.create_backup()
        node_1, _ = self.bss_data.base_network.get_nearest_node(self.initial_anchor_position)
        node_2, _ = self.bss_data.base_network.get_nearest_node(final_click_location)
        if node_1 == node_2:
            return
        self.bss_data.manual_bond_addition_deletion(node_1, node_2)
        self.refresh_plot()

    def onclick(self, event: MouseEvent):
        if event.button == 1:  # Left click
            node, _ = self.bss_data.base_network.get_nearest_node([event.xdata, event.ydata])
            self.bss_data.delete_node_and_merge_rings(node)
            self.refresh_plot()

        elif event.button == 3:  # Right click
            if self.bss_data_backups:
                self.bss_data = self.bss_data_backups.pop()
                self.refresh_plot()
                return
            print("Cannot undo any further")

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

    def on_resize(self, event: ResizeEvent):
        window_height = self.ax.get_window_extent().height
        self.instructions.set_fontsize(window_height / 80)

    @ property
    def drag_time(self) -> float:
        if self.drag_start_time is not None:
            return time.time() - self.drag_start_time
        return 0

    def refresh_plot(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        plt.gca().set_aspect('equal', adjustable='box')
        self.bss_data.draw_graph(True, True, True, False, True, False, False)
        # Only set xlim and ylim if they have been changed from their default values
        if xlim != (0.0, 1.0):
            self.ax.set_xlim(xlim)
        if ylim != (0.0, 1.0):
            self.ax.set_ylim(ylim)
        instructions = ("Left click - Delete node\n"
                        "Right click - Undo (max of 10)\n"
                        "Left click and drag - Move node (ring or base)\n"
                        "Q/Escape - Finish")
        self.instructions = self.ax.text(0.5, 0, instructions, transform=self.ax.transAxes, horizontalalignment='center')
        plt.pause(0.001)

    def plot(self):
        self.refresh_plot()
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_resize = self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.bss_data.check()
        plt.title("Network Editor")
        plt.show()
