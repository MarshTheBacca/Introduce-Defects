import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
import os
from sys import exit
list_x = []
list_y = []

node_list = []
node_cnxs = []

input_folder = 't-3200_s0_same'

with open(input_folder + '/test_A_crds.dat', 'r') as f:
    A_crds = np.genfromtxt(f)
with open(input_folder + '/test_A_net.dat', 'r') as f:
    A_net = np.genfromtxt(f)
with open(input_folder + '/test_A_dual.dat', 'r') as f:
    A_dual = np.genfromtxt(f)

with open(input_folder + '/test_B_crds.dat', 'r') as f:
    B_crds = np.genfromtxt(f)
with open(input_folder + '/test_B_net.dat', 'r') as f:
    B_net = np.genfromtxt(f)
with open(input_folder + '/test_B_dual.dat', 'r') as f:
    B_dual = np.genfromtxt(f)


def import_Nodes():
    nNodes = A_crds.shape[0]
    Nodes = nx.Graph()
    nodes = {}

    nDual = B_crds.shape[0]
    Dual = nx.Graph()
    dual = {}

    for i in range(nNodes):
        nodes['{:}'.format(i)] = {}
        nodes['{:}'.format(i)]['crds'] = A_crds[i, :]
        nodes['{:}'.format(i)]['net'] = A_net[i, :]
        nodes['{:}'.format(i)]['dual'] = A_dual[i, :]
    for i in range(nDual):
        dual['{:}'.format(i)] = {}
        dual['{:}'.format(i)]['crds'] = B_crds[i, :]
        dual['{:}'.format(i)]['net'] = B_net[i, :]
        dual['{:}'.format(i)]['dual'] = B_dual[i, :]

    for i in range(nNodes):
        Nodes.add_node('{:}'.format(i), pos=A_crds[i, :])
    for i in range(nNodes):
        for j in range(A_net.shape[1]):
            Nodes.add_edge('{:}'.format(i), '{:}'.format(A_net[i, j]))

    for i in range(nDual):
        Dual.add_node('{:}'.format(i), pos=B_crds[i, :])
    for i in range(nDual):
        for j in range(B_net.shape[1]):
            Dual.add_edge('{:}'.format(i), '{:}'.format(B_net[i, j]))

    return nodes, dual, Nodes, Dual


nodes, dual, Nodes, Dual = import_Nodes()


class DrawLineWidget:
    def __init__(self,):
        self.original_image = cv2.imread('bg.png')
        self.clone = copy.deepcopy(self.original_image)
        self.new_ring = 0
        self.folder = "Results"
        self.lj = 1
        self.rings_to_remove = []
        self.nodes, self.dual, self.Nodes, self.Dual = import_Nodes()
        self.deleted_nodes = []
        self.undercoordinated = []
        self.broken_rings = []
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

        # local variables to help recenter
        # dimensions of box = 1700x1000

        self.x_offset = 100
        self.y_offset = 100
        self.scale = int(1000 / np.sqrt(len(self.dual.keys())))
        self.refresh_new_cnxs([])

    def check(self):
        print("---checking...")
        for node in self.nodes.keys():
            for net in [int(i) for i in self.nodes[node]['net']]:
                print('UNAVALIABLE NODES !! ')
                if int(node) in [int(i) for i in self.nodes.keys()]:
                    print(node, [int(i) for i in self.nodes[node]['net']])
                if int(net) in [int(i) for i in self.nodes.keys()]:
                    print(net, [int(i) for i in self.nodes[net]['net']])

                if int(node) not in [int(i) for i in self.nodes.keys()]:
                    print(node)
                if int(net) not in [int(i) for i in self.nodes.keys()]:
                    print(net)
                if int(node) not in [int(i) for i in self.nodes['{:}'.format(int(net))]['net']]:
                    print('########################################')
                    print(node)
                    print(self.nodes['{:}'.format(int(node))]['net'])
                    print(net)
                    print(self.nodes['{:}'.format(int(net))]['net'])
                    print('Broken Node-Node')
                    time.sleep(100)
        for node in nodes.keys():
            for ring in nodes[node]['dual']:
                if int(node) not in dual['{:}'.format(int(ring))]['dual']:
                    print('Broken Node-Dual')
                    time.sleep(100)
        print("---...checked")

    def update_Nodes(self,):

        # Clear All Nodes and Duals
        self.Nodes.clear()
        self.Dual.clear()

        # Add back all nodes with crds
        for i in self.nodes.keys():
            self.Nodes.add_node('{:}'.format(i), pos=self.nodes['{:}'.format(i)]['crds'])
        # Add all edges
        for i in self.nodes.keys():
            for j in self.nodes['{:}'.format(i)]['net']:
                self.Nodes.add_edge('{:}'.format(i), '{:}'.format(j))

        for i in self.dual.keys():
            self.Dual.add_node('{:}'.format(i), pos=self.dual['{:}'.format(i)]['crds'])
        for i in self.dual.keys():
            for j in self.dual['{:}'.format(i)]['net']:
                self.Dual.add_edge('{:}'.format(i), '{:}'.format(j))

        return self.Nodes, self.Dual

    def refresh_new_cnxs(self, atoms):
        self.update_Nodes()
        node_pos = nx.get_node_attributes(self.Nodes, 'pos')
        for node in node_pos:
            center_coordinates = (
                int(self.scale * node_pos[node][0] + self.x_offset), int(self.scale * node_pos[node][1] + self.y_offset))
            radius = int(0.1 * self.scale)
            color = (255, 0, 0)
            thickness = 1
            cv2.circle(self.clone, center_coordinates, radius, color, thickness)

        ring_pos = nx.get_node_attributes(self.Dual, 'pos')
        for ring in ring_pos:
            center_coordinates = (
                int(self.scale * ring_pos[ring][0] + self.x_offset), int(self.scale * ring_pos[ring][1] + self.y_offset))
            radius = int(0.1 * self.scale)
            color = (0, 255, 0)
            thickness = 1
            cv2.circle(self.clone, center_coordinates, radius, color, thickness)

        for node in self.Nodes:
            edges = self.Nodes.edges(node)
            for edge0, edge1 in edges:
                edge0, edge1 = int(float(edge0)), int(float(edge1))
                if edge0 < edge1 and '{:}'.format(edge0) in self.Nodes.nodes and '{:}'.format(edge1) in self.Nodes.nodes:
                    crd0, crd1 = (int(self.scale * node_pos['{:}'.format(edge0)][0] + self.x_offset),
                                  int(self.scale * node_pos['{:}'.format(edge0)][1] + self.y_offset)), \
                        (int(self.scale * node_pos['{:}'.format(edge1)][0] + self.x_offset),
                         int(self.scale * node_pos['{:}'.format(edge1)][1] + self.y_offset))
                    (x0, y0) = crd0
                    (x1, y1) = crd1
                    dx = x1 - x0
                    dy = y1 - y0
                    if abs(dx) < 100 and abs(dy) < 100:
                        cv2.line(self.clone, crd0, crd1, (255, 36, 12), 2)

        for ring in self.Dual:
            edges = self.Dual.edges(ring)
            for edge0, edge1 in edges:
                edge0, edge1 = int(float(edge0)), int(float(edge1))
                if edge0 < edge1:
                    crd0, crd1 = (int(self.scale * ring_pos['{:}'.format(edge0)][0] + self.x_offset),
                                  int(self.scale * ring_pos['{:}'.format(edge0)][1] + self.y_offset)), \
                        (int(self.scale * ring_pos['{:}'.format(edge1)][0] + self.x_offset),
                         int(self.scale * ring_pos['{:}'.format(edge1)][1] + self.y_offset))
                    (x0, y0) = crd0
                    (x1, y1) = crd1
                    dx = x1 - x0
                    dy = y1 - y0
                    if abs(dx) < 100 and abs(dy) < 100:
                        cv2.line(self.clone, crd0, crd1, (255, 36, 12), 1)

        for i in self.undercoordinated:
            if '{:}'.format(i) in self.nodes.keys():
                x_uncoord, y_uncoord = self.scale * self.nodes['{:}'.format(i)]['crds'][
                    0] + self.x_offset, self.scale * self.nodes['{:}'.format(i)]['crds'][1] + self.y_offset
                cv2.circle(self.clone, (int(x_uncoord), int(y_uncoord)), 15, (36, 255, 12), -1)
            else:
                print("Uncoordinated Atoms Doesnt Exist")
        for i in atoms:
            print(i, atoms.index(i))
            if '{:}'.format(i) in self.nodes.keys():
                x_uncoord, y_uncoord = self.scale * self.nodes['{:}'.format(i)]['crds'][
                    0] + self.x_offset, self.scale * self.nodes['{:}'.format(i)]['crds'][1] + self.y_offset
                r, g, b = (5 + atoms.index(i) * 20) % 255, 36, abs(255 - atoms.index(i) * 20)
                cv2.circle(self.clone, (int(x_uncoord), int(y_uncoord)), 10, (r, g, b), -1)
            else:
                print("Starting From atom doesn't Exist!")

        for i in range(0, len(atoms) - 1, 2):
            atom0 = atoms[i]
            atom1 = atoms[i + 1]
            x_uncoord_0, y_uncoord_0 = self.scale * self.nodes['{:}'.format(atom0)]['crds'][
                0] + self.x_offset, self.scale * self.nodes['{:}'.format(atom0)]['crds'][1] + self.y_offset
            x_uncoord_1, y_uncoord_1 = self.scale * self.nodes['{:}'.format(atom1)]['crds'][
                0] + self.x_offset, self.scale * self.nodes['{:}'.format(atom1)]['crds'][1] + self.y_offset
            r, g, b = 1, 1, 1
            cv2.line(self.clone, (int(x_uncoord_0), int(y_uncoord_0)), (int(x_uncoord_1), int(y_uncoord_1)), (r, g, b), 5)

        return

    def write_local(self, local_nodes, local_dual, which_ring):

        key = {}
        for node in local_nodes.keys():
            node_val = int(node)
            for k in self.deleted_nodes:
                if node_val >= k:
                    node_val -= 1
            key['{:}'.format(node)] = node_val

        folder = "Results_"
        maxRingSize = max([local_dual[i]['net'].shape[0] for i in local_dual.keys()])
        dimension = len(local_nodes.keys())
        folder += "{:}_".format(maxRingSize)
        folder += "{:}_".format(dimension)
        for i in local_dual.keys():
            if local_dual[i]['net'].shape[0] < 6:
                folder += "{:}".format(int(local_dual[i]['net'].shape[0]))
        if self.lj:
            folder += "_LJ"
        self.folder = folder

        print("FOLDER NAME !!!")
        print(folder + "\n\n\n\n")

        if not os.path.isdir(folder):
            os.mkdir(folder)

        with open(folder + '/key.dat', 'w') as f:
            for i in self.deleted_nodes:
                f.write('{:<5}'.format(i))
            f.write('\n')
            for node in local_nodes.keys():
                node_val = int(node)
                for k in self.deleted_nodes:
                    if node_val >= k:
                        node_val -= 1
                f.write('{:<10}{:<10}\n'.format(int(node), int(node_val)))

        def ordered_cnxs_dual_to_node(dual_dict, node_dict):
            for ring in dual_dict.keys():
                cnxs_list = dual_dict[ring]['dual']
                if not isinstance(cnxs_list, list):
                    cnxs_list = cnxs_list.tolist()
                new_cnxs_list = []
                new_crd_list = []
                new_cnxs_list.append(cnxs_list[0])
                new_crd_list.append(node_dict['{:}'.format(int(cnxs_list[0]))]['crds'])
                i = 0
                while i < len(cnxs_list) - 1:
                    node0 = int(new_cnxs_list[i])
                    connected_to_0 = node_dict['{:}'.format(node0)]['net'].tolist()
                    options = set(connected_to_0).intersection(cnxs_list)
                    for val in new_cnxs_list:
                        if val in options:
                            options.remove(val)
                    options = list(options)
                    new_cnxs_list.append(options[0])
                    new_crd_list.append(node_dict['{:}'.format(int(options[0]))]['crds'])
                    i += 1

                area = 0
                for i in range(len(cnxs_list)):
                    x0, y0, x1, y1 = new_crd_list[i - 1][0], new_crd_list[i - 1][1], new_crd_list[i][0], \
                        new_crd_list[i][1]
                    area += (x0 * y1 - x1 * y0)
                if area > 0:
                    new_cnxs_list.reverse()
                dual_dict[ring]['dual'] = np.asarray(new_cnxs_list)
            return

        ordered_cnxs_dual_to_node(local_dual, local_nodes)

        def ordered_cnxs_dual_to_dual(dual_dict, node_dict):
            for ring in dual_dict.keys():
                cnxs_list = dual_dict[ring]['net']
                if not isinstance(cnxs_list, list):
                    cnxs_list = cnxs_list.tolist()
                new_cnxs_list = []
                new_crd_list = []
                new_cnxs_list.append(cnxs_list[0])
                new_crd_list.append(dual_dict['{:}'.format(int(cnxs_list[0]))]['crds'])
                i = 0
                while i < len(cnxs_list) - 1:
                    ring0 = int(new_cnxs_list[i])
                    connected_to_0 = dual_dict['{:}'.format(ring0)]['net'].tolist()
                    options = set(connected_to_0).intersection(cnxs_list)
                    for val in new_cnxs_list:
                        if val in options:
                            options.remove(val)
                    options = list(options)
                    new_cnxs_list.append(options[0])
                    new_crd_list.append(dual_dict['{:}'.format(int(options[0]))]['crds'])
                    i += 1

                area = 0
                for i in range(len(cnxs_list)):
                    x0, y0, x1, y1 = new_crd_list[i - 1][0], new_crd_list[i - 1][1], new_crd_list[i][0], \
                        new_crd_list[i][1]
                    area += (x0 * y1 - x1 * y0)
                if area > 0:
                    new_cnxs_list.reverse()
                dual_dict[ring]['net'] = np.asarray(new_cnxs_list)
            return

        ordered_cnxs_dual_to_dual(local_dual, local_nodes)

        deleted_nodes = self.deleted_nodes
        deleted_nodes.sort(reverse=True)
        print(deleted_nodes)
        maxNodeSize = max([local_nodes[i]['net'].shape[0] for i in local_nodes.keys()])
        with open(input_folder + '/test_A_aux.dat', 'r') as f:
            pbc = np.genfromtxt(f, skip_header=3)
        with open(folder + '/test{:}_A_aux.dat'.format(which_ring), 'w') as f:
            f.write('{:}\n'.format(len(local_nodes.keys())))
            f.write('{:<10}{:<10}\n'.format(maxNodeSize, maxNodeSize))  #
            f.write('2DE\n')
            f.write('{:<26}{:<26}\n'.format(pbc[0, 0], pbc[0, 1]))
            f.write('{:<26}{:<26}\n'.format(pbc[1, 0], pbc[1, 1]))

        with open(folder + '/test{:}_A_crds.dat'.format(which_ring), 'w') as f:
            for node in local_nodes.keys():
                for j in range(2):
                    f.write('{:<10.4f}'.format(local_nodes[node]['crds'][j]))
                f.write('\n')
        with open(folder + '/test{:}_A_net.dat'.format(which_ring), 'w') as f:
            for node in local_nodes.keys():
                for j in range(3):
                    ConnectedNode = int(local_nodes[node]['net'][j])
                    for k in deleted_nodes:
                        if ConnectedNode >= k:
                            ConnectedNode -= 1
                    if ConnectedNode < 0:
                        print("Error in Atom Repositioning")
                    f.write('{:<10}'.format(int(ConnectedNode)))
                f.write('\n')

        deleted_rings = self.rings_to_remove
        deleted_rings.sort(reverse=True)

        with open(folder + '/test{:}_A_dual.dat'.format(which_ring), 'w') as f:
            for node in local_nodes.keys():
                for j in range(3):
                    ConnectedRing = int(local_nodes[node]['dual'][j])
                    for k in deleted_rings:
                        if ConnectedRing >= k:
                            ConnectedRing -= 1
                    if ConnectedRing < 0:
                        print("Error in Ring Repositioning")
                    f.write('{:<10}'.format(int(ConnectedRing)))
                f.write('\n')
        maxRingSize = max([local_dual[i]['net'].shape[0] for i in local_dual.keys()])
        with open(input_folder + '/test_B_aux.dat', 'r') as f:
            pbc = np.genfromtxt(f, skip_header=3)
        with open(folder + '/test{:}_B_aux.dat'.format(which_ring), 'w') as f:
            f.write('{:}\n'.format(len(local_dual.keys())))
            f.write('{:<10}{:<10}\n'.format(maxRingSize, maxRingSize))  #
            f.write('2DE\n')
            f.write('{:<26}{:<26}\n'.format(pbc[0, 0], pbc[0, 1]))
            f.write('{:<26}{:<26}\n'.format(pbc[1, 0], pbc[1, 1]))

        with open(folder + '/test{:}_B_crds.dat'.format(which_ring), 'w') as f:
            for ring in local_dual.keys():
                for j in range(2):
                    f.write('{:<10.4f}'.format(local_dual[ring]['crds'][j]))
                f.write('\n')
        with open(folder + '/test{:}_B_net.dat'.format(which_ring), 'w') as f:
            for ring in local_dual.keys():
                for j in range(local_dual[ring]['net'].shape[0]):
                    ConnectedRing = int(local_dual[ring]['net'][j])
                    for k in deleted_rings:
                        if ConnectedRing >= k:
                            ConnectedRing -= 1
                    if ConnectedRing < 0:
                        print("Error in Ring Repositioning")
                    if ConnectedRing >= len(local_dual.keys()):
                        print('Including Illegal Connections')
                    f.write('{:<10}'.format(int(ConnectedRing)))
                f.write('\n')
        with open(folder + '/test{:}_B_dual.dat'.format(which_ring), 'w') as f:
            for ring in local_dual.keys():
                for j in range(local_dual[ring]['dual'].shape[0]):
                    ConnectedNode = int(local_dual[ring]['dual'][j])
                    for k in deleted_nodes:
                        if ConnectedNode >= k:
                            ConnectedNode -= 1
                    if ConnectedNode < 0:
                        print("Error in Atom Repositioning")

                    f.write('{:<10}'.format(int(ConnectedNode)))
                f.write('\n')

        with open(folder + '/fixed_rings.dat', 'w') as f:
            f.write('{:}\n'.format(1))
            f.write('{:}\n'.format(self.new_ring))

        old_make_crds_marks_bilayer(folder, self.lj, True, True)
        print('Completed Lammps Scripts')

    def write(self):

        key = {}
        for node in self.nodesA.keys():
            node_val = int(node)
            for k in self.deleted_nodes:
                if node_val >= k:
                    node_val -= 1
            key['{:}'.format(node)] = node_val
        with open('Results/key.dat', 'w') as f:
            for i in self.deleted_nodes:
                f.write('{:<5}'.format(i))
            f.write('\n')
            for node in self.nodesA.keys():
                node_val = int(node)
                for k in self.deleted_nodes:
                    if node_val >= k:
                        node_val -= 1
                f.write('{:<10}{:<10}\n'.format(int(node), int(node_val)))

        deleted_nodes = []
        maxNodeSize = max([self.nodesA[i]['net'].shape[0] for i in self.nodesA.keys()])
        with open(input_folder + '/test_A_aux.dat', 'r') as f:
            pbc = np.genfromtxt(f, skip_header=3)
        with open('Results/testAD_A_aux.dat', 'w') as f:
            f.write('{:}\n'.format(len(self.nodesA.keys())))
            f.write('{:<10}{:<10}\n'.format(maxNodeSize, maxNodeSize))  #
            f.write('2DE\n')
            f.write('{:<26}{:<26}\n'.format(pbc[0, 0], pbc[0, 1]))
            f.write('{:<26}{:<26}\n'.format(pbc[1, 0], pbc[1, 1]))

        with open('Results/testAD_A_crds.dat', 'w') as f:
            for node in self.nodesA.keys():
                for j in range(2):
                    f.write('{:<10.4f}'.format(self.nodesA[node]['crds'][j]))
                f.write('\n')
        with open('Results/testAD_A_net.dat', 'w') as f:
            for node in self.nodesA.keys():
                for j in range(3):
                    ConnectedNode = int(self.nodesA[node]['net'][j])
                    # time.sleep(100)
                    for k in deleted_nodes:
                        if ConnectedNode >= k:
                            ConnectedNode -= 1
                    if ConnectedNode < 0:
                        print("Error in Atom Repositioning")
                    f.write('{:<10}'.format(int(ConnectedNode)))
                f.write('\n')

        with open('Results/test_AD_dual.dat', 'w') as f:
            for node in self.nodesA.keys():
                for j in range(3):
                    ConnectedRing = int(self.nodesA[node]['dual'][j])
                    for k in self.rings_to_remove:
                        if ConnectedRing >= k:
                            ConnectedRing -= 1
                    if ConnectedRing < 0:
                        print("Error in Ring Repositioning")
                    f.write('{:<10}'.format(int(ConnectedRing)))
                f.write('\n')
        maxRingSize = max([self.dualA[i]['net'].shape[0] for i in self.dualA.keys()])
        with open(input_folder + '/test_B_aux.dat', 'r') as f:
            pbc = np.genfromtxt(f, skip_header=3)
        with open('Results/testAD_B_aux.dat', 'w') as f:
            f.write('{:}\n'.format(len(self.dualA.keys())))
            f.write('{:<10}{:<10}\n'.format(maxRingSize, maxRingSize))
            f.write('2DE\n')
            f.write('{:<26}{:<26}\n'.format(pbc[0, 0], pbc[0, 1]))
            f.write('{:<26}{:<26}\n'.format(pbc[1, 0], pbc[1, 1]))

        with open('Results/testAD_B_crds.dat', 'w') as f:
            for ring in self.dualA.keys():
                for j in range(2):
                    f.write('{:<10.4f}'.format(self.dualA[ring]['crds'][j]))
                f.write('\n')
        with open('Results/testAD_B_net.dat', 'w') as f:
            for ring in self.dualA.keys():
                for j in range(self.dualA[ring]['net'].shape[0]):
                    ConnectedRing = int(self.dualA[ring]['net'][j])
                    for k in self.rings_to_remove:
                        if ConnectedRing >= k:
                            ConnectedRing -= 1
                    if ConnectedRing < 0:
                        print("Error in Ring Repositioning")
                    if ConnectedRing >= len(self.dualA.keys()):
                        print('Including Illegal Connections')
                    if int(ConnectedRing) == 1599:
                        print('Fucked here')
                    if int(ring) == 0:
                        print(ConnectedRing)
                    f.write('{:<10}'.format(int(ConnectedRing)))
                f.write('\n')
        with open('Results/testAD_B_dual.dat', 'w') as f:
            for ring in self.dualA.keys():
                for j in range(self.dualA[ring]['dual'].shape[0]):
                    ConnectedNode = int(self.dualA[ring]['dual'][j])
                    for k in self.deleted_nodes:
                        if ConnectedNode >= k:
                            ConnectedNode -= 1
                    if ConnectedNode < 0:
                        print("Error in Atom Repositioning")

                    f.write('{:<10}'.format(int(ConnectedNode)))
                f.write('\n')

        with open('Results/fixed_rings.dat', 'w') as f:
            f.write('{:}\n'.format(1))
            f.write('{:}\n'.format(self.new_ring))

        def ordered_cnxs_dual_to_node(dual_dict, node_dict):
            for ring in dual_dict.keys():
                print(ring)
                cnxs_list = dual_dict[ring]['dual']
                if not isinstance(cnxs_list, list):
                    cnxs_list = cnxs_list.tolist()
                new_cnxs_list = []
                new_crd_list = []
                new_cnxs_list.append(cnxs_list[0])
                new_crd_list.append(node_dict['{:}'.format(int(cnxs_list[0]))]['crds'])
                i = 0
                print(len(cnxs_list))
                while i < len(cnxs_list) - 1:
                    print(i)
                    node0 = int(new_cnxs_list[i])
                    print(new_cnxs_list)
                    connected_to_0 = node_dict['{:}'.format(node0)]['net'].tolist()
                    print(connected_to_0)

                    options = set(connected_to_0).intersection(cnxs_list)
                    print(options)
                    for val in new_cnxs_list:
                        if val in options:
                            options.remove(val)
                    print(options)
                    options = list(options)
                    new_cnxs_list.append(options[0])
                    new_crd_list.append(node_dict['{:}'.format(int(options[0]))]['crds'])
                    i += 1

                area = 0
                for i in range(len(cnxs_list)):
                    x0, y0, x1, y1 = new_crd_list[i - 1][0], new_crd_list[i - 1][1], new_crd_list[i][0], \
                        new_crd_list[i][1]
                    area += (x0 * y1 - x1 * y0)
                if area > 0:
                    new_cnxs_list.reverse()
                dual_dict[ring]['dual'] = np.asarray(new_cnxs_list)
            return

        ordered_cnxs_dual_to_node(self.dualA, self.nodesA)

        def ordered_cnxs_dual_to_dual(dual_dict, node_dict):
            for ring in dual_dict.keys():
                print(ring)
                cnxs_list = dual_dict[ring]['net']
                if not isinstance(cnxs_list, list):
                    cnxs_list = cnxs_list.tolist()
                new_cnxs_list = []
                new_crd_list = []
                new_cnxs_list.append(cnxs_list[0])
                new_crd_list.append(dual_dict['{:}'.format(int(cnxs_list[0]))]['crds'])
                i = 0
                print(len(cnxs_list))
                while i < len(cnxs_list) - 1:
                    print(i)
                    ring0 = int(new_cnxs_list[i])
                    print(new_cnxs_list)
                    connected_to_0 = dual_dict['{:}'.format(ring0)]['net'].tolist()
                    print(connected_to_0)

                    options = set(connected_to_0).intersection(cnxs_list)
                    print(options)
                    for val in new_cnxs_list:
                        if val in options:
                            options.remove(val)
                    print(options)
                    options = list(options)
                    new_cnxs_list.append(options[0])
                    new_crd_list.append(dual_dict['{:}'.format(int(options[0]))]['crds'])
                    i += 1

                area = 0
                for i in range(len(cnxs_list)):
                    x0, y0, x1, y1 = new_crd_list[i - 1][0], new_crd_list[i - 1][1], new_crd_list[i][0], \
                        new_crd_list[i][1]
                    area += (x0 * y1 - x1 * y0)
                if area > 0:
                    new_cnxs_list.reverse()
                dual_dict[ring]['net'] = np.asarray(new_cnxs_list)
            return

        ordered_cnxs_dual_to_dual(self.dualA, self.nodesA)

        deleted_nodes = self.deleted_nodes
        deleted_nodes.sort(reverse=True)
        print(deleted_nodes)
        maxNodeSize = max([self.nodesA[i]['net'].shape[0] for i in self.nodesA.keys()])
        with open(input_folder + '/test_A_aux.dat', 'r') as f:
            pbc = np.genfromtxt(f, skip_header=3)
        with open('Results/testA_A_aux.dat', 'w') as f:
            f.write('{:}\n'.format(len(self.nodesA.keys())))
            f.write('{:<10}{:<10}\n'.format(maxNodeSize, maxNodeSize))  #
            f.write('2DE\n')
            f.write('{:<26}{:<26}\n'.format(pbc[0, 0], pbc[0, 1]))
            f.write('{:<26}{:<26}\n'.format(pbc[1, 0], pbc[1, 1]))

        with open('Results/testA_A_crds.dat', 'w') as f:
            for node in self.nodesA.keys():
                for j in range(2):
                    f.write('{:<10.4f}'.format(self.nodesA[node]['crds'][j]))
                f.write('\n')
        with open('Results/testA_A_net.dat', 'w') as f:
            for node in self.nodesA.keys():
                for j in range(3):
                    ConnectedNode = int(self.nodesA[node]['net'][j])
                    for k in deleted_nodes:
                        if ConnectedNode >= k:
                            ConnectedNode -= 1
                    if ConnectedNode < 0:
                        print("Error in Atom Repositioning")
                    f.write('{:<10}'.format(int(ConnectedNode)))
                f.write('\n')

        deleted_rings = self.rings_to_remove
        deleted_rings.sort(reverse=True)

        with open('Results/testA_A_dual.dat', 'w') as f:
            for node in self.nodesA.keys():
                for j in range(3):
                    ConnectedRing = int(self.nodesA[node]['dual'][j])
                    for k in deleted_rings:
                        if ConnectedRing >= k:
                            ConnectedRing -= 1
                    if ConnectedRing < 0:
                        print("Error in Ring Repositioning")
                    f.write('{:<10}'.format(int(ConnectedRing)))
                f.write('\n')
        maxRingSize = max([self.dualA[i]['net'].shape[0] for i in self.dualA.keys()])
        with open(input_folder + '/test_B_aux.dat', 'r') as f:
            pbc = np.genfromtxt(f, skip_header=3)
        with open('Results/testA_B_aux.dat', 'w') as f:
            f.write('{:}\n'.format(len(self.dualA.keys())))
            f.write('{:<10}{:<10}\n'.format(maxRingSize, maxRingSize))
            f.write('2DE\n')
            f.write('{:<26}{:<26}\n'.format(pbc[0, 0], pbc[0, 1]))
            f.write('{:<26}{:<26}\n'.format(pbc[1, 0], pbc[1, 1]))

        with open('Results/testA_B_crds.dat', 'w') as f:
            for ring in self.dualA.keys():
                for j in range(2):
                    f.write('{:<10.4f}'.format(self.dualA[ring]['crds'][j]))
                f.write('\n')
        with open('Results/testA_B_net.dat', 'w') as f:
            for ring in self.dualA.keys():
                for j in range(self.dualA[ring]['net'].shape[0]):
                    ConnectedRing = int(self.dualA[ring]['net'][j])
                    for k in deleted_rings:
                        if ConnectedRing >= k:
                            ConnectedRing -= 1
                    if ConnectedRing < 0:
                        print("Error in Ring Repositioning")
                    if ConnectedRing >= len(self.dualA.keys()):
                        print('Including Illegal Connections')
                    f.write('{:<10}'.format(int(ConnectedRing)))
                f.write('\n')
        with open('Results/testA_B_dual.dat', 'w') as f:
            for ring in self.dualA.keys():
                for j in range(self.dualA[ring]['dual'].shape[0]):
                    ConnectedNode = int(self.dualA[ring]['dual'][j])
                    for k in deleted_nodes:
                        if ConnectedNode >= k:
                            ConnectedNode -= 1
                    if ConnectedNode < 0:
                        print("Error in Atom Repositioning")

                    f.write('{:<10}'.format(int(ConnectedNode)))
                f.write('\n')

        with open('Results/fixed_rings.dat', 'w') as f:
            f.write('{:}\n'.format(1))
            f.write('{:}\n'.format(self.new_ring))

        make_crds_marks_bilayer()
        print('Completed Lammps Scripts')

    def left_click(self, x, y):
        self.image_coordinates = x, y
        print('Clicked Coords')
        x0, y0 = self.image_coordinates

        # Setup Variables
        Recognised = False
        RanOutOfOptions = False
        Recognised_Val = -1

        print('Checking for Duplicates')

        # Import node coordinates
        node_pos = nx.get_node_attributes(self.Nodes, 'pos')
        while not Recognised and not RanOutOfOptions:
            for node in node_pos:
                center_coordinates = np.array([int(self.scale * node_pos[node][0] + self.x_offset),
                                               int(self.scale * node_pos[node][1] + self.y_offset)])

                r0 = np.linalg.norm(np.subtract(center_coordinates, np.array([x0, y0])))

                if r0 < self.scale * 2 / 3:
                    Recognised = True
                    Recognised_Val = int(node)
                    print("Recognised : ", Recognised_Val)
            RanOutOfOptions = True

        if Recognised:

            print('Deleted : ', Recognised_Val)
            print('Nodes --- Connected to ', [int(node) for node in self.nodes['{:}'.format(Recognised_Val)]['net']])
            print('Rings --- Connected to ', [int(node) for node in self.nodes['{:}'.format(Recognised_Val)]['dual']])

            # add deleted node to list
            self.deleted_nodes.append(Recognised_Val)

            # show these rings as broken
            for ring in self.nodes['{:}'.format(Recognised_Val)]['dual']:
                if ring not in self.broken_rings:
                    self.broken_rings.append(int(ring))

            # add newly broken rings to list
            for ring in self.broken_rings:
                if Recognised_Val in self.dual['{:}'.format(ring)]['dual']:
                    self.dual['{:}'.format(ring)]['dual'] = self.dual['{:}'.format(ring)]['dual'].tolist()
                    self.dual['{:}'.format(ring)]['dual'].remove(Recognised_Val)
                    self.dual['{:}'.format(ring)]['dual'] = np.asarray(self.dual['{:}'.format(ring)]['dual'])

            # add the newly undercoordinated nodes to list
            for node in self.nodes['{:}'.format(Recognised_Val)]['net']:
                self.undercoordinated.append(int(node))
                node = int(node)
                self.nodes['{:}'.format(node)]['net'] = self.nodes['{:}'.format(node)]['net'].tolist()
                self.nodes['{:}'.format(node)]['net'].remove(Recognised_Val)
                self.nodes['{:}'.format(node)]['net'] = np.asarray(self.nodes['{:}'.format(node)]['net'])

                if node in self.deleted_nodes:
                    self.undercoordinated.remove(node)
            # check nodes
            for node in self.nodes.keys():
                for deleted_node in self.deleted_nodes:
                    #                        print(self.nodes[node]['net'].tolist())
                    if int(deleted_node) in [int(i) for i in self.nodes[node]['net'].tolist()]:
                        print('Broken')
            print('########')

            del self.nodes['{:}'.format(Recognised_Val)]

            self.clone = self.original_image.copy()
            self.refresh_new_cnxs([])
            cv2.imshow("image", self.clone)
            print("Waiting...")

    def merge_rings_A(self, rings_to_merge):
        # ids of merged ring
        merged_ring = min(rings_to_merge)
        # we define the new ring as the lowest ring count of all rings to merge
        print('New Ring id   : ', merged_ring)
        # Rings connected to new merged ring
        connected_rings = []
        for ring in rings_to_merge:
            for connected in self.dualA['{:}'.format(int(ring))]['net']:
                if int(connected) not in rings_to_merge and int(connected) not in connected_rings:
                    connected_rings.append(int(connected))
        print("Connected Rings : ", connected_rings)

        # update merged ring connections
        self.dualA['{:}'.format(merged_ring)]['net'] = np.asarray(connected_rings)

        self.rings_to_remove = rings_to_merge.copy()
        self.rings_to_remove.remove(merged_ring)

        # remove connections to unmerged rings
        for ring in connected_rings:
            for vals in self.rings_to_remove:
                if vals in self.dualA['{:}'.format(ring)]['net']:

                    self.dualA['{:}'.format(ring)]['net'] = self.dualA['{:}'.format(ring)]['net'].tolist()
                    self.dualA['{:}'.format(ring)]['net'].remove(vals)
                    if merged_ring not in self.dualA['{:}'.format(ring)]['net']:
                        self.dualA['{:}'.format(ring)]['net'].append(merged_ring)
                    self.dualA['{:}'.format(ring)]['net'] = np.asarray(self.dualA['{:}'.format(ring)]['net'])

        connected_nodes = []

        # replace all ring-node connections
        for ring in rings_to_merge:
            for node in self.dualA['{:}'.format(ring)]['dual']:
                if node not in connected_nodes:
                    connected_nodes.append(node)
        self.dualA['{:}'.format(merged_ring)]['dual'] = np.asarray(connected_nodes)

        # replace all node-ring connections
        for node in self.nodesA.keys():
            for ring in self.nodesA[node]['dual']:
                if ring in rings_to_merge:
                    index = np.where(self.nodesA[node]['dual'] == ring)
                    self.nodesA[node]['dual'][index] = merged_ring

        # update in networkx

        new_ring_crds = self.dualA['{:}'.format(merged_ring)]['crds']
        for ring in rings_to_merge:
            if ring != merged_ring:
                new_ring_crds = np.add(new_ring_crds, self.dualA['{:}'.format(ring)]['crds'])

        new_crds = np.divide(new_ring_crds, len(rings_to_merge))
        self.dualA['{:}'.format(merged_ring)]['crds'] = new_crds

        for ring in rings_to_merge:
            if ring != merged_ring:
                print('deleting : ', int(ring))
                del self.dualA['{:}'.format(int(ring))]

        print("Merged Rings : ", rings_to_merge)
        return merged_ring

    def local_merge_rings(self, rings_to_merge, local_nodes, local_dual):
        # ids of merged ring
        merged_ring = min(rings_to_merge)
        # we define the new ring as the lowest ring count of all rings to merge
        print('New Ring id   : ', merged_ring)
        # Rings connected to new merged ring
        connected_rings = []
        for ring in rings_to_merge:
            for connected in local_dual['{:}'.format(int(ring))]['net']:
                if int(connected) not in rings_to_merge and int(connected) not in connected_rings:
                    connected_rings.append(int(connected))
        print("Connected Rings : ", connected_rings)

        # update merged ring connections
        local_dual['{:}'.format(merged_ring)]['net'] = np.asarray(connected_rings)

        self.rings_to_remove = rings_to_merge.copy()
        self.rings_to_remove.remove(merged_ring)

        # remove connections to unmerged rings
        for ring in connected_rings:
            for vals in self.rings_to_remove:
                if vals in local_dual['{:}'.format(ring)]['net']:

                    local_dual['{:}'.format(ring)]['net'] = local_dual['{:}'.format(ring)]['net'].tolist()
                    local_dual['{:}'.format(ring)]['net'].remove(vals)
                    if merged_ring not in local_dual['{:}'.format(ring)]['net']:
                        local_dual['{:}'.format(ring)]['net'].append(merged_ring)
                    local_dual['{:}'.format(ring)]['net'] = np.asarray(local_dual['{:}'.format(ring)]['net'])

        connected_nodes = []

        # replace all ring-node connections
        for ring in rings_to_merge:
            for node in local_dual['{:}'.format(ring)]['dual']:
                if node not in connected_nodes:
                    connected_nodes.append(node)
        local_dual['{:}'.format(merged_ring)]['dual'] = np.asarray(connected_nodes)

        # replace all node-ring connections
        for node in local_nodes.keys():
            for ring in local_nodes[node]['dual']:
                if ring in rings_to_merge:
                    index = np.where(local_nodes[node]['dual'] == ring)
                    local_nodes[node]['dual'][index] = merged_ring

        # update in networkx

        new_ring_crds = local_dual['{:}'.format(merged_ring)]['crds']
        for ring in rings_to_merge:
            if ring != merged_ring:
                new_ring_crds = np.add(new_ring_crds, local_dual['{:}'.format(ring)]['crds'])

        new_crds = np.divide(new_ring_crds, len(rings_to_merge))
        local_dual['{:}'.format(merged_ring)]['crds'] = new_crds

        for ring in rings_to_merge:
            if ring != merged_ring:
                print('deleting : ', int(ring))
                del local_dual['{:}'.format(int(ring))]

        print("Merged Rings : ", rings_to_merge)
        return merged_ring

    def check_undercoordinated_deleted(self):
        # Check no 'undercoordinated' atoms are also deleted
        print('Overlap : ', set(self.deleted_nodes).intersection(self.undercoordinated))

        while len(set(self.deleted_nodes).intersection(self.undercoordinated)) > 0:
            for val in set(self.deleted_nodes).intersection(self.undercoordinated):
                self.undercoordinated.remove(val)
        print('Overlap : ', set(self.deleted_nodes).intersection(self.undercoordinated))

    def rekey(self):
        # Deleting nodes changes the ordering
        key = {}
        for node in self.nodesA.keys():
            node_val = int(node)
            for k in self.deleted_nodes:
                if node_val >= k:
                    node_val -= 1
            key['{:}'.format(node)] = node_val
        return key

    def find_shared_cnxs_list(self, atom0):
        # Find Nodes associated with rings0
        rings0 = self.nodes['{:}'.format(atom0)]['dual']
        # Check which nodes share these rings
        shared_cnxs_list = [0 for i in range(len(self.undercoordinated))]
        for i in range(len(self.undercoordinated)):
            atom1 = int(self.undercoordinated[i])
            if int(atom1) not in [int(i) for i in self.nodes.keys()]:
                print('This atom is deleted, code breaks here')
                time.sleep(100)
            shared_cnxs_list[i] = len(set(rings0).intersection(self.nodes['{:}'.format(atom1)]['dual']))

        # 0 connections - More than one ring apart
        # 1 connection  - One ring apart
        # 2 connections - share an edge
        # 3 connections - same node!
        shared_cnxs_list = np.asarray(shared_cnxs_list)
        paths = np.where(shared_cnxs_list == 1)[0]
        print(paths)
        print("&" * 40)
        if len(paths) == 1:
            print("One Way Round the Ring...")
            return paths
        elif len(paths) == 2:
            print("Two Ways Round the Ring...")
            return paths
        elif len(paths) == 3:
            print("Three Ways Round the Ring...")
            return paths
        elif len(paths) == 0:
            paths = np.where(shared_cnxs_list == 2)[0]
            if len(paths) == 1:
                print("One Way Round the Ring via shared edge ...")
                return paths
            else:
                print("No Cross Ring Connections")
                exit(1)

    def local_find_shared_cnxs_list(self, atom0, local_nodes, local_undercoordinated):
        # Find Nodes associated with rings0
        rings0 = local_nodes['{:}'.format(atom0)]['dual']
        # Check which nodes share these rings
        shared_cnxs_list = [0 for i in range(len(local_undercoordinated))]
        for i in range(len(local_undercoordinated)):
            atom1 = int(local_undercoordinated[i])
            if int(atom1) not in [int(i) for i in local_nodes.keys()]:
                print('This atom is deleted, code breaks here')
                time.sleep(100)
            shared_cnxs_list[i] = len(set(rings0).intersection(local_nodes['{:}'.format(atom1)]['dual']))

        # 0 connections - More than one ring apart
        # 1 connection  - One ring apart
        # 2 connections - share an edge
        # 3 connections - same node!

        shared_cnxs_list = np.asarray(shared_cnxs_list)

        paths = np.where(shared_cnxs_list == 1)[0]

        if len(paths) == 1:
            print("One Way Round the Ring...")
            return paths
        elif len(paths) == 2:
            print("Two Ways Round the Ring...")
            return paths
        elif len(paths) == 3:
            print("Three Ways Round the Ring...")
            return paths
        elif len(paths) == 0:
            paths = np.where(shared_cnxs_list == 2)[0]
            if len(paths) == 1:
                print("One Way Round the Ring via shared edge ...")
                return paths
            elif len(paths) == 0:
                print("No Cross Ring Connections")
                exit(1)
            else:
                print("Multiple Ways Round the Ring via shared edge ...")

    def check_path(self, atom0, atomA):
        precheck_undercoordinated = copy.deepcopy(self.undercoordinated)
        precheck_undercoordinated.remove(atom0)
        precheck_undercoordinated.remove(atomA)

        # check for 3 memebered rings
        intersection = list(set(self.nodesA['{:}'.format(atom0)]['net'].tolist()).intersection(
            self.nodesA['{:}'.format(atomA)]['net'].tolist()))
        print(intersection, self.nodesA['{:}'.format(atom0)]['net'].tolist(),
              self.nodesA['{:}'.format(atomA)]['net'].tolist())

        if len(intersection) == 0:
            return True
        else:
            return False

    def extract_coordinates(self, event, x, y, flags, parameters):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_click(x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            print('\n\n#######################################################################\n\n')
            print('Recombining')

            def merge_rings_A(rings_to_merge):
                # ids of merged ring
                merged_ring = min(rings_to_merge)
                print('New Ring id   : ', merged_ring)
                # rings connected to new merged ring
                connected_rings = []
                for ring in rings_to_merge:
                    for connected in self.dualA['{:}'.format(int(ring))]['net']:
                        if int(connected) not in rings_to_merge and int(connected) not in connected_rings:
                            connected_rings.append(int(connected))
                print('\n\n\n')
                print('Connected Rings : ', connected_rings)
                print(len(connected_rings))
                # update merged ring connections
                self.dualA['{:}'.format(merged_ring)]['net'] = np.asarray(connected_rings)

                self.rings_to_remove = rings_to_merge.copy()
                self.rings_to_remove.remove(merged_ring)
                # replace all ring-ring connections

                for ring in connected_rings:
                    for vals in self.rings_to_remove:
                        if vals in self.dualA['{:}'.format(ring)]['net']:

                            self.dualA['{:}'.format(ring)]['net'] = self.dualA['{:}'.format(ring)]['net'].tolist()
                            self.dualA['{:}'.format(ring)]['net'].remove(vals)
                            if merged_ring not in self.dualA['{:}'.format(ring)]['net']:
                                self.dualA['{:}'.format(ring)]['net'].append(merged_ring)
                            self.dualA['{:}'.format(ring)]['net'] = np.asarray(self.dualA['{:}'.format(ring)]['net'])

                connected_nodes = []
                for ring in rings_to_merge:
                    for node in self.dualA['{:}'.format(ring)]['dual']:
                        connected_nodes.append(node)
                self.dualA['{:}'.format(merged_ring)]['dual'] = np.asarray(connected_nodes)

                # replace all node-ring connections
                for node in self.nodesA.keys():
                    for ring in self.nodesA[node]['dual']:
                        if ring in rings_to_merge:
                            index = np.where(self.nodesA[node]['dual'] == ring)
                            self.nodesA[node]['dual'][index] = merged_ring

                # update in networkx
                new_ring_crds = self.dualA['{:}'.format(merged_ring)]['crds']
                for ring in rings_to_merge:
                    if ring != merged_ring:
                        new_ring_crds = np.add(new_ring_crds, self.dualA['{:}'.format(ring)]['crds'])

                new_crds = np.divide(new_ring_crds, len(rings_to_merge))
                self.dualA['{:}'.format(merged_ring)]['crds'] = new_crds

                for ring in rings_to_merge:
                    if ring != merged_ring:
                        print('deleting : ', int(ring))
                        del self.dualA['{:}'.format(int(ring))]

                print("Merged Rings : ", rings_to_merge)
                return merged_ring

            print('Deleted Nodes to recombine = ', self.deleted_nodes)
            print('Uncoordinated Nodes          ', self.undercoordinated)

            # Check that these two lists don't overlap!
            self.check_undercoordinated_deleted()

            # there are always two connection patterns
            self.nodesA = copy.deepcopy(self.nodes)
            self.dualA = copy.deepcopy(self.dual)
            self.nodesB = copy.deepcopy(self.nodes)
            self.dualB = copy.deepcopy(self.dual)

            self.rekey()

            # There is more than one route round a ring !
            # Pick Random Start Point
            atom0 = self.undercoordinated[0]
            print('Starting from ', atom0)

            # Add a graphical aid to show new connections
            atoms = []
            atoms.append(atom0)

            # Visualise
            self.clone = self.original_image.copy()
            self.refresh_new_cnxs(atoms)
            cv2.imshow("image", self.clone)
            print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            paths = self.find_shared_cnxs_list(atom0)
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
            atomA, atomB = -1, -1
            if self.check_path(atom0, self.undercoordinated[paths[0]]):
                atomA = self.undercoordinated[paths[0]]
            if len(paths) > 1:
                if self.check_path(atom0, self.undercoordinated[paths[1]]):
                    if atomA == -1:
                        atomA = self.undercoordinated[paths[1]]
                    elif atomA != -1:
                        atomB = self.undercoordinated[paths[1]]
            if len(paths) > 2:
                if self.check_path(atom0, self.undercoordinated[paths[2]]):
                    if atomA == -1:
                        atomA = self.undercoordinated[paths[2]]
                    elif atomB == -1:
                        atomB = self.undercoordinated[paths[2]]

            self.clone = self.original_image.copy()
            self.refresh_new_cnxs(atoms)
            cv2.imshow("image", self.clone)
            cv2.imshow('image', draw_line_widget.show_image())
            cv2.waitKey(1)
            print('############ Initial Broken Rings : ', self.broken_rings)
            print('>>>>>>>>>>>> Initial Undercoordinated : ', self.undercoordinated)
            # SPLIT HERE TO N OPTIONS
            self.undercoordinatedA = self.undercoordinated.copy()
            self.broken_ringsA = self.broken_rings.copy()

            self.undercoordinatedB = self.undercoordinated.copy()
            self.broken_ringsB = self.broken_rings.copy()

            self.cloneA = self.original_image.copy()
            self.cloneB = self.original_image.copy()

            print('Atom A : ', atomA, '    Atom B : ', atomB)

            # PATH A

            def create_initial_path(atom0, atomA):
                # FIND INITIAL CONNECTION
                # VS
                # FIND SECONDARY CONNECTION

                local_undercoordinated = self.undercoordinated.copy()
                local_broken_rings = self.broken_rings.copy()

                local_nodes = copy.deepcopy(self.nodes)
                local_dual = copy.deepcopy(self.dual)

                local_nodes['{:}'.format(atomA)]['net'] = local_nodes['{:}'.format(atomA)]['net'].tolist()
                local_nodes['{:}'.format(atomA)]['net'].append(atom0)
                local_nodes['{:}'.format(atomA)]['net'] = np.asarray(local_nodes['{:}'.format(atomA)]['net'])

                local_nodes['{:}'.format(atom0)]['net'] = local_nodes['{:}'.format(atom0)]['net'].tolist()
                local_nodes['{:}'.format(atom0)]['net'].append(atomA)
                local_nodes['{:}'.format(atom0)]['net'] = np.asarray(local_nodes['{:}'.format(atom0)]['net'])

                local_undercoordinated.remove(int(atom0))
                local_undercoordinated.remove(int(atomA))

                for ring in local_broken_rings:
                    if atom0 in local_dual['{:}'.format(ring)]['dual'] and atomA in local_dual['{:}'.format(ring)][
                            'dual']:
                        local_broken_rings.remove(ring)

                atoms.append(atomA)

                self.clone = self.original_image.copy()
                self.refresh_new_cnxs(atoms)
                cv2.imshow("image", self.clone)
                cv2.imshow('image', draw_line_widget.show_image())
                cv2.waitKey(1)

                print('############ One Connection Broken Rings : ', local_broken_rings)
                print('>>>>>>>>>>>> One Connection Undercoordinated : ', local_undercoordinated)

                return local_undercoordinated, local_broken_rings, local_nodes, local_dual

            def check_undercoordinated(local_undercoordinated, local_nodes):
                for i in local_undercoordinated:
                    if len(local_nodes['{:}'.format(int(i))]['net'].tolist()) == 3:
                        print(i, " is not Undercoordinated !!!!")
                        print(i, local_nodes['{:}'.format(i)]['net'].tolist())
                        time.sleep(100)
                    elif len(local_nodes['{:}'.format(int(i))]['net'].tolist()) > 3:
                        print(i, " is Overcoordinated !!!!")
                        print(i, local_nodes['{:}'.format(i)]['net'].tolist())
                        time.sleep(100)

            def create_secondary_path(atom0, atomA, local_undercoordinated, local_broken_rings, local_nodes, local_dual):

                # Check if the newly formed connection is to a site with no further connections
                if atomA not in local_undercoordinated:
                    # If so, we travel around to the next undercoordinated atom ...
                    paths = self.local_find_shared_cnxs_list(atomA, local_nodes, local_undercoordinated)
                    print("**** ", paths, " ****")
                    if not paths:
                        # If there are no further atoms, we break here
                        paths = self.local_find_shared_cnxs_list(atom0, local_nodes, local_undercoordinated)
                        if not paths:
                            print("Not Working")
                            exit(1)
                    else:
                        # Otherwise, we continue
                        if len(paths) == 1:
                            atomA = local_undercoordinated[paths[0]]
                        else:
                            atomA = local_undercoordinated[paths[0]]

                while len(local_undercoordinated) > 0:
                    if atomA not in local_undercoordinated:
                        print("*********************")
                        print("atomA : ", atomA)
                        print("local_undercoordinated : ", local_undercoordinated)
                        paths = self.local_find_shared_cnxs_list(atomA, local_nodes, local_undercoordinated)
                        if len(paths) != 0:
                            atomA = local_undercoordinated[paths[0]]
                        else:
                            "Ran Out of Connections"
                    # Check we haven't made an error!
                    check_undercoordinated(local_undercoordinated, local_nodes)

                    paths = self.local_find_shared_cnxs_list(atomA, local_nodes, local_undercoordinated)

                    print("Paths : ", paths)
                    print("local undercoordinated : ", local_undercoordinated)
                    # Check if our new atom is connected to any further atoms
                    if len(paths) == 0:

                        print("Atoms : ", atoms)
                        print("Uncoordinated Atoms : ", local_undercoordinated)

                        # If it isn't, code broken
                        print("Still Not Working")
                        exit(1)
                    atomZ = local_undercoordinated[paths[0]]

                    if atomZ not in local_nodes['{:}'.format(atomA)]['net'] and atomA not in \
                            local_nodes['{:}'.format(atomZ)]['net']:

                        local_nodes['{:}'.format(atomA)]['net'] = local_nodes['{:}'.format(atomA)]['net'].tolist()
                        local_nodes['{:}'.format(atomA)]['net'].append(atomZ)
                        local_nodes['{:}'.format(atomA)]['net'] = np.asarray(local_nodes['{:}'.format(atomA)]['net'])

                        local_nodes['{:}'.format(atomZ)]['net'] = local_nodes['{:}'.format(atomZ)]['net'].tolist()
                        local_nodes['{:}'.format(atomZ)]['net'].append(atomA)
                        local_nodes['{:}'.format(atomZ)]['net'] = np.asarray(local_nodes['{:}'.format(atomZ)]['net'])

                        local_undercoordinated.remove(atomA)
                        local_undercoordinated.remove(atomZ)

                        atoms.append(atomA)
                        atoms.append(atomZ)

                        self.clone = self.original_image.copy()
                        self.refresh_new_cnxs(atoms)
                        cv2.imshow("image", self.clone)
                        cv2.imshow('image', draw_line_widget.show_image())
                        cv2.waitKey(1)

                        for ring in local_broken_rings:
                            if atomA in [int(i) for i in local_dual['{:}'.format(ring)]['dual']] and atomZ in [int(i) for i in local_dual['{:}'.format(ring)]['dual']]:
                                local_broken_rings.remove(ring)
                    else:
                        print("Nodes Already Connected!")

                    atomA = atomZ

                print('############ Remaining Broken Rings : ', local_broken_rings)
                new_ring = self.local_merge_rings(local_broken_rings, local_nodes, local_dual)
                self.new_ring = new_ring
                print("Done A ! ")

                return new_ring

            def do(atom0, proposed_atom, proposed_string):

                if self.check_path(atom0, proposed_atom):
                    local_undercoordinated, local_broken_rings, local_nodes, local_dual = create_initial_path(atom0, proposed_atom)
                    create_secondary_path(atom0, proposed_atom, local_undercoordinated, local_broken_rings, local_nodes, local_dual)
                    self.refresh_new_cnxs(atoms)
                    cv2.imshow("image", self.clone)
                    cv2.imshow('image', draw_line_widget.show_image())
                    cv2.waitKey(1)
                    self.write_local(local_nodes, local_dual, "A")

                return

            do(atom0, atomA, "A")

    def show_image(self):
        return self.clone


def old_make_crds_marks_bilayer(folder, intercept_1, triangle_raft, bilayer):
    area = 1.00
    # NAMING

    AREA_SCALING = np.sqrt(area)
    UNITS_SCALING = 1 / 0.52917721090380
    si_si_distance = UNITS_SCALING * 1.609 * np.sqrt((32.0 / 9.0))
    si_o_length = UNITS_SCALING * 1.609
    o_o_distance = UNITS_SCALING * 1.609 * np.sqrt((8.0 / 3.0))
    h = UNITS_SCALING * np.sin((19.5 / 180) * np.pi) * 1.609

    displacement_vectors_norm = np.asarray([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 3]])
    displacement_vectors_factored = np.multiply(displacement_vectors_norm, 0.5)

    with open(folder + '/testA_A_aux.dat', 'r') as f:
        n_nodes = np.genfromtxt(f, max_rows=1)
        n_nodes = int(n_nodes)
    with open(folder + '/testA_A_aux.dat', 'r') as f:
        dims = np.genfromtxt(f, skip_header=3, skip_footer=1)
        dim_x, dim_y = dims[0], dims[1]

    dim = np.array([dim_x, dim_y, 30])
    with open(folder + '/testA_A_net.dat', 'r') as f:
        net = np.genfromtxt(f)

    with open(folder + '/testA_A_crds.dat', 'r') as f:
        node_crds = np.genfromtxt(f)

    with open(folder + '/testA_B_crds.dat', 'r') as f:
        dual_crds = np.genfromtxt(f)
    number_scaling = np.sqrt(dual_crds.shape[0] / B_crds.shape[0])
    print(dim_x, dim_y)
    dim_x, dim_y = number_scaling * dim_x, number_scaling * dim_y
    print(dim_x, dim_y)
    print(dual_crds.shape[0], B_crds.shape[0])
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

    print('Monolayer n {:}'.format(monolayer_crds.shape[0]))
    print('Monolayer harmpairs {:}'.format(monolayer_harmpairs.shape[0]))

    def plot_monolayer():
        plt.scatter(monolayer_crds[:, 0], monolayer_crds[:, 1], color='k', s=0.4)
        for i in range(monolayer_crds.shape[0]):
            atom_1_crds = monolayer_crds[int(monolayer_harmpairs[i, 0]), :]
            atom_2_crds = monolayer_crds[int(monolayer_harmpairs[i, 1]), :]
            atom_2_crds = np.add(atom_1_crds, pbc_v(atom_1_crds, atom_2_crds))
            plt.plot([atom_1_crds[0], atom_2_crds[0]], [atom_1_crds[1], atom_2_crds[1]], color='k')
        plt.show()

        print('plotting...?')

    with open(folder + '/PARM_Si.lammps', 'w') as f:
        f.write('bond_style harmonic        \n')
        f.write('bond_coeff 1 0.800 1.000  \n')
        f.write('angle_style cosine/squared       \n')
        f.write('angle_coeff 1 0.200 120   \n')

    with open(folder + '/Si.data', 'w') as f:
        f.write('DATA FILE Produced from netmc results (cf David Morley)\n')
        f.write('{:} atoms\n'.format(monolayer_crds.shape[0]))
        f.write('{:} bonds\n'.format(monolayer_harmpairs.shape[0]))
        f.write('{:} angles\n'.format(monolayer_angles.shape[0]))
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

    with open(folder + '/Si.in', 'w') as f:
        f.write('log Si.log\n')
        f.write('units                   electron                                                   \n')
        f.write('dimension               2                                                          \n')
        f.write('processors              * * *                                                        \n')
        f.write('boundary                p p p                                                      \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('variable time equal 25*0.02418884326                                               \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#read data\n')
        f.write('atom_style              molecular                                                  \n')
        f.write('read_data               Results/Si.data                                                  \n')
        f.write('timestep ${time}                                                                   \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#potential                                                                         \n')
        f.write('include                 Results/PARM_Si.lammps                                                \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#outputs                                                                           \n')
        f.write('thermo                  0                                                       \n')
        f.write('thermo_style            custom step pe ke epair ebond etotal vol temp              \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('dump                    1 all custom 1000000000 Si_dump.lammpstrj id element type x y z     \n')
        f.write('dump_modify             1 element Si                                             \n')
        f.write('thermo_modify           line yaml                                                  \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#initial minimisation                                                              \n')
        f.write('\n')
        f.write('min_style               sd                                                         \n')
        f.write('minimize        1.0e-6 0.0 1000000 10000000                                       \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#write_data              Si_results.dat                                              \n')
        f.write('#write_restart  C_results.rest                                                     \n')

    # Tersoff Graphene

    print("########### Tersoff Graphene ###############")
    tersoff_crds = np.multiply(node_crds, 1.42)
    with open(folder + '/PARM_C.lammps', 'w') as f:
        f.write('pair_style tersoff\n')
        f.write('pair_coeff * * Results/BNC.tersoff C\n')

    with open(folder + '/C.in', 'w') as f:
        f.write('log C.log\n')
        f.write('units                   metal                                                   \n')
        f.write('dimension               2                                                          \n')
        f.write('processors              * * *                                                        \n')
        f.write('boundary                p p p                                                      \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('variable time equal 1800.0                                               \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#read data\n')
        f.write('atom_style              atomic                                                  \n')
        f.write('read_data               Results/C.data                                                  \n')
        f.write('timestep 0.001                                                                   \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#potential                                                                         \n')
        f.write('include                 Results/PARM_C.lammps                                                \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#outputs                                                                           \n')
        f.write('thermo                  0                                                       \n')
        f.write('thermo_style            custom step pe ke epair ebond etotal vol temp              \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('dump                    1 all custom 1000000000 C_dump.lammpstrj id element type x y z     \n')
        f.write('dump_modify             1 element C                                             \n')
        f.write('thermo_modify           line yaml                                                  \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#initial minimisation                                                              \n')
        f.write('\n')
        f.write('min_style               sd                                                         \n')
        f.write('minimize        1.0e-6 0.0 1000000 10000000                                       \n')
        f.write('\n')
        f.write('#####################################################################              \n')
        f.write('\n')
        f.write('#write_data              C_results.dat                                              \n')
        f.write('#write_restart  C_results.rest                                                     \n')

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
                oxygen_0_crds = np.add(unperturbed_oxygen_0_crds, displacement_vectors_factored[selection, :])

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

        with open(folder + '/Si2O3.in', 'w') as f:
            f.write('log Si2O3.log\n')
            f.write('units                   electron                                                   \n')
            f.write('dimension               2                                                          \n')
            f.write('processors              * * *                                                      \n')
            f.write('boundary                p p p                                                      \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('variable time equal 25*0.02418884326                                               \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#read data\n')
            f.write('atom_style              molecular                                                  \n')
            f.write('read_data               Results/Si2O3.data                                                  \n')
            f.write('timestep ${time}                                                                   \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#potential                                                                         \n')
            f.write('include                 Results/PARM_Si2O3.lammps                                                \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#outputs                                                                           \n')
            f.write('thermo                  0                                                       \n')
            f.write('thermo_style            custom step pe ke epair ebond etotal vol temp              \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('dump                    1 all custom 1000000000 Si2O3_dump.lammpstrj id element type x y z     \n')
            f.write('dump_modify             1 element O Si                                             \n')
            f.write('thermo_modify           line yaml                                                  \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#initial minimisation                                                              \n')
            f.write('\n')
            f.write('min_style               sd                                                         \n')
            f.write('minimize        1.0e-6 1.0e-6 1000000 10000000                                       \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#write_data              Si2O3_results.dat                                              \n')
            f.write('#write_restart  C_results.rest                                                     \n')

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
            return
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

        with open(folder + '/SiO2.in', 'w') as f:
            f.write('log SiO2.log\n')
            f.write('units                   electron                                                   \n')
            f.write('dimension               3                                                          \n')
            f.write('processors              * * *                                                       \n')
            f.write('boundary                p p p                                                      \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('variable time equal 25*0.02418884326                                               \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#read data\n')
            f.write('atom_style              molecular                                                  \n')
            f.write('read_data               Results/SiO2.data                                                  \n')
            f.write('timestep ${time}                                                                   \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#potential                                                                         \n')
            f.write('include                 Results/PARM_SiO2.lammps                                                \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#outputs                                                                           \n')
            f.write('thermo                  0                                                       \n')
            f.write('thermo_style            custom step pe ke epair ebond etotal vol temp              \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('dump                    1 all custom 1000000000 SiO2_dump.lammpstrj id element type x y z     \n')
            f.write('dump_modify             1 element O Si                                             \n')
            f.write('thermo_modify           line yaml                                                  \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#initial minimisation                                                              \n')
            f.write('\n')
            f.write('min_style               sd                                                         \n')
            f.write('minimize        1.0e-6 1.0e-6 1000000 10000000                                       \n')
            f.write('\n')
            f.write('#####################################################################              \n')
            f.write('\n')
            f.write('#write_data              SiO2_results.dat                                              \n')
            f.write('#write_restart  C_results.rest                                                     \n')

    print('Finished')
    return


if __name__ == '__main__':
    draw_line_widget = DrawLineWidget()

    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)
