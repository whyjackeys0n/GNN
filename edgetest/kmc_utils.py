"""
kmc_utils.py: Functions for the implementation of the KMC algorithm.
=====

Functions:
    1. elem_dict: Return a dictionary with element names as keys and a nparray of element coordinates as values.
    2. transfer_boundary: Transfer the boundary sites for hopping into the structure.
    3. create_supercell: Create a supercell of the structure.

Class:
    1. Lattice: The lattice class.
        1.1 create_periodic_boundary: Create the periodic boundary for the lattice.
"""

import numpy as np


def elem_dict(elem, coord):
    """
    Return a dictionary with element names as keys and a nparray of element coordinates as values
    :param elem: The element names.
    :param coord: The element coordinates.
    :return: A dictionary with element names as keys and a nparray of element coordinates as values.
    """
    elem_coord_dict = {}
    for i, ele in enumerate(elem):
        if ele in elem_coord_dict:
            elem_coord_dict[ele] = np.c_[elem_coord_dict[ele], coord[i]]
        else:
            elem_coord_dict[ele] = coord[i]
    for i, j in elem_coord_dict.items():
        elem_coord_dict[i] = np.transpose(np.array(j))

    return elem_coord_dict


def transfer_boundary(structure, x_next, y_next, z_next):
    """
    Transfer the boundary sites for hopping into the structure.
    :param structure: The structure object.
    :param x_next: The x coordinate of the next site in periodic boundary to be transferred.
    :param y_next: The y coordinate of the next site in periodic boundary to be transferred.
    :param z_next: The z coordinate of the next site in periodic boundary to be transferred.
    :return: The x, y, z coordinates of the next site inside the structure.
    """
    perid_x_next = x_next
    perid_y_next = y_next
    perid_z_next = z_next
    if x_next == structure.perid_x[0]:
        perid_x_next = structure.x[1]
        if y_next == structure.perid_y[0]:
            perid_y_next = structure.y[1]
            if z_next == structure.perid_z[0]:
                perid_z_next = structure.z[1]
            elif z_next == structure.perid_z[1]:
                perid_z_next = structure.z[0]
        elif y_next == structure.perid_y[1]:
            perid_y_next = structure.y[0]
            if z_next == structure.perid_z[0]:
                perid_z_next = structure.z[1]
            elif z_next == structure.perid_z[1]:
                perid_z_next = structure.z[0]
        elif z_next == structure.perid_z[0]:
            perid_z_next = structure.z[1]
        elif z_next == structure.perid_z[1]:
            perid_z_next = structure.z[0]
    elif x_next == structure.perid_x[1]:
        perid_x_next = structure.x[0]
        if y_next == structure.perid_y[0]:
            perid_y_next = structure.y[1]
            if z_next == structure.perid_z[0]:
                perid_z_next = structure.z[1]
            elif z_next == structure.perid_z[1]:
                perid_z_next = structure.z[0]
        elif y_next == structure.perid_y[1]:
            perid_y_next = structure.y[0]
            if z_next == structure.perid_z[0]:
                perid_z_next = structure.z[1]
            elif z_next == structure.perid_z[1]:
                perid_z_next = structure.z[0]
        elif z_next == structure.perid_z[0]:
            perid_z_next = structure.z[1]
        elif z_next == structure.perid_z[1]:
            perid_z_next = structure.z[0]
    elif y_next == structure.perid_y[0]:
        perid_y_next = structure.y[1]
        if z_next == structure.perid_z[0]:
            perid_z_next = structure.z[1]
        elif z_next == structure.perid_z[1]:
            perid_z_next = structure.z[0]
    elif y_next == structure.perid_y[1]:
        perid_y_next = structure.y[0]
        if z_next == structure.perid_z[0]:
            perid_z_next = structure.z[1]
        elif z_next == structure.perid_z[1]:
            perid_z_next = structure.z[0]
    elif z_next == structure.perid_z[0]:
        perid_z_next = structure.z[1]
    elif z_next == structure.perid_z[1]:
        perid_z_next = structure.z[0]
    return perid_x_next, perid_y_next, perid_z_next


def create_supercell(x, y, z, structure):
    """
    Create the supercell for the structure.
    :param x: Range x of the supercell.
    :param y: Range y of the supercell.
    :param z: Range z of the supercell.
    :param structure: The lattice structure object.
    :return: The supercell coordinates and element names.
    """
    supercell_coord = structure.coord.copy()
    supercell_elem = structure.elem.copy()
    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                si = i * structure.perid_x[1]
                sj = j * structure.perid_y[1]
                sk = k * structure.perid_z[1]
                supercell_coord = np.r_[supercell_coord, structure.coord + [si, sj, sk]]
                supercell_elem = np.r_[supercell_elem, structure.elem]
    supercell_coord = np.delete(supercell_coord, np.s_[0:structure.n_atom], axis=0)
    supercell_elem = np.delete(supercell_elem, np.s_[0:structure.n_atom], axis=0)
    return supercell_coord, supercell_elem


def find_coord_index(coord, target_coord):
    """
    Find the index of the target coordinate in the supercell.
    :param coord: The supercell coordinates.
    :param target_coord: The target coordinate.
    :return: The index of the target coordinate.
    """
    ind = np.where((coord[:, 0] == target_coord[0]) & (coord[:, 1] == target_coord[1]) & (coord[:, 2] == target_coord[2]))
    return ind


class Lattice:
    """
    Lattice class
    :param coord: nparray of coordinates
    :param elem: nparray of element names
    :param elem2num: dictionary of element names and their corresponding index
    """

    def __init__(self, coord, elem, elem2num):
        self.coord = coord
        self.elem = elem
        self.elem_num = np.array([elem2num[ele] for ele in elem])
        self.n_atom = len(self.coord)
        self.n_elem = len(np.unique(self.elem_num))
        self.elem_dict = elem_dict(self.elem, self.coord)

        self.perid_elem = np.array([])
        self.perid_coord = np.array([])
        self.create_periodic_boundary()
        self.perid_elem_num = np.array([elem2num[ele] for ele in self.perid_elem])
        self.perid_n_atom = len(self.perid_coord)
        self.perid_n_elem = len(np.unique(self.perid_elem_num))
        self.perid_elem_dict = elem_dict(self.perid_elem, self.perid_coord)

        self.x = [min(self.coord[:, 0]), max(self.coord[:, 0])]
        self.y = [min(self.coord[:, 1]), max(self.coord[:, 1])]
        self.z = [min(self.coord[:, 2]), max(self.coord[:, 2])]

        self.perid_x = [min(self.perid_coord[:, 0]), max(self.perid_coord[:, 0])]
        self.perid_y = [min(self.perid_coord[:, 1]), max(self.perid_coord[:, 1])]
        self.perid_z = [min(self.perid_coord[:, 2]), max(self.perid_coord[:, 2])]

    def create_periodic_boundary(self):
        """
        Create a periodic boundary for a lattice.
        """
        x_min = min(self.coord[:, 0])
        x_max = max(self.coord[:, 0])
        y_min = min(self.coord[:, 1])
        y_max = max(self.coord[:, 1])
        z_min = min(self.coord[:, 2])
        z_max = max(self.coord[:, 2])

        # x min face boundary
        x_min_coord = self.coord[self.coord[:, 0] == x_min]
        x_min_coord[:, 0] = x_max + 1
        x_min_elem = self.elem[self.coord[:, 0] == x_min]
        # start from the pristine lattice
        self.perid_coord = np.r_[self.coord, x_min_coord]
        self.perid_elem = np.r_[self.elem, x_min_elem]

        # x max face boundary
        x_max_coord = self.coord[self.coord[:, 0] == x_max]
        x_max_coord[:, 0] = x_min - 1
        x_max_elem = self.elem[self.coord[:, 0] == x_max]
        self.perid_coord = np.r_[self.perid_coord, x_max_coord]
        self.perid_elem = np.r_[self.perid_elem, x_max_elem]

        # y min face boundary
        y_min_coord = self.coord[self.coord[:, 1] == y_min]
        y_min_coord[:, 1] = y_max + 1
        y_min_elem = self.elem[self.coord[:, 1] == y_min]
        self.perid_coord = np.r_[self.perid_coord, y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, y_min_elem]

        # y max face boundary
        y_max_coord = self.coord[self.coord[:, 1] == y_max]
        y_max_coord[:, 1] = y_min - 1
        y_max_elem = self.elem[self.coord[:, 1] == y_max]
        self.perid_coord = np.r_[self.perid_coord, y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, y_max_elem]

        # z min face boundary
        z_min_coord = self.coord[self.coord[:, 2] == z_min]
        z_min_coord[:, 2] = z_max + 1
        z_min_elem = self.elem[self.coord[:, 2] == z_min]
        self.perid_coord = np.r_[self.perid_coord, z_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_elem]

        # z max face boundary
        z_max_coord = self.coord[self.coord[:, 2] == z_max]
        z_max_coord[:, 2] = z_min - 1
        z_max_elem = self.elem[self.coord[:, 2] == z_max]
        self.perid_coord = np.r_[self.perid_coord, z_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_elem]

        # z min x min edge boundary
        z_min_x_min_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 2] == z_min)]
        z_min_x_min_coord[:, 0] = x_max + 1
        z_min_x_min_coord[:, 2] = z_max + 1
        z_min_x_min_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_x_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_x_min_elem]

        # z min x max edge boundary
        z_min_x_max_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 2] == z_min)]
        z_min_x_max_coord[:, 0] = x_min - 1
        z_min_x_max_coord[:, 2] = z_max + 1
        z_min_x_max_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_x_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_x_max_elem]

        # z min y min edge boundary
        z_min_y_min_coord = self.coord[(self.coord[:, 1] == y_min) & (self.coord[:, 2] == z_min)]
        z_min_y_min_coord[:, 1] = y_max + 1
        z_min_y_min_coord[:, 2] = z_max + 1
        z_min_y_min_elem = self.elem[(self.coord[:, 1] == y_min) & (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_y_min_elem]

        # z min y max edge boundary
        z_min_y_max_coord = self.coord[(self.coord[:, 1] == y_max) & (self.coord[:, 2] == z_min)]
        z_min_y_max_coord[:, 1] = y_min - 1
        z_min_y_max_coord[:, 2] = z_max + 1
        z_min_y_max_elem = self.elem[(self.coord[:, 1] == y_max) & (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_y_max_elem]

        # z max x min edge boundary
        z_max_x_min_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 2] == z_max)]
        z_max_x_min_coord[:, 0] = x_max + 1
        z_max_x_min_coord[:, 2] = z_min - 1
        z_max_x_min_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_x_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_x_min_elem]

        # z max x max edge boundary
        z_max_x_max_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 2] == z_max)]
        z_max_x_max_coord[:, 0] = x_min - 1
        z_max_x_max_coord[:, 2] = z_min - 1
        z_max_x_max_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_x_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_x_max_elem]

        # z max y min edge boundary
        z_max_y_min_coord = self.coord[(self.coord[:, 1] == y_min) & (self.coord[:, 2] == z_max)]
        z_max_y_min_coord[:, 1] = y_max + 1
        z_max_y_min_coord[:, 2] = z_min - 1
        z_max_y_min_elem = self.elem[(self.coord[:, 1] == y_min) & (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_y_min_elem]

        # z max y max edge boundary
        z_max_y_max_coord = self.coord[(self.coord[:, 1] == y_max) & (self.coord[:, 2] == z_max)]
        z_max_y_max_coord[:, 1] = y_min - 1
        z_max_y_max_coord[:, 2] = z_min - 1
        z_max_y_max_elem = self.elem[(self.coord[:, 1] == y_max) & (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_y_max_elem]

        # x min y min edge boundary
        x_min_y_min_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_min)]
        x_min_y_min_coord[:, 0] = x_max + 1
        x_min_y_min_coord[:, 1] = y_max + 1
        x_min_y_min_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_min)]
        self.perid_coord = np.r_[self.perid_coord, x_min_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, x_min_y_min_elem]

        # x min y max edge boundary
        x_min_y_max_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_max)]
        x_min_y_max_coord[:, 0] = x_max + 1
        x_min_y_max_coord[:, 1] = y_min - 1
        x_min_y_max_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_max)]
        self.perid_coord = np.r_[self.perid_coord, x_min_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, x_min_y_max_elem]

        # x max y min edge boundary
        x_max_y_min_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_min)]
        x_max_y_min_coord[:, 0] = x_min - 1
        x_max_y_min_coord[:, 1] = y_max + 1
        x_max_y_min_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_min)]
        self.perid_coord = np.r_[self.perid_coord, x_max_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, x_max_y_min_elem]

        # x max y max edge boundary
        x_max_y_max_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_max)]
        x_max_y_max_coord[:, 0] = x_min - 1
        x_max_y_max_coord[:, 1] = y_min - 1
        x_max_y_max_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_max)]
        self.perid_coord = np.r_[self.perid_coord, x_max_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, x_max_y_max_elem]

        # z min x min y min point boundary
        z_min_x_min_y_min_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_min) &
                                             (self.coord[:, 2] == z_min)]
        z_min_x_min_y_min_coord[:, 0] = x_max + 1
        z_min_x_min_y_min_coord[:, 1] = y_max + 1
        z_min_x_min_y_min_coord[:, 2] = z_max + 1
        z_min_x_min_y_min_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_min) &
                                           (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_x_min_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_x_min_y_min_elem]

        # z min x min y max point boundary
        z_min_x_min_y_max_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_max) &
                                             (self.coord[:, 2] == z_min)]
        z_min_x_min_y_max_coord[:, 0] = x_max + 1
        z_min_x_min_y_max_coord[:, 1] = y_min - 1
        z_min_x_min_y_max_coord[:, 2] = z_max + 1
        z_min_x_min_y_max_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_max) &
                                           (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_x_min_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_x_min_y_max_elem]

        # z min x max y min point boundary
        z_min_x_max_y_min_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_min) &
                                             (self.coord[:, 2] == z_min)]
        z_min_x_max_y_min_coord[:, 0] = x_min - 1
        z_min_x_max_y_min_coord[:, 1] = y_max + 1
        z_min_x_max_y_min_coord[:, 2] = z_max + 1
        z_min_x_max_y_min_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_min) &
                                           (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_x_max_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_x_max_y_min_elem]

        # z min x max y max point boundary
        z_min_x_max_y_max_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_max) &
                                             (self.coord[:, 2] == z_min)]
        z_min_x_max_y_max_coord[:, 0] = x_min - 1
        z_min_x_max_y_max_coord[:, 1] = y_min - 1
        z_min_x_max_y_max_coord[:, 2] = z_max + 1
        z_min_x_max_y_max_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_max) &
                                           (self.coord[:, 2] == z_min)]
        self.perid_coord = np.r_[self.perid_coord, z_min_x_max_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_min_x_max_y_max_elem]

        # z max x min y min point boundary
        z_max_x_min_y_min_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_min) &
                                             (self.coord[:, 2] == z_max)]
        z_max_x_min_y_min_coord[:, 0] = x_max + 1
        z_max_x_min_y_min_coord[:, 1] = y_max + 1
        z_max_x_min_y_min_coord[:, 2] = z_min - 1
        z_max_x_min_y_min_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_min) &
                                           (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_x_min_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_x_min_y_min_elem]

        # z max x min y max point boundary
        z_max_x_min_y_max_coord = self.coord[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_max) &
                                             (self.coord[:, 2] == z_max)]
        z_max_x_min_y_max_coord[:, 0] = x_max + 1
        z_max_x_min_y_max_coord[:, 1] = y_min - 1
        z_max_x_min_y_max_coord[:, 2] = z_min - 1
        z_max_x_min_y_max_elem = self.elem[(self.coord[:, 0] == x_min) & (self.coord[:, 1] == y_max) &
                                           (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_x_min_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_x_min_y_max_elem]

        # z max x max y min point boundary
        z_max_x_max_y_min_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_min) &
                                             (self.coord[:, 2] == z_max)]
        z_max_x_max_y_min_coord[:, 0] = x_min - 1
        z_max_x_max_y_min_coord[:, 1] = y_max + 1
        z_max_x_max_y_min_coord[:, 2] = z_min - 1
        z_max_x_max_y_min_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_min) &
                                           (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_x_max_y_min_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_x_max_y_min_elem]

        # z max x max y max point boundary
        z_max_x_max_y_max_coord = self.coord[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_max) &
                                             (self.coord[:, 2] == z_max)]
        z_max_x_max_y_max_coord[:, 0] = x_min - 1
        z_max_x_max_y_max_coord[:, 1] = y_min - 1
        z_max_x_max_y_max_coord[:, 2] = z_min - 1
        z_max_x_max_y_max_elem = self.elem[(self.coord[:, 0] == x_max) & (self.coord[:, 1] == y_max) &
                                           (self.coord[:, 2] == z_max)]
        self.perid_coord = np.r_[self.perid_coord, z_max_x_max_y_max_coord]
        self.perid_elem = np.r_[self.perid_elem, z_max_x_max_y_max_elem]
