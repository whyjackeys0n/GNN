import pandas as pd
import kmc_post
from kmc_utils import Lattice

import dgl
import torch
import numpy as np
import networkx as nx
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as tf
import dgl.function as df
import matplotlib.pyplot as plt

# Load the lattice coordinates file
data = pd.read_csv('llto.csv')
in_coord = data[['x', 'y', 'z']].values
in_elem = data['e'].values

# Set dictionary for elements
in_elem2num = {'na': 0, 'Li': 3, 'La': 57, 'Ti': 22, 'O': 8}
mov_elem = 'Li'

# Creat a lattice object
llto = Lattice(in_coord, in_elem, in_elem2num)

# Get the coordinates of each element
coord = llto.coord
elem = llto.elem
elem_dict = llto.elem_dict
elem_num = llto.elem_num

# Get the periodic structure of the lattice
perid_coord = llto.perid_coord
perid_elem = llto.perid_elem
perid_elem_dict = llto.perid_elem_dict
perid_elem_num = llto.perid_elem_num

src = []
dst = []

for i in range(0, len(elem_dict[mov_elem])):
    # Get the coordinates of each element
    coord = llto.coord
    elem = llto.elem
    elem_dict = llto.elem_dict
    elem_num = llto.elem_num

    # Get the periodic structure of the lattice
    perid_coord = llto.perid_coord
    perid_elem = llto.perid_elem
    perid_elem_dict = llto.perid_elem_dict
    perid_elem_num = llto.perid_elem_num

    for j in range(0, len(elem_dict[mov_elem])):
        x_dis = abs(elem_dict[mov_elem][i, 0] - elem_dict[mov_elem][j, 0])
        y_dis = abs(elem_dict[mov_elem][i, 1] - elem_dict[mov_elem][j, 1])
        z_dis = abs(elem_dict[mov_elem][i, 2] - elem_dict[mov_elem][j, 2])
        if x_dis == 1 and y_dis == 1:
            src.append(i)
            dst.append(j)
        elif x_dis == 1 and z_dis == 1:
            src.append(i)
            dst.append(j)
        elif y_dis == 1 and z_dis == 1:
            src.append(i)
            dst.append(j)

kmc_post.dump_output(llto, 1, "./dump.llto", mode='w')

edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])), num_nodes=54)
gx = dgl.to_networkx(edge_pred_graph)
nx.draw_networkx(gx)
plt.show()
