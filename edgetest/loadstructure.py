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

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


# 1 Construct a two-layer GNN model.
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = tf.relu(h)
        h = self.conv2(graph, h)
        return h


# 2 Generate data randomly.
edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])), num_nodes=54)
gx = dgl.to_networkx(edge_pred_graph)
nx.draw_networkx(gx)
plt.show()

edge_pred_graph.ndata['feature'] = torch.randn(54, 10)
edge_pred_graph.edata['feature'] = torch.randn(1480, 10)
edge_pred_graph.edata['label'] = torch.randn(1480)
edge_pred_graph.edata['train_mask'] = torch.zeros(1480, dtype=torch.bool).bernoulli(0.6)


# 3 Define predictor to compute feature of edge. Here gives two predictors `DotProductPredictor`.
class DotProductPredictor(nn.Module):
    # Compute the feature of edge by do dot production using the source node and dst
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(df.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


# 4 Define model.
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


node_features = edge_pred_graph.ndata['feature']
edge_label = edge_pred_graph.edata['label']
train_mask = edge_pred_graph.edata['train_mask']

# Train model.
model = Model(10, 20, 5)
opt = torch.optim.Adam(model.parameters())
for epoch in range(100):
    pred = model(edge_pred_graph, node_features)
    loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())
