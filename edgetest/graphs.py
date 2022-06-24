# -*- coding: UTF-8 -*-

import dgl
import torch
import numpy as np
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


# 1 Construct a two-layer GNN model.
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


# 2 Generate data randomly.
src = np.random.randint(0, 100, 500)
dst = np.random.randint(0, 100, 500)
edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
edge_pred_graph.ndata['feature'] = torch.randn(100, 10)
edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
edge_pred_graph.edata['label'] = torch.randn(1000)
edge_pred_graph.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)


# 3 Define predictor to compute feature of edge.
# Here gives two predictors `DotProductPredictor` and `MLPPredictor`, but we only apply the former predictor `DotProductPredictor`.
class DotProductPredictor(nn.Module):
    # Simply compute the feature of edge by do dot production using the source node and dst
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
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
edge_label = edge_pred_graph.edata['label']  # This is not label, but a value only. In this case we just do regression.
train_mask = edge_pred_graph.edata['train_mask']

# Train model.
model = Model(10, 20, 5)
opt = torch.optim.Adam(model.parameters())
for epoch in range(1000):
    pred = model(edge_pred_graph, node_features)
    loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

# Save model.
torch.save(model.state_dict(), 'edge_sage.m')
