from pymatgen.core.structure import Structure, Molecule
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_graph_info(dataset):
    print('** G attrs: ', '\n', dataset.keys)
    print('** G node features: ', '\n', dataset.x)
    print('** G node num: ', '\n', dataset.num_nodes)
    print('** G edge num: ', '\n', dataset.num_edges)
    print('** G node feature num: ', '\n', dataset.num_node_features)
    print('** G isolated nodes: ', '\n', dataset.has_isolated_nodes())
    print('** G self loops: ', '\n', dataset.has_self_loops())
    print('** G is_directed: ', '\n', dataset.is_directed())


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


input_path = "S:/projects/10_LLTO_U2_N0_OV0/LLTO_U2_N0_OV0_00/STEP2/CONTCAR"

structure_from_contcar = Structure.from_file(input_path)
distance_matrix = []
for i in range(len(structure_from_contcar)):
    distance_list = []
    for j in range(len(structure_from_contcar)):
        distance_list.append(structure_from_contcar.get_distance(i, j))
    distance_matrix.append(distance_list)

# Get node features from pymatgen structure
atomic_number_list = torch.tensor(structure_from_contcar.atomic_numbers)

# Get adjacency matrix from pymatgen structure
distance_np_matrix = np.array(distance_matrix)

# y label
y = torch.tensor(atomic_number_list)

G = nx.Graph()

for i in range(len(distance_np_matrix)):
    for j in range(len(distance_np_matrix)):
        G.add_edge(i, j, weight=distance_np_matrix[i][j])


adj = nx.to_scipy_sparse_array(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)

dataset = Data(x=atomic_number_list, edge_index=edge_index, num_classes=2)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(hidden_channels=16)
print(model)

model = GCN(hidden_channels=16)
model.eval()

out = model(dataset.x, dataset.edge_index)
visualize(out, color=dataset.y)
