import os.path as osp
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from pymatgen.core.structure import Structure, Molecule
import networkx as nx
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Visualization function for NX graph or PyTorch tensor
# def visualize(h, color, epoch=None, loss=None):
#     plt.figure(figsize=(7, 7))
#     plt.xticks([])
#     plt.yticks([])
#
#     if torch.is_tensor(h):
#         h = h.detach().cpu().numpy()
#         plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
#         if epoch is not None and loss is not None:
#             plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
#     else:
#         nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
#     plt.show()


def get_node_features(structure):
    """
    This will return a matrix / 2d array of the shape
    [Number of Nodes, Node Feature Size]
    """
    all_node_feats = []

    for atom in structure.atomic_numbers:
        node_feats = []
        # Feature 1: Atomic number
        node_feats.append(structure.atomic_numbers[atom])
        # Append node features to matrix
        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)


def get_edge_features(structure):
    """
    This will return a matrix / 2d array of the shape
    [Number of Edges, Edge Feature Size]
    """
    all_edge_feats = []
    for i in range(len(structure)):
        for j in range(len(structure)):
            edge_feats = []
            # Feature 1: Bond length
            edge_feats.append(structure.get_distance(i, j))
            all_edge_feats.append(edge_feats)

    all_edge_feats = np.asarray(all_edge_feats)
    return torch.tensor(all_edge_feats, dtype=torch.float)


def get_adjacency_info(structure):
    distance_matrix = []
    for i in range(len(structure)):
        distance_list = []
        for j in range(len(structure)):
            distance_list.append(structure.get_distance(i, j))
        distance_matrix.append(distance_list)

    distance_np_matrix = np.array(distance_matrix)
    G = nx.Graph()

    for i in range(len(distance_np_matrix)):
        for j in range(len(distance_np_matrix)):
            G.add_edge(i, j)

    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def get_labels(label):
    label = np.asarray(label)
    y = torch.from_numpy(label).type(torch.long)
    return y


class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data_' + str(data_index) + '.pt' for data_index in range(len(self.raw_file_names))]

    def download(self):
        pass

    def process(self):
        idx = 0
        label_list = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            structure_from_contcar = Structure.from_file(raw_path)

            # Get node features
            node_features = get_node_features(structure_from_contcar)
            # Get edge features
            edge_features = get_edge_features(structure_from_contcar)
            # Get adjacency information
            edge_index = get_adjacency_info(structure_from_contcar)
            # Get labels information
            label = get_labels([label_list[idx]])

            # Create data object
            data = Data(edge_index=edge_index, x=node_features, edge_attr=edge_features, y=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


dataset = MoleculeDataset(root="data/")
dataset.num_classes = 3

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(587)
dataset = dataset.shuffle()

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(587)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=4)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        print(pred)
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 11):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
