import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from pymatgen.core.structure import Structure
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        # gmp: Global Max Pooling
        # gap: Global Average Pooling
        # Twice size of the linear output layer
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden


def get_node_features(structure):
    """
    This will return a matrix / 2d array of the shape
    [Number of Nodes, Node Feature Size]
    """
    all_node_feats = []

    for atom in range(len(structure)):
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
    y = torch.from_numpy(label).type(torch.float)
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
        label_list = pd.read_csv("energy.csv")["E"].tolist()
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


data = MoleculeDataset(root="data/")
data.num_classes = 48

# Investigating the dataset
print("Dataset type:", type(data))
print("Dataset features:", data.num_features)
print("Dataset target:", data.num_classes)
print("Dataset length:", data.len)
# A sample
print("Dataset sample:", data[0])
print("Sample nodes:", data[0].num_nodes)
print("Sample edges:", data[0].num_edges)

embedding_size = 64

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 13
train_loader = DataLoader(data[:int(data_size * 0.8)], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


def train(data):
    # Enumerate over the data
    for batch in train_loader:
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch.y)
        loss.backward()
        # Update using the gradients
        optimizer.step()
    return loss, embedding


print("Starting training...")
losses = []
for epoch in range(350):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Train Loss {loss}")

# Visualize learning (training loss)
losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i, l in enumerate(losses_float)]
sns.lineplot(loss_indices, losses_float)
plt.show()

# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()

sns.scatterplot(df["y_real"].to_list(), sum(df["y_pred"], []))
plt.show()
