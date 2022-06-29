import os.path as osp
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from pymatgen.core.structure import Structure, Molecule
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def _get_node_features(structure):
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


def _get_edge_features(structure):
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


def _get_adjacency_info(structure):
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


class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [contcar + '.pt' for contcar in self.raw_file_names]

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            structure_from_contcar = Structure.from_file(raw_path)

            # Get node features
            node_features = _get_node_features(structure_from_contcar)

            # Get edge features
            edge_features = _get_edge_features(structure_from_contcar)

            # Get adjacency information
            edge_index = _get_adjacency_info(structure_from_contcar)

            # Create data object
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

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


# Test the dataset
dataset = MoleculeDataset(root="data/")

# Print the number of samples in the dataset
print(dataset[2].edge_index.t())
print(dataset[2].x)
print(dataset[2].edge_attr)
