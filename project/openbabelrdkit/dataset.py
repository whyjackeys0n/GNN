import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from rdkit.Chem import rdmolops
from openbabel import pybel
from rdkit import Chem


class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['1552087.cif', '7202540.cif', '9016301.cif']

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', 'data_3.pt']

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            mol = next(pybel.readfile("cif", raw_path))
            pybelmol = pybel.Outputfile("sdf", "outputfile.sdf", overwrite=True)
            pybelmol.write(mol)
            pybelmol.close()

            suppl = Chem.SDMolSupplier('outputfile.sdf')
            rdmol = next(suppl)

            # Get node features
            node_features = self._get_node_features(rdmol)
            # Get edge features
            edge_features = self._get_edge_features(rdmol)
            # Get adjacency information
            edge_index = self._get_adjacency_info(rdmol)

            # Create data object
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature Size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization state
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of Edges, Edge Feature Size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append edge features to matrix
            all_edge_feats.append(edge_feats)

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row, col)))
        coo = np.reshape(coo, (2, -1))
        return torch.tensor(coo, dtype=torch.long)

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

