from pymatgen.core.structure import Structure, Molecule
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

input_path = "S:/projects/10_LLTO_U2_N0_OV0/LLTO_U2_N0_OV0_00/STEP2/CONTCAR"

structure_from_contcar = Structure.from_file(input_path)
distance_matrix = []
for i in range(len(structure_from_contcar)):
    distance_list = []
    for j in range(len(structure_from_contcar)):
        distance_list.append(structure_from_contcar.get_distance(i, j))
    distance_matrix.append(distance_list)

distance_np_matrix = np.array(distance_matrix)

G = nx.Graph()

for i in range(len(distance_np_matrix)):
    for j in range(len(distance_np_matrix)):
        G.add_edge(i, j, weight=distance_np_matrix[i][j])

x = torch.eye(G.number_of_nodes(), dtype=torch.float)
adj = nx.to_scipy_sparse_array(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)



# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)

# sg = StructureGraph.with_local_env_strategy(structure_from_contcar, JmolNN())
# molecules = sg.get_subgraphs_as_molecules()
# print(molecules[0])
#
# new_molecule = Molecule.from_sites(sorted(molecules[0], key=lambda site: site.species))
# print(new_molecule)
#
# sg.draw_graph_to_file("Li7La3Zr2O12.png")

