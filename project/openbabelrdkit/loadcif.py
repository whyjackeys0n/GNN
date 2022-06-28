from rdkit import Chem
from openbabel import pybel
import numpy as np
import torch
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN


def get_node_features(mol):
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

mol1 = next(pybel.readfile("cif", "data/raw/1552087.cif"))
mol2 = next(pybel.readfile("cif", "data/raw/7202540.cif"))
# pybelmol = pybel.Molecule(mol)
# pybelmol.write("sdf", "outputfile.sdf", overwrite=True)
pybelmol = pybel.Outputfile("sdf", "outputfile.sdf", overwrite=True)
pybelmol.write(mol1)
pybelmol.write(mol2)
pybelmol.close()


suppl = Chem.SDMolSupplier('outputfile.sdf')
rdmol1 = next(suppl)
rdmol2 = next(suppl)

for atom in rdmol1.GetAtoms():
    print(atom.GetAtomicNum())  # Atom number

kk = get_node_features(rdmol1)

# structure = Structure.from_file("data\\raw\\Li7La3Zr2O12.cif")
# sg = StructureGraph.with_local_env_strategy(structure, JmolNN())
# g = sg.graph


