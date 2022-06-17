from rdkit import Chem
from openbabel import pybel
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN

mol1 = next(pybel.readfile("cif", "./data/raw/Li7La3Zr2O12.cif"))
mol2 = next(pybel.readfile("cif", "./data/raw/LiLa5Ti8O24.cif"))
# pybelmol = pybel.Molecule(mol)
# pybelmol.write("sdf", "outputfile.sdf", overwrite=True)
pybelmol = pybel.Outputfile("sdf", "outputfile.sdf", overwrite=True)
pybelmol.write(mol1)
pybelmol.write(mol2)
pybelmol.close()


# suppl = Chem.SDMolSupplier('outputfile.sdf')
#
# mol1 = next(suppl)

# for atom in rdmol.GetAtoms():
#     print(atom.GetAtomicNum())  # Atom number


structure = Structure.from_file("data\\raw\\Li7La3Zr2O12.cif")
sg = StructureGraph.with_local_env_strategy(structure, JmolNN())
g = sg.graph


