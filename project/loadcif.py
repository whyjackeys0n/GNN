from rdkit import Chem
from openbabel import pybel

mol = next(pybel.readfile("cif", "./data/raw/Li7La3Zr2O12.cif"))
pybelmol = pybel.Molecule(mol)
pybelmol.write("sdf", "outputfile.sdf")
rdmol = Chem.SDMolSupplier('outputfile.sdf')

for atom in rdmol.GetAtoms():
    print(atom.GetAtomicNum())  # Atom number