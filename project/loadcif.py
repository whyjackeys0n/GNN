from rdkit import Chem
from openbabel import pybel

mol = next(pybel.readfile("cif", "./data/raw/Li7La3Zr2O12.cif"))


