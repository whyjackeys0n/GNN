from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
import graphviz

# load the molecular crystal from a CIF file
# if a single CIF file contains multiple structures,
# you will need to use pymatgen.io.cif.CifParser(your_file).get_structures() instead
acetonitrile = Structure.from_file("Li7La3Zr2O12.cif")

# attempt to guess the bonds present using a bonding strategy
# (here JmolNN but others are available)
# this creates a StructureGraph
sg = StructureGraph.with_local_env_strategy(acetonitrile, JmolNN())

# now extract individual molecules, which are isolated subgraphs
# (i.e. groups of atoms not connected to each other)
molecules = sg.get_subgraphs_as_molecules()

# this gives a list of molecules
# here, there is only a single unique subgraph so only one is returned

print(molecules[0])
# Molecule Summary
# Site: H (-1.5303, 0.3235, -1.2207)
# Site: C (-0.7109, 0.7843, -0.7147)
# Site: H (-1.2961, 1.4446, -0.1864)
# Site: C (0.0955, -0.1069, 0.0933)
# Site: N (0.7421, -0.8002, 0.7281)
# Site: H (-0.1532, 1.2798, -1.3066)

# you can then sort the sites according to whatever logic you want,
# and create a new Molecule object from the sorted sites
new_molecule = Molecule.from_sites(sorted(molecules[0], key=lambda site: site.species))
print(new_molecule)

# Molecule Summary
# Site: N (0.7421, -0.8002, 0.7281)
# Site: C (-0.7109, 0.7843, -0.7147)
# Site: C (0.0955, -0.1069, 0.0933)
# Site: H (-1.5303, 0.3235, -1.2207)
# Site: H (-1.2961, 1.4446, -0.1864)
# Site: H (-0.1532, 1.2798, -1.3066)

sg.draw_graph_to_file("Li7La3Zr2O12.png")
