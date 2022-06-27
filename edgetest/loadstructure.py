import pandas as pd
import kmc_post
from kmc_utils import Lattice

# Load the lattice coordinates file
data = pd.read_csv('llto.csv')
in_coord = data[['x', 'y', 'z']].values
in_elem = data['e'].values

# Set dictionary for elements
in_elem2num = {'na': 0, 'Li': 3, 'La': 57, 'Ti': 22, 'O': 8}
mov_elem = 'Li'

# Creat a lattice object
llto = Lattice(in_coord, in_elem, in_elem2num)

# Get the coordinates of each element
coord = llto.coord
elem = llto.elem
elem_dict = llto.elem_dict
elem_num = llto.elem_num

# Get the periodic structure of the lattice
perid_coord = llto.perid_coord
perid_elem = llto.perid_elem
perid_elem_dict = llto.perid_elem_dict
perid_elem_num = llto.perid_elem_num

for i in range(0, len(elem_dict[mov_elem])):
    # Get the coordinates of each element
    coord = llto.coord
    elem = llto.elem
    elem_dict = llto.elem_dict
    elem_num = llto.elem_num

    # Get the periodic structure of the lattice
    perid_coord = llto.perid_coord
    perid_elem = llto.perid_elem
    perid_elem_dict = llto.perid_elem_dict
    perid_elem_num = llto.perid_elem_num



kmc_post.dump_output(llto, 1, "./dump.llto", mode='w')

