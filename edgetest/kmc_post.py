"""
kmc_post.py: This script is used for post processing.
=====

Functions:
    1. dump_output: This function is used to generate dump output file.
    2. dump_perid_output: This function is used to generate perid dump output file.
    3. plot_msd: This function is used to generate msd plot.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dump_output(structure, time_step, file_name, mode='w'):
    item_id = np.arange(1, len(structure.coord) + 1).reshape((len(structure.coord), 1))
    item_elem = structure.elem_num.reshape((len(structure.coord), 1))
    image = np.c_[item_id, item_elem, structure.coord]
    box_size = structure.x + structure.y + structure.z
    line1 = "ITEM: TIMESTEP\n"
    line2 = str(time_step) + "          " + str(time_step * 10) + "\n"
    line3 = "ITEM: NUMBER OF ATOMS\n" + str(structure.n_atom) + "\n"
    line4 = "ITEM: BOX BOUNDS\n"
    line5 = str(box_size[0]) + " " + str(box_size[1]) + "\n"
    line6 = str(box_size[2]) + " " + str(box_size[3]) + "\n"
    line7 = str(box_size[4]) + " " + str(box_size[5]) + "\n"
    line8 = "ITEM: ATOMS id type x y z\n"
    # Append to the file
    with open(file_name, mode) as f:
        f.write(line1)
        f.write(line2)
        f.write(line3)
        f.write(line4)
        f.write(line5)
        f.write(line6)
        f.write(line7)
        f.write(line8)
        np.savetxt(f, image, fmt='%d %d %d %d %d ')
    f.close()


def dump_perid_output(structure, time_step, file_name, mode='w'):
    item_id = np.arange(1, len(structure.perid_coord) + 1).reshape((len(structure.perid_coord), 1))
    item_elem = structure.perid_elem_num.reshape((len(structure.perid_coord), 1))
    image = np.c_[item_id, item_elem, structure.perid_coord]
    box_size = structure.perid_x + structure.perid_y + structure.perid_z
    line1 = "ITEM: TIMESTEP\n"
    line2 = str(time_step) + "          " + str(time_step * 10) + "\n"
    line3 = "ITEM: NUMBER OF ATOMS\n" + str(structure.perid_n_atom) + "\n"
    line4 = "ITEM: BOX BOUNDS\n"
    line5 = str(box_size[0]) + " " + str(box_size[1]) + "\n"
    line6 = str(box_size[2]) + " " + str(box_size[3]) + "\n"
    line7 = str(box_size[4]) + " " + str(box_size[5]) + "\n"
    line8 = "ITEM: ATOMS id type x y z\n"
    # Append to the file
    with open(file_name, mode) as f:
        f.write(line1)
        f.write(line2)
        f.write(line3)
        f.write(line4)
        f.write(line5)
        f.write(line6)
        f.write(line7)
        f.write(line8)
        np.savetxt(f, image, fmt='%d %d %d %d %d ')
    f.close()


def plot_msd(file_name):
    data = pd.read_csv("msd.csv", delimiter=',', dtype=np.float64)
    data_x = data['msd_x']
    data_y = data['msd_y']
    data_z = data['msd_z']
    plt.plot(range(len(data_x)), data_x, label='x')
    plt.plot(range(len(data_y)), data_y, label='y')
    plt.plot(range(len(data_z)), data_z, label='z')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Mean square displacement')
    plt.savefig(file_name)
    plt.close()
