from pymatgen.core.structure import Structure, Molecule
import os
import json

contcar_path = os.getcwd() + "/CONTCAR"
contcar_list = os.listdir(contcar_path)

json_path = os.getcwd() + "/data/raw"

for i in range(len(contcar_list)):
    print(f'Processing {i+1}/{len(contcar_list)}')
    # Change the file name to CONTCAR
    os.rename(contcar_path + "/" + contcar_list[i], contcar_path + "/CONTCAR")

    structure_from_contcar = Structure.from_file(contcar_path + "/CONTCAR")
    structure_from_contcar.to(filename=json_path + "/" + contcar_list[i] + ".json")

    # Change CONTCAR back to the file name
    os.rename(contcar_path + "/CONTCAR", contcar_path + "/" + contcar_list[i])

json_list = os.listdir(contcar_path)
