import os
import shutil
from pymatgen.core.structure import Structure


def copy_allfiles(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


contcar_path = os.getcwd() + "/CONTCAR"
contcar_list = os.listdir(contcar_path)
yaml_path = os.getcwd() + "/data/raw"

# Clear the files in the yaml file directory (raw data directory)
delete_list = os.listdir(yaml_path)
for f in delete_list:
    file_path = os.path.join(yaml_path, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Change the format of the CONTCAR files to yaml files
for i in range(len(contcar_list)):
    print(f'Processing {i + 1}/{len(contcar_list)}')
    # Change the file name to CONTCAR
    structure_from_contcar = Structure.from_file(contcar_path + "/" + contcar_list[i])
    structure_from_contcar.to(filename=yaml_path + "/" + contcar_list[i].strip(".contcar") + ".yaml")

yaml_list = os.listdir(yaml_path)
