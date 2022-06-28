from mp_api import MPRester
from pymatgen.analysis.graphs import StructureGraph

MP_API = "zUkBM3Sid3ny0pDHwf1uFVlPLloCW5Df"

with MPRester(MP_API) as mpr:
    structure = mpr.get_structure_by_material_id("mp-942733")



