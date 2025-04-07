import ase.io
from ase.calculators.emt import EMT
from mace.calculators import MACECalculator

from make_interface.surface import Surfaces


bulk = ase.io.read("TiO2.xsf")
surfaces = Surfaces(bulk, (1, 1, 0), 8.0)  # search for 110 surfaces
print(surfaces)

calc = MACECalculator(model_paths=["mace.model"])  # point to suitable MACE model
slab = surfaces.make_slab(0, 14.0, 10.0, calc)
ase.io.write("out-TiO2-110.xsf", slab)
