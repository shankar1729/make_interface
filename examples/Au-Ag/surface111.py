import ase.io
from ase.calculators.emt import EMT

from make_interface.surface import Surfaces


bulkAg = ase.io.read("Ag.xsf")
surfaces = Surfaces(bulkAg, (1, 1, 1), 6.0)  # search for 111 surfaces
print(surfaces)

slab = surfaces.make_slab(0, 10.0, 10.0, EMT())
ase.io.write("out-Ag111.xsf", slab)
