from copy import deepcopy

import ase.io
from ase.calculators.emt import EMT

from make_interface import Surfaces, Interfaces

view_dir = (1, 1, 1)
thickness = 10.0

# Find surfaces with specified view direction
bulk = ase.io.read("../bulk_structures/Ag.xsf")
surfaces = Surfaces(bulk, (0, 0, 0), 7.5, vector1=view_dir)
print(surfaces)

# Make a normal-reflected version of each:
bulk_reflected = ase.io.read("../bulk_structures/Ag.xsf")  # read chiral partner
surfaces_reflected = deepcopy(surfaces)
surfaces_reflected.bulk = bulk_reflected
print(surfaces_reflected)

# Compute zero-strain interfaces to fetch twin boundaries:
interfaces = Interfaces(surfaces, surfaces_reflected, strain_max=1E-6)
print(interfaces)

# Make and write all the twin boundary structures:
calc = EMT()
for i_interface in range(len(interfaces.strain)):
    slab, _, _ = interfaces.make_slab(
        i_interface, thickness, thickness, calc, reflect2=True, optimize_spacing=True
    )
    ase.io.write(f"grain_boundary_{i_interface}.xsf", slab)

