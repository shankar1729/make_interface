import os
import ase.io
from ase.calculators.emt import EMT
from mace.calculators import MACECalculator
import numpy as np 
from make_interface import Surfaces, Interfaces
from ase.optimize import LBFGS
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
import warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

Lmax = 40.0
strain_max = 1

print(os.getcwd())
original_directory=os.getcwd()

 
#Select Calculator
calc = MACECalculator(model_paths=["MACE-matpes-pbe-omat-ft.model"], device="cuda") #Add Specific location of your mace calcualtor

#Relax material 1 and create surfaces
os.chdir(original_directory)
print("\n---- Mat 1 ----\n")
bulkMat1 = ase.io.read("CHA.xsf")
bulkMat1.calc = calc
#LBFGS(FrechetCellFilter(bulkMat1), logfile="-").run(fmax=0.01)
opt = LBFGS(FrechetCellFilter(bulkMat1), logfile='opt1.log')
converged = opt.run(fmax=0.01, steps=2000)

print("Optimization converged:", opt.converged())
print("Number of steps taken:", opt.nsteps)
surfacesMat1 = Surfaces(bulkMat1, (0, 0, 0), Lmax)  # all surfaces
print(surfacesMat1)

# Relax material 2 and create surfaces
print("\n---- Mat2 ----\n")
bulkMat2 = ase.io.read("AEI.xsf")
bulkMat2.calc = calc
opt = LBFGS(FrechetCellFilter(bulkMat2), logfile='opt2.log')
converged = opt.run(fmax=0.01, steps=2000)

print("Optimization converged:", opt.converged())
print("Number of steps taken:", opt.nsteps)

surfacesMat2 = Surfaces(bulkMat2, (0, 0, 0), Lmax)  # all surfaces
print(surfacesMat2)

#Create Mat1/ MAt2 Interfaces 
print("\n---- Mat1 - Mat2 interface ----\n")

interfaces = Interfaces(surfacesMat1, surfacesMat2, strain_max)
sorted_indices = np.argsort(interfaces.strain_max[:])  #Sort by strain
Area_temp=interfaces.area[sorted_indices[:200]] #Sort the top 200 by Area 
indices_temp=sorted_indices[:200]
sorted_indices_2 = np.argsort(Area_temp)
#print(indices_temp[sorted_indices_2[:20]])
length=10 # Total number of top interfaces to study
indices=indices_temp[sorted_indices_2[:length]]



for i_interface in indices:
    Area=interfaces.area[i_interface]
    Strain=interfaces.strain_max[i_interface]
    surf1=''.join(map(str, (np.array(interfaces.surfaces1.sup[interfaces.index1[i_interface]][:,2]))))
    surf2=''.join(map(str, (np.array(interfaces.surfaces2.sup[interfaces.index2[i_interface]][:,2]))))
    print([surf1, surf2, Area, Strain])


#Interface Optimization
#Loop through top 10 interfaces 

print(original_directory)
#int_energy=np.zeros((length, 3))
#int_energy = [None] * (length+1)
int_energy = [[] for _ in range(length+1)]
data=f"surf1, surf2, Strain, Area, interface_energy, NofBMat1, NofBMat2, slab, Mat1, Mat2, converged"
int_energy[0]=data
count=0
for i_interface in indices:
    print(count)
    print(i_interface)
    surf1=''.join(map(str, (np.array(interfaces.surfaces1.sup[interfaces.index1[i_interface]][:,2]))))
    surf2=''.join(map(str, (np.array(interfaces.surfaces2.sup[interfaces.index2[i_interface]][:,2]))))
    print([surf1, surf2])
    folder_name=f'{count}'
    os.makedirs(folder_name, exist_ok=True)
    os.chdir(folder_name)
    slab, slabMat1, slabMat2 = interfaces.make_slab(i_interface, 20, 20, calc, 10)
    n_bulk_Mat1 = len(slabMat1) / len(bulkMat1)
    n_bulk_Mat2 = len(slabMat2) / len(bulkMat2)
    slab.calc = calc
    print("Performing interface relaxation")
    #opt = BFGS(slab, trajectory='opt.traj', logfile='opt.log')
    opt = LBFGS(FrechetCellFilter(slab), logfile='opt.log')
    converged = opt.run(fmax=0.01, steps=2000)
    print("Optimization converged:", opt.converged())
    print("Number of steps taken:", opt.nsteps)
    conv = opt.converged()
    base_area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    interface_energy = (slab.get_potential_energy() - (bulkMat1.get_potential_energy() * n_bulk_Mat1  + bulkMat2.get_potential_energy() * n_bulk_Mat2)) / (2 * base_area)
    
    slab_pot=slab.get_potential_energy()
    mat1_pot=bulkMat1.get_potential_energy()
    mat2_pot=bulkMat2.get_potential_energy()
    print([slab_pot, mat1_pot, mat2_pot])
    print(f"{interface_energy = :.4f} eV/A^2")
    filename1=f'interface_Mat1_{surf1}_Mat2_{surf2}.xsf'
    filename2=f'interface_Mat1_{surf1}_Mat2_{surf2}.lammps.data'
    filename3=f'Mat1.lammps.data'
    filename4=f'Mat2.lammps.data'
    filename5=f'interface.lammps.data'

    #Writing out the structures in lammps format for further analysis with other potentials if required
    Area=interfaces.area[i_interface]
    Strain=interfaces.strain_max[i_interface]
    ase.io.write(filename1, slab)
    ase.io.write(filename2, slab, format='lammps-data', atom_style='charge')
    ase.io.write(filename3, bulkMat1, format='lammps-data', atom_style='charge' )
    ase.io.write(filename4, bulkMat2, format='lammps-data', atom_style='charge' )
    ase.io.write(filename5, slab, format='lammps-data', atom_style='charge')
    data=f"{surf1}, {surf2}, {Strain}, {Area}, {interface_energy}, {n_bulk_Mat1}, {n_bulk_Mat2}, {slab_pot}, {mat1_pot}, {mat2_pot}, {conv}" # Colelcted Output data
    count=count+1
    int_energy[count]=data
   
    print('**************************')
    os.chdir(original_directory)

print(os.getcwd())
with open("CHA_AEI.txt", "w") as f:
    for row in int_energy:
        f.write(row + "\n")
