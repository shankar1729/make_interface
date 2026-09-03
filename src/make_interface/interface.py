from __future__ import annotations
from typing import Union

import numpy as np
from scipy.optimize import minimize
from ase import Atoms

from . import Surfaces


class Interfaces:
    """List of possible interfaces between two surfaces, constrained by strain."""
    surfaces1: Surfaces  #: List of first surfaces to choose from
    surfaces2: Surfaces  #: List of second surfaces to choose from
    index1: np.ndarray  #: Selected indices from `surfaces1` (N integers)
    index2: np.ndarray  #: Selected indices from `surfaces2` (N integers)
    strain: np.ndarray  #: In-plane strain tensor for selected pairs (Nx2x2)
    strain_max: np.ndarray  #: N maximum magnitudes of strain components
    a: np.ndarray  #: N lengths (Angstroms) of shorter in-plane lattice vector
    b: np.ndarray  #: N lengths (Angstroms) of longer in-plane lattice vector
    theta: np.ndarray  #: N angles (radians) between in-plane lattice vectors
    area: np.ndarray  #: N areas of the interface supercells

    def __init__(
        self, surfaces1: Surfaces, surfaces2: Surfaces, strain_max: float
    ) -> None:
        """Find interfaces between pairs from `surfaces1` and `surfaces2`
        with all strain components less than `strain_max` (dimensionless fraction,
        not a percentage)."""
        self.surfaces1 = surfaces1
        self.surfaces2 = surfaces2

        # Compute strains and halfway cells for matching each pair of A with B:
        strain, a, b, theta = compute_strain(
            surfaces1.a[:, None],
            surfaces1.b[:, None],
            surfaces1.theta[:, None],
            surfaces2.a[None, :],
            surfaces2.b[None, :],
            surfaces2.theta[None, :],
        )
        area = a * b * np.sin(theta)  # basal area of averaged cell

        # Select strains below threshold:
        sel1, sel2 = np.where(np.max(np.abs(strain), axis=(-1, -2)) <= strain_max)
        sort_index = area[sel1, sel2].argsort()
        self.index1 = sel1[sort_index]
        self.index2 = sel2[sort_index]
        sel_sorted = self.index1, self.index2
        self.strain = strain[sel_sorted]
        self.strain_max = np.max(np.abs(self.strain), axis=(-1, -2))
        self.a = a[sel_sorted]
        self.b = b[sel_sorted]
        self.theta = theta[sel_sorted]
        self.area = area[sel_sorted]

    def __str__(self) -> str:
        """Neatly summarize in tabular form."""
        lines = [
            f"\nIndex | {'Area':>6s} {'a':>6s} {'b':>6s} {'theta':>6s}"
            f" | {'max|e|':>7s} {'e_xx':>7s} {'e_yy':>7s} {'e_xy':>7s} | Index1 Index2",
            '-'*6 + '+' + '-'*29 + '+' + '-'*29 + '+' + '-'*13
        ]
        theta_deg = np.degrees(self.theta)
        for i, (area, a, b, theta, index1, index2) in enumerate(
            zip(self.area, self.a, self.b, theta_deg, self.index1, self.index2)
        ):
            lines.append(
                f"{i:5d} | {area:6.2f} {a:6.3f} {b:6.3f} {theta:6.2f}"
                f" | {self.strain_max[i]:7.4f} {self.strain[i, 0, 0]:7.4f}"
                f" {self.strain[i, 1, 1]:7.4f} {self.strain[i, 0, 1]:7.4f}"
                f" | {index1:6d} {index2:6d}"
            )
        return "\n".join(lines)

    def make_slab(
        self,
        i_interface: int,
        minimum_thickness1: float,
        minimum_thickness2: float,
        calculator: Calculator,
        n_initial_offsets: int = 10,
        *,
        TOL = 1E-6,
        reflect2: bool = False,
    ) -> tuple[Atoms, Atoms, Atoms]:
        """Make periodically-repeated slab (superlattice) corresponding to index
        `i_interface` from search results, with specified `minimum_thickness1`
        and `minimum_thickness2` for the two materials in Angstroms, using
        the specified `calculator` to find the lowest-energy stackings.
        TOL specifies the tolerance in detecting equivalent layers.
        Use reflect2 to reflect the second slab along its normal, 
        which could be used for constructing twin boundaries.
        Also return the individual slabs stacked to form the interface."""
        index1 = int(self.index1[i_interface])
        index2 = int(self.index2[i_interface])

        # Make slabs of each material
        VACUUM_THICKNESS = 10.0  # vacuum thickness used for slabs (removed later)
        slab1 = self.surfaces1.make_slab(
            index1, minimum_thickness1, VACUUM_THICKNESS, calculator, TOL=TOL
        )
        slab2 = self.surfaces2.make_slab(
            index2, minimum_thickness2, VACUUM_THICKNESS, calculator, TOL=TOL
        )
        if reflect2:
            pos2 = slab2.get_positions()
            pos2[:, 2] *= -1
            slab2.set_positions(pos2)

        # Strain slabs to common base
        a = self.a[i_interface]
        b = self.b[i_interface]
        theta = self.theta[i_interface]
        area = self.area[i_interface]
        RT = np.array([[a, 0, 0], [b * np.cos(theta), b * np.sin(theta), 0], [0, 0, 0]])
        RT[2, 2] = slab1.cell[2, 2]
        slab1.set_cell(RT, scale_atoms=True)
        RT[2, 2] = slab2.cell[2, 2]
        slab2.set_cell(RT, scale_atoms=True)

        # Make slabs compatible for stacking together
        c1 = slab1.cell[2, 2] - VACUUM_THICKNESS
        c2 = slab2.cell[2, 2] - VACUUM_THICKNESS
        RT[2, 2] = c1 + c2  # final combined thickness (no vacuum)
        slab1.translate((0, 0, c1 / 2))
        slab1.wrap()
        slab1.set_cell(RT, scale_atoms=False)
        slab2.translate((0, 0, c2 / 2))
        slab2.wrap()
        slab2.set_cell(RT, scale_atoms=False)
        slab2.translate((0, 0, c1))  # place slab2 above slab1

        # Compute reference energy
        # Note that binding energy reported below is relative to strained slabs
        # This avoids including the thickness-dependent strain energy.
        slab1.calc = calculator
        slab2.calc = calculator
        energy_ref = slab1.get_potential_energy() + slab2.get_potential_energy()

        # Optimize relative offsets between slabs
        def get_interface(offsets: np.ndarray) -> Atoms:
            dx21 = offsets[:2]  # fractional offset between slab2 and slab1
            dx_cell = dx21 + offsets[2:]  # fractional offset of next slab1
            dx_cell -= np.floor(0.5 + dx_cell)  # wrap to [-0.5, 0.5)
            dr21 = dx21 @ RT[:2, :2]  # Cartesian offset of slab2 from slab1
            dr_cell = dx_cell @ RT[:2, :2]  # Cartesian offset to shear cell
            # Shift slab2
            slab2_shifted = slab2.copy()
            slab2_shifted.translate((*dr21, 0.0))
            # Shear interface
            RT[2, :2] = dr_cell
            interface = slab1 + slab2_shifted
            interface.set_cell(RT, scale_atoms=False)
            return interface

        def get_energy(offsets: np.ndarray) -> float:
            interface = get_interface(offsets)
            interface.calc = calculator
            surf_energy = (interface.get_potential_energy() - energy_ref) / (2 * area)
            return surf_energy

        print("\nOptimizing stacking of slabs:")
        best_energy = np.inf
        best_offsets = None
        for i_initial_offset in range(n_initial_offsets):
            res = minimize(get_energy, np.random.rand(4), method='BFGS', tol=1E-4)
            energy = res.fun
            print(f"  Offset: {i_initial_offset}  surface binding: {energy:.3} eV/A^2")
            if energy < best_energy:
                best_energy = energy
                best_offsets = res.x

        print(f"  Selected best stacking with surface binding {best_energy:.3} eV/A^2")
        return get_interface(best_offsets), slab1, slab2


def compute_strain(
    a1: np.ndarray,
    b1: np.ndarray,
    theta1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    theta2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute strain between 2D lattices specified by lengths and angles,
    (a1, b1, theta1) and (a2, b2, theta2)
    Return the strain tensors and the averaged (a, b, theta).
     Broadcastable over all inputs in order to compute pairwise over pairs
     of surfaces, or compute for a single pair of surfaces."""
    cos_theta1, sin_theta1 = np.cos(theta1), np.sin(theta1)
    cos_theta2, sin_theta2 = np.cos(theta2), np.sin(theta2)

    # Compute rotation angle needed to get symmetric strain tensor:
    phi = 0.5 * np.arctan(
        (a1 * b2 * cos_theta2 - a2 * b1 * cos_theta1) /
        (a2 * b1 * sin_theta1 + a1 * b2 * sin_theta2)
    )
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    # Compute exp(symmetric strain tensor):
    def matrix_array22(m11, m12, m21, m22):
        """Stack four arrays into a 2x2 atrix array"""
        return np.stack(
            (np.stack((m11, m12), axis=-1), np.stack((m21, m22), axis=-1)), axis=-2
        )

    R1 = matrix_array22(a1, b1 * cos_theta1, np.zeros_like(a1), b1 * sin_theta1)
    R2 = matrix_array22(a2, b2 * cos_theta2, np.zeros_like(a2), b2 * sin_theta2)
    U = matrix_array22(cos_phi, -sin_phi, sin_phi, cos_phi)
    UT = U.swapaxes(-1, -2)
    exp_strain = (U @ R2) @ np.linalg.inv(UT @ R1)
    strain = matrix_func(np.log, exp_strain)

    # Compute mid-way lattice vectors:
    R = np.einsum('...ba, ...bc -> ...ac', matrix_func(np.exp, 0.5 * strain), UT @ R1)
    lengths = np.linalg.norm(R, axis=-2)
    a = lengths[..., 0]
    b = lengths[..., 1]
    theta = np.arccos(np.sum(np.prod(R, axis=-1), axis=-1) / (a * b))
    return strain, a, b, theta


def matrix_func(func, M):
    """Compute matrix function of symmetric matrix."""
    E, V = np.linalg.eigh(M)
    return np.einsum('...ab, ...b, ...cb -> ...ac', V, func(E), V.conj())
