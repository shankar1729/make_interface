from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import make_supercell, rotate
from ase.calculators.calculator import Calculator


class Surfaces:
    """List of possible surface supercells, constrained by surface size."""
    bulk: Atoms  #: Reference bulk structure
    sup: np.ndarray  #: N x 3 x 3 linear combinations defining each supercell
    a: np.ndarray  #: N lengths (Angstroms) of shorter in-plane lattice vector
    b: np.ndarray  #: N lengths (Angstroms) of longer in-plane lattice vector
    theta: np.ndarray  #: N angles (radians) between in-plane lattice vectors
    area: np.ndarray  #: N areas of the surface supercells

    def __init__(
        self,
        bulk: Atoms,
        surface_index: np.ndarray,
        Lmax: float,
        theta_max=np.radians(135),
        *,
        TOL = 1E-5,
        vector1: np.ndarray | None = None,
    ) -> None:
        """
        Find surface unit cells of `bulk` with miller index `surface_index`,
        such that both in-plane lattice vectors are shorter than `Lmax` 
        and have a reasonable angle between 90 degrees and `theta_max`.
        
        If `surface_index` is (0, 0, 0), this results in all surface supercells
        with in-plane lengths less than `Lmax`, regardless of surface direction.
        
        If `vector1` is specified, constrain first vector of the surface unit cell
        to it. This direction must be orthogonal to `surface_index` if also given.
        """ 

        # Get lattice vectors and compute their lengths:
        vectors = lattice_vectors(bulk, surface_index, Lmax)
        RT = bulk.cell[:]
        vectors_cart = vectors @ RT
        vector_lengths = np.linalg.norm(vectors_cart, axis=1)
        
        # Compute angle between all pairs of vectors:
        theta = np.arccos(
            np.clip(
                (vectors_cart @ vectors_cart.T)
                / np.outer(vector_lengths, vector_lengths),
                -1,
                +1
            )
        )
        
        # Select pairs with v1 shorter than v2 and angle in range:
        theta_min = 0.5 * np.pi - TOL  # 90 degrees (with round-off margin)
        v1_shorter = (vector_lengths[:, None] <= vector_lengths[None, :] + TOL)
        if vector1 is not None:
            # Override by forcing first vector to be the required value:
            v1_shorter = np.linalg.norm((vectors - vector1) @ RT, axis=1)[:, None] < TOL
        sel1, sel2 = np.where(v1_shorter & (theta >= theta_min) & (theta <= theta_max))
        if not len(sel1):
            raise KeyError(
                f"No surface unit cells with lattice vector lengths <= {Lmax}"
                " and a reasonable angle"
            )
        
        # Compute and reduce normals:
        normals = np.cross(vectors[sel1], vectors[sel2])
        normals_gcd = np.gcd(normals[..., 0], np.gcd(normals[..., 1], normals[..., 2]))
        normals //= normals_gcd[..., None]
        
        # Collect and sort selected pairs:
        sup = np.stack((vectors[sel1], vectors[sel2], normals), axis=2)
        a = vector_lengths[sel1]
        b = vector_lengths[sel2]
        theta = theta[sel1, sel2]
        area = a * b * np.sin(theta)

        # Determine keys for sorting output surfaces (most to least important):
        sort_properties = [area, theta, a, b]
        sort_properties.append(np.abs(normals).sum(axis=-1))  # simplest of equiv. specs
        if np.linalg.norm(surface_index):
            sort_properties.append(-normals @ surface_index)  # fix sign when possible
        sort_properties.extend(-normals.T)  # sort descending by normal components
        sort_properties.extend(-vectors[sel1].T)  # sort descending by v1 components
        sort_properties.extend(-vectors[sel2].T)  # sort descending by v2 components
        sort_properties = np.stack(sort_properties, axis=1)

        class SortKey:
            """Indexed sorter of unit cells by area, theta, a, b."""
            def __init__(self, i):
                self.index = i  # index into relevant arrays

            def compare(self, other) -> int:
                key_diff = sort_properties[self.index] - sort_properties[other.index]
                for d in key_diff:
                    if np.abs(d) > TOL:
                        return int(np.copysign(1, d))
                return 0
            
            def equivalent(self, other) -> bool:
                """Only check area and angle for equivalence."""
                area_equal = (np.abs(area[self.index] - area[other.index]) <= TOL)
                theta_equal = (np.abs(theta[self.index] - theta[other.index]) <= TOL)
                return area_equal and theta_equal

            def __lt__(self, other): return self.compare(other) < 0
            def __gt__(self, other): return self.compare(other) > 0
            def __eq__(self, other): return self.compare(other) == 0
            def __le__(self, other): return self.compare(other) <= 0
            def __ge__(self, other): return self.compare(other) >= 0
            def __ne__(self, other): return self.compare(other) != 0

        sorted_keys = sorted([SortKey(i) for i in range(len(sel1))])
        sel = [sorted_keys[0].index]
        for prev_key, key in zip(sorted_keys, sorted_keys[1:]):
            if not prev_key.equivalent(key):  # pick first of equivalent cells
                sel.append(key.index)
        
        # Set properties:
        self.bulk = bulk
        self.sup = sup[sel]
        self.a = a[sel]
        self.b = b[sel]
        self.theta = theta[sel]
        self.area = area[sel]

    def __str__(self) -> str:
        """Neatly summarize in tabular form."""
        lines = [
            f"\nIndex | {'Area':>6s} {'a':>6s} {'b':>6s} {'theta':>6s}"
            " | surface1 | surface2 |  normal",
            '-'*6 + '+' + '-'*29 + '+' + '-'*10 + '+' + '-'*10 + '+' + '-'*10
        ]
        theta_deg = np.degrees(self.theta)
        for i, (area, a, b, theta, (a0, b0, n0, a1, b1, n1, a2, b2, n2)) in enumerate(
            zip(self.area, self.a, self.b, theta_deg, self.sup.reshape(-1, 9))
        ):
            lines.append(
                f"{i:5d} | {area:6.2f} {a:6.3f} {b:6.3f} {theta:6.2f}"
                f" | {a0:2d} {a1:2d} {a2:2d} | {b0:2d} {b1:2d} {b2:2d}"
                f" | {n0:2d} {n1:2d} {n2:2d}"
            )
        return "\n".join(lines)

    def make_slab(
        self,
        i_surface: int,
        minimum_thickness: float,
        vacuum_spacing: float,
        calculator: Calculator,
        *,
        TOL = 1E-6,
    ) -> Atoms:
        """Make slab corresponding to index `i_surface` from search results,
        with specified `minimum_thickness` and `vacuum_spacing` in Angstroms,
        using the specified `calculator` to find the lowest-energy cut.
        TOL specifies the tolerance in detecting equivalent layers."""

        # Construct smallest supercell units needed to reach thickness
        print(f"\nConstructing slab for surface index {i_surface}")
        sup_unit = self.sup[i_surface]
        RTsup_unit = sup_unit.T @ self.bulk.cell[:]
        volume_unit = abs(np.linalg.det(RTsup_unit))
        base_area = np.linalg.norm(np.cross(RTsup_unit[0], RTsup_unit[1]))
        thickness_unit = volume_unit / base_area
        n_units = int(np.ceil(minimum_thickness / thickness_unit))
        sup = sup_unit @ np.diag([1, 1, n_units])
        sup_thickness = n_units * thickness_unit
        supercell = make_supercell(self.bulk, sup.T, wrap=True, order="atom-major")
        print(
            f"  Constructed surface supercell with {n_units} periodic units"
            f" and thickness {sup_thickness:.3f} A"
        )

        # Find minimum number of equivalent sub-layers:
        n_layers = count_layers(supercell, TOL=TOL)
        layer_thickness = sup_thickness / n_layers
        n_layers_needed = int(np.ceil(minimum_thickness / layer_thickness))
        print(
            f"  Identified {n_layers} equivalent layers"
            f" of thickness {layer_thickness:.3f} A each\n"
            f"  Selecting {n_layers_needed} equivalent layers"
            f" to get slab thickness {n_layers_needed * layer_thickness:.3f} A"
        )

        # Identify potential cut planes within one equivalent layer:
        z = supercell.get_scaled_positions(wrap=True)[:, 2]  # fractional z in [0, 1)
        z_layer_sorted = np.sort(z[z < 1/n_layers])
        z_layer_padded = np.concatenate(
            (z_layer_sorted, [z_layer_sorted[0] + 1/n_layers])
        )  # add periodically wrapped point to end (for first layer in z)
        z_gap = z_layer_padded[1:] - z_layer_sorted  # lengths of gaps
        gap_sel = np.where(z_gap > 1E-3)[0]  # reject tiny gaps from nearly planar atoms
        z_gap = z_gap[gap_sel]
        z_mid = 0.5 * (z_layer_padded[gap_sel] + z_layer_padded[gap_sel + 1])

        # Evaluate all possible equivalent cuts:
        print(f"  Evaluating {len(z_mid)} possible cuts within equivalent layer")
        best_slab = None
        best_energy = np.inf
        for z_start, gap_thickness in zip(z_mid, z_gap * sup_thickness):
            z_diff = z - z_start
            z_diff -= np.floor(z_diff)  # (0,1) wrapped difference from cut plane
            atom_sel = np.where(z_diff < n_layers_needed/n_layers)[0]
            slab = supercell[atom_sel]
            make_vacuum_orthogonal(slab)
            wrap_cut_plane(slab, z_start)
            slab.center(
                vacuum=(vacuum_spacing + gap_thickness)/2, axis=2, about=(0, 0, 0)
            )
            slab.calc = calculator
            energy = slab.get_potential_energy()
            if energy < best_energy:
                best_energy = energy
                best_slab = slab

        # Report surface energy
        self.bulk.calc = calculator
        bulk_energy = self.bulk.get_potential_energy()
        n_bulk_units = len(best_slab) / len(self.bulk)
        surface_energy = (best_energy - n_bulk_units * bulk_energy) / (2 * base_area)
        print(f"  Selected cut with surface energy {surface_energy:.3} eV/A^2")

        # Rotate best slab into standard orientation
        rotate(best_slab, best_slab.cell[0], (1, 0, 0), best_slab.cell[1], (0, 1, 0))
        assert best_slab.cell[2, 2] > 0
        return best_slab


def lattice_vectors(bulk: Atoms, surface_index: np.ndarray, Lmax: float) -> np.ndarray:
    """
    Find surface lattice vectors for `bulk` with Miller indices `surface_index`.
    Return (N x 3) linear combination coefficients for each lattice vector
    up to maximum length `Lmax` (in Angstroms).
    
    If `surface_index` is (0, 0, 0), return all lattice vectors with length <= `Lmax`.
    """
    
    # Find n_max for each dimension to cover all lattice vectors with length upto Lmax
    RT = bulk.cell[:]
    RTinv = np.linalg.inv(RT)
    n_max = np.ceil(Lmax * np.linalg.norm(RTinv, axis=0)).astype(int)
    
    # List all linear combinations with coefficients up to n_max
    grids1d =  [np.arange(-n, n+1) for n in n_max]
    vectors = np.stack(np.meshgrid(*grids1d, indexing="ij"), axis=-1).reshape(-1, 3)
    vector_lengths = np.linalg.norm(vectors @ RT, axis=1)
    
    # Select non-zero vectors hsorter than Lmax and perpendicular to surface_index
    sel = np.where(
        (vector_lengths > 0)
        & (vector_lengths <= Lmax)
        & (vectors @ surface_index == 0)
    )[0]
    if not len(sel):
        raise KeyError(f"No surface vectors with lengths <= {Lmax}")
    return vectors[sel]


def count_layers(supercell: Atoms, *, TOL) -> int:
    """Find the number of equivalent layers along third lattice direction of
    a periodic supercell from the scattering structure factor. This identifies
    symmetry-equivalent layers twisted relative to each other when applicable."""
    Z = supercell.get_atomic_numbers()
    uniqueZ = set(Z)
    z = supercell.get_scaled_positions(wrap=True)[:, 2]  # fractional z coord in [0, 1)
    MAX_LAYERS = 1000
    SF_SQ_CUT = TOL * (len(Z)  ** 2)  # tolerance for testing structure factor
    # Find first finite reciprocal lattice vector with non-zero structure factor:
    for n_layers in range(1, MAX_LAYERS):
        Sf_sq = 0.0
        for Zi in uniqueZ:
            sel = np.where(Z == Zi)[0]  # atoms of this type
            Sf_sq += np.abs(np.sum(np.exp(2j * np.pi * z[sel] * n_layers))) ** 2
        if Sf_sq > SF_SQ_CUT:
            return n_layers
    raise KeyError("Could not identify number of sublayers in supercell")


def make_vacuum_orthogonal(slab: Atoms) -> None:
    """Make the third vacuum direction of slab orthogonal to others."""
    RT = slab.cell[:]
    n_vec = np.cross(RT[0], RT[1])  # normal direction
    n_hat = n_vec / np.linalg.norm(n_vec)
    RT[2] = n_hat * (n_hat @ RT[2])  # project third direction to normal
    slab.set_cell(RT)


def wrap_cut_plane(slab: Atoms, z_cut: float) -> None:
    """Wrap positions relative to fractional cut plane in third direction."""
    pos = slab.get_scaled_positions()
    pos[:, 2] -= z_cut  # reference to cut plane
    pos -= np.floor(pos)  # wrap to [0, 1) in all directions
    slab.set_scaled_positions(pos)
