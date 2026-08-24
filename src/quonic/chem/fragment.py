"""Molecular fragmentation for DMET and FMO methods.

Splits a molecule into fragments based on bond connectivity using
distance-based covalent radii.  No external dependencies required.

Example::

    from quonic.chem import Molecule, fragment_molecule

    mol = Molecule.from_xyz('''4\\nH4\\nH 0 0 0\\nH 0 0 0.74\\nH 0 0 2.0\\nH 0 0 2.74''')
    frags = fragment_molecule(mol, max_fragment_size=2)
    print(len(frags))  # 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .._i18n import tr

# Covalent radii in Angstroms (Alvarez 2008, selected elements)
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22, "Br": 1.20,
    "I": 1.39,
}

# Tolerance factor for bond detection (multiply sum of radii by this)
_BOND_TOLERANCE = 1.3


@dataclass(frozen=True)
class Fragment:
    """A molecular fragment.

    Attributes:
        atom_indices: Indices of atoms in the parent molecule.
        atoms: Element symbols.
        coords: Cartesian coordinates in Angstroms.
        charge: Fragment charge.
        spin: Number of unpaired electrons (2S).
    """

    atom_indices: tuple[int, ...]
    atoms: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    charge: int = 0
    spin: int = 0

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


def fragment_molecule(
    molecule: Any,
    method: str = "auto",
    max_fragment_size: int = 10,
    charge: int = 0,
) -> list[Fragment]:
    """Partition a molecule into fragments.

    Args:
        molecule: A ``quonic.chem.Molecule``.
        method: Fragmentation strategy:

            - ``"auto"`` — connected components by distance (default).
            - ``"by_atom_count"`` — greedy partitioning into chunks of
              *max_fragment_size* atoms.

        max_fragment_size: Maximum atoms per fragment.
        charge: Total molecular charge to distribute.

    Returns:
        List of :class:`Fragment` objects.

    Raises:
        ValueError: If fragmentation produces no fragments.
    """
    if method == "auto":
        return _by_distance(molecule, max_fragment_size, charge)
    elif method == "by_atom_count":
        return _by_atom_count(molecule, max_fragment_size, charge)
    else:
        raise ValueError(f"Unknown fragmentation method: {method!r}")


# ------------------------------------------------------------------
# Distance-based fragmentation
# ------------------------------------------------------------------

def _by_distance(
    molecule: Any,
    max_size: int,
    charge: int,
) -> list[Fragment]:
    atoms = list(molecule.atoms)
    coords = [tuple(c) for c in molecule.coords]
    n = len(atoms)

    # Build adjacency list based on covalent radii
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = _distance(coords[i], coords[j])
            r_i = _COVALENT_RADII.get(atoms[i], 1.5)
            r_j = _COVALENT_RADII.get(atoms[j], 1.5)
            if dist < (r_i + r_j) * _BOND_TOLERANCE:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find connected components
    visited = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        comp: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    # Split large components
    fragments: list[Fragment] = []
    for comp in components:
        if len(comp) <= max_size:
            fragments.append(_make_fragment(molecule, comp))
        else:
            # Greedy split
            for i in range(0, len(comp), max_size):
                chunk = comp[i : i + max_size]
                fragments.append(_make_fragment(molecule, chunk))

    if not fragments:
        raise ValueError(tr("err.chem.fragment_empty"))

    return fragments


def _by_atom_count(
    molecule: Any,
    max_size: int,
    charge: int,
) -> list[Fragment]:
    n = len(molecule.atoms)
    fragments: list[Fragment] = []
    for i in range(0, n, max_size):
        indices = list(range(i, min(i + max_size, n)))
        fragments.append(_make_fragment(molecule, indices))
    if not fragments:
        raise ValueError(tr("err.chem.fragment_empty"))
    return fragments


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_fragment(molecule: Any, indices: list[int]) -> Fragment:
    return Fragment(
        atom_indices=tuple(indices),
        atoms=tuple(molecule.atoms[i] for i in indices),
        coords=tuple(molecule.coords[i] for i in indices),
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
