"""Molecular format parsers — PDB, MOL2, FCIDUMP.

Pure Python parsers for common quantum chemistry file formats.
No external dependencies required (FCIDUMP → OpenFermion mapping is optional).

Example::

    from quonic.chem.formats import from_pdb, from_fcidump

    mol = from_pdb("protein.pdb")
    result = from_fcidump("FCIDUMP")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .._i18n import tr
from ..result import Result

# ------------------------------------------------------------------
# PDB parser
# ------------------------------------------------------------------

def from_pdb(path: str | Path) -> Any:
    """Parse a PDB (Protein Data Bank) file into a Molecule.

    Only reads ``ATOM`` and ``HETATM`` records.  Alternate locations
    (``ALTLOC``) use the first occurrence.  Multi-model files use the
    first model only.

    Args:
        path: Path to the PDB file.

    Returns:
        A ``quonic.chem.Molecule`` with atoms and coordinates.
    """
    from .molecule import Molecule

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    atoms: list[str] = []
    coords: list[tuple[float, float, float]] = []
    seen_model = False

    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            if seen_model:
                break  # only first model
            seen_model = True
        if record not in ("ATOM", "HETATM"):
            continue
        # PDB format: cols 13-16 atom name, 17 altloc, 31-38 x, 39-46 y, 47-54 z
        element = line[76:78].strip()
        if not element:
            # Fallback: extract from atom name (cols 13-16)
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[:1].upper()
            if len(atom_name) > 1 and atom_name[1].isalpha() and atom_name[1].isupper():
                element = atom_name[:2].capitalize()
            else:
                element = atom_name[0].upper()
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        atoms.append(element.capitalize())
        coords.append((x, y, z))

    if not atoms:
        raise ValueError(tr("err.chem.pdb_parse", path=str(path)))

    return Molecule(atoms=tuple(atoms), coords=tuple(coords))


# ------------------------------------------------------------------
# MOL2 parser
# ------------------------------------------------------------------

def from_mol2(path: str | Path) -> Any:
    """Parse a Tripos MOL2 file into a Molecule.

    Reads the ``@<TRIPOS>ATOM`` section.  Multi-molecule files return
    the first molecule only.

    Args:
        path: Path to the MOL2 file.

    Returns:
        A ``quonic.chem.Molecule`` with atoms and coordinates.
    """
    from .molecule import Molecule

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    atoms: list[str] = []
    coords: list[tuple[float, float, float]] = []

    in_atom_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            continue
        if stripped.startswith("@<TRIPOS>"):
            if in_atom_section:
                break  # next section
            continue
        if not in_atom_section or not stripped:
            continue

        # MOL2 ATOM: atom_id atom_name x y z atom_type [subst_id [subst_name [charge]]]
        parts = stripped.split()
        if len(parts) < 6:
            continue
        # atom_name like "C1", "O2", "N.am" — extract element
        atom_name = parts[1]
        element = re.match(r"([A-Za-z]+)", atom_name)
        if element:
            elem = element.group(1).capitalize()
        else:
            elem = atom_name[0].upper()
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        atoms.append(elem)
        coords.append((x, y, z))

    if not atoms:
        raise ValueError(tr("err.chem.mol2_parse", path=str(path)))

    return Molecule(atoms=tuple(atoms), coords=tuple(coords))


# ------------------------------------------------------------------
# FCIDUMP parser
# ------------------------------------------------------------------

def from_fcidump(
    path: str | Path,
    mapping: str = "jordan_wigner",
) -> Result:
    """Read integrals from a FCIDUMP file and build a qubit Hamiltonian.

    FCIDUMP is the standard Fortran format for one-/two-electron integrals,
    used by PySCF, PSI4, Molpro, ORCA, etc.

    Args:
        path: Path to the FCIDUMP file.
        mapping: ``"jordan_wigner"`` or ``"bravyi_kitaev"``.

    Returns:
        ``Result.from_value(nuclear_repulsion, hamiltonian=terms,
        n_qubits=..., n_orbitals=..., n_electrons=...)``
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    n_orbitals, n_electrons, _ms2 = _parse_fcidump_header(text)
    integrals = _parse_fcidump_integrals(text, n_orbitals)

    h1 = integrals["h1"]
    h2 = integrals["h2"]
    nuclear_repulsion = integrals["enuc"]

    # Build qubit Hamiltonian via OpenFermion
    try:
        import openfermion
    except ImportError as exc:
        raise ImportError(tr("err.chem.openfermion_missing")) from exc

    fermion_op = _build_fermion_operator(h1, h2, nuclear_repulsion)

    if mapping == "jordan_wigner":
        qubit_op = openfermion.jordan_wigner(fermion_op)
    elif mapping == "bravyi_kitaev":
        qubit_op = openfermion.bravyi_kitaev(fermion_op)
    else:
        raise ValueError(tr("err.chem.mapping_unknown", mapping=mapping))

    from ..algorithms.hamiltonians_ext import from_openfermion

    n_qubits = 2 * n_orbitals
    terms = from_openfermion(qubit_op)

    return Result.from_value(
        nuclear_repulsion,
        hamiltonian=terms,
        n_qubits=n_qubits,
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
    )


def _parse_fcidump_header(text: str) -> tuple[int, int, int]:
    """Extract NORB, NELEC, MS2 from the &FCIDUMP namelist."""
    n_orb = 0
    n_elec = 0
    ms2 = 0
    in_namelist = False
    for line in text.splitlines():
        s = line.strip().upper()
        if "&FCIDUMP" in s:
            in_namelist = True
            continue
        if in_namelist and s.startswith("/"):
            break
        if in_namelist:
            m = re.search(r"NORB\s*=\s*(\d+)", s)
            if m:
                n_orb = int(m.group(1))
            m = re.search(r"NELEC\s*=\s*(\d+)", s)
            if m:
                n_elec = int(m.group(1))
            m = re.search(r"MS2\s*=\s*(\d+)", s)
            if m:
                ms2 = int(m.group(1))
    if n_orb == 0:
        raise ValueError(tr("err.chem.fcidump_parse", reason="NORB not found in header"))
    return n_orb, n_elec, ms2


def _parse_fcidump_integrals(
    text: str,
    n_orbitals: int,
) -> dict[str, Any]:
    """Parse integral blocks from FCIDUMP.

    Returns dict with 'h1' (n,n), 'h2' (n,n,n,n), 'enuc' (float).
    """
    h1 = np.zeros((n_orbitals, n_orbitals))
    h2 = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    enuc = 0.0

    # Skip header (everything before the first data line after /)
    past_header = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if "&FCIDUMP" in s.upper():
            continue
        if s.startswith("/"):
            past_header = True
            continue
        if not past_header:
            continue
        if s.startswith(("&", "!")):
            continue

        parts = s.split()
        if len(parts) < 5:
            continue

        try:
            val = float(parts[0])
            i = int(parts[1])
            j = int(parts[2])
            k = int(parts[3])
            l = int(parts[4])
        except (ValueError, IndexError):
            continue

        # Convention: i,j,k,l are 1-based
        # Two-electron: (ij|kl) — all > 0
        # One-electron: h(i,j) — k,l = 0
        # Nuclear repulsion: i,j,k,l = 0
        if i == 0 and j == 0 and k == 0 and l == 0:
            enuc = val
        elif k == 0 and l == 0:
            # One-electron integral
            h1[i - 1, j - 1] = val
            h1[j - 1, i - 1] = val  # symmetric
        else:
            # Two-electron integral (chemist notation (ij|kl))
            h2[i - 1, j - 1, k - 1, l - 1] = val
            # Symmetry: (ij|kl) = (ji|lk) = (kl|ij) = (lk|ji)
            h2[j - 1, i - 1, k - 1, l - 1] = val
            h2[i - 1, j - 1, l - 1, k - 1] = val
            h2[j - 1, i - 1, l - 1, k - 1] = val
            h2[k - 1, l - 1, i - 1, j - 1] = val
            h2[l - 1, k - 1, i - 1, j - 1] = val
            h2[k - 1, l - 1, j - 1, i - 1] = val
            h2[l - 1, k - 1, j - 1, i - 1] = val

    return {"h1": h1, "h2": h2, "enuc": enuc}


def _build_fermion_operator(
    h1: Any,
    h2: Any,
    nuclear_repulsion: float,
) -> Any:
    """Build OpenFermion FermionOperator from integrals."""
    import openfermion

    n_orbitals = h1.shape[0]
    fermion_op = openfermion.FermionOperator((), nuclear_repulsion)

    # One-electron terms
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            coeff = h1[p, q]
            if abs(coeff) < 1e-12:
                continue
            fermion_op += openfermion.FermionOperator(((2*p, 1), (2*q, 0)), coeff)
            fermion_op += openfermion.FermionOperator(((2*p+1, 1), (2*q+1, 0)), coeff)

    # Two-electron terms (chemist notation → physicist notation)
    h2_phys = np.einsum("ijkl->ikjl", h2)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coeff = h2_phys[p, q, r, s]
                    if abs(coeff) < 1e-12:
                        continue
                    # Alpha-Alpha
                    fermion_op += openfermion.FermionOperator(
                        ((2*p, 1), (2*q, 1), (2*r, 0), (2*s, 0)), 0.5 * coeff
                    )
                    # Beta-Beta
                    fermion_op += openfermion.FermionOperator(
                        ((2*p+1, 1), (2*q+1, 1), (2*r+1, 0), (2*s+1, 0)), 0.5 * coeff
                    )
                    # Alpha-Beta
                    fermion_op += openfermion.FermionOperator(
                        ((2*p, 1), (2*q+1, 1), (2*r+1, 0), (2*s, 0)), 0.5 * coeff
                    )
                    # Beta-Alpha
                    fermion_op += openfermion.FermionOperator(
                        ((2*p+1, 1), (2*q, 1), (2*r, 0), (2*s+1, 0)), 0.5 * coeff
                    )

    return fermion_op
