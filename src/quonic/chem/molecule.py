"""Molecular geometry representation.

Provides the ``Molecule`` dataclass for holding atomic coordinates and
electronic structure metadata.  Supports loading from XYZ strings/files,
SMILES strings, and SDF files.

Example::

    from quonic.chem import Molecule

    mol = Molecule.from_xyz('''
    2
    H2 at equilibrium
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    ''')
    print(mol.n_electrons)  # 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .._i18n import tr

# Atomic numbers for common elements (Z for electron count)
_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "I": 53,
}


@dataclass(frozen=True)
class Molecule:
    """Molecular geometry and electronic structure metadata.

    Attributes:
        atoms: Element symbols, e.g. ("H", "H", "O").
        coords: Nx3 Cartesian coordinates in Angstroms.
        charge: Net molecular charge.
        spin: Number of unpaired electrons (2S).  0 = singlet.
        basis: Basis set name (default ``"sto-3g"``).
    """

    atoms: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    charge: int = 0
    spin: int = 0
    basis: str = "sto-3g"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_xyz(
        cls,
        xyz: str,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ) -> Molecule:
        """Parse a standard XYZ string.

        Expected format::

            <natoms>
            comment line
            H  0.0  0.0  0.0
            H  0.0  0.0  0.74
        """
        lines = [line.strip() for line in xyz.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(tr("err.chem.xyz_parse", reason="too few lines"))
        try:
            n = int(lines[0])
        except ValueError as exc:
            raise ValueError(
                tr("err.chem.xyz_parse", reason="first line must be atom count")
            ) from exc
        # lines[1] is comment; lines[2:] are atoms
        atom_lines = lines[2 : 2 + n]
        if len(atom_lines) < n:
            raise ValueError(
                tr("err.chem.xyz_parse", reason=f"expected {n} atoms, got {len(atom_lines)}")
            )
        atoms: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for line in atom_lines:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    tr("err.chem.xyz_parse", reason=f"bad atom line: {line!r}")
                )
            atoms.append(parts[0].capitalize())
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
        return cls(
            atoms=tuple(atoms),
            coords=tuple(coords),
            charge=charge,
            spin=spin,
            basis=basis,
        )

    @classmethod
    def from_xyz_file(
        cls,
        path: str | Path,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ) -> Molecule:
        """Load from an XYZ file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_xyz(text, charge=charge, spin=spin, basis=basis)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ) -> Molecule:
        """Convert a SMILES string to 3D geometry via RDKit.

        Requires: ``pip install rdkit`` or ``pip install 'quonic[chem-rdkit]'``
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise ImportError(tr("err.chem.rdkit_missing")) from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(tr("err.chem.smiles_convert", smiles=smiles, reason="invalid SMILES"))
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            raise ValueError(
                tr("err.chem.smiles_convert", smiles=smiles, reason="3D embedding failed")
            )
        AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atoms: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append(atom.GetSymbol().capitalize())
            coords.append((pos.x, pos.y, pos.z))
        return cls(
            atoms=tuple(atoms),
            coords=tuple(coords),
            charge=charge,
            spin=spin,
            basis=basis,
        )

    @classmethod
    def from_sdf(
        cls,
        path: str | Path,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ) -> Molecule:
        """Load from an SDF file via RDKit.

        Requires: ``pip install rdkit`` or ``pip install 'quonic[chem-rdkit]'``
        """
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(tr("err.chem.rdkit_missing")) from exc

        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next(iter(supplier), None)
        if mol is None:
            raise ValueError(tr("err.chem.xyz_parse", reason=f"no molecule in SDF file: {path}"))

        conf = mol.GetConformer()
        atoms: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append(atom.GetSymbol().capitalize())
            coords.append((pos.x, pos.y, pos.z))
        return cls(
            atoms=tuple(atoms),
            coords=tuple(coords),
            charge=charge,
            spin=spin,
            basis=basis,
        )

    @classmethod
    def from_pdb(
        cls,
        path: str | Path,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ) -> Molecule:
        """Load from a PDB file.  Pure Python, no RDKit required."""
        from .formats import from_pdb as _from_pdb

        mol = _from_pdb(path)
        return cls(
            atoms=mol.atoms,
            coords=mol.coords,
            charge=charge,
            spin=spin,
            basis=basis,
        )

    @classmethod
    def from_mol2(
        cls,
        path: str | Path,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ) -> Molecule:
        """Load from a MOL2 file.  Pure Python, no RDKit required."""
        from .formats import from_mol2 as _from_mol2

        mol = _from_mol2(path)
        return cls(
            atoms=mol.atoms,
            coords=mol.coords,
            charge=charge,
            spin=spin,
            basis=basis,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons = sum(atomic numbers) - charge."""
        total_z = sum(_ATOMIC_NUMBERS.get(a, 0) for a in self.atoms)
        return total_z - self.charge

    @property
    def spin_multiplicity(self) -> int:
        """Spin multiplicity = 2S + 1."""
        return self.spin + 1

    # ------------------------------------------------------------------
    # PySCF bridge
    # ------------------------------------------------------------------

    def to_pyscf_mol(self, basis: str | None = None):
        """Convert to a PySCF ``Mole`` object.

        Requires: ``pip install pyscf`` or ``pip install 'quonic[chem]'``
        """
        try:
            from pyscf import gto
        except ImportError as exc:
            raise ImportError(tr("err.chem.pyscf_missing")) from exc

        mol = gto.Mole()
        atom_str = ";".join(
            f"{a} {x} {y} {z}" for a, (x, y, z) in zip(self.atoms, self.coords)
        )
        mol.atom = atom_str
        mol.basis = basis or self.basis
        mol.charge = self.charge
        mol.spin = self.spin
        mol.build()
        return mol

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        formula = self._formula()
        return f"Molecule({formula}, charge={self.charge}, spin={self.spin}, basis={self.basis})"

    def _formula(self) -> str:
        """Hill system formula."""
        from collections import Counter

        counts = Counter(self.atoms)
        parts: list[str] = []
        for elem in ("C", "H"):
            if elem in counts:
                parts.append(elem if counts[elem] == 1 else f"{elem}{counts[elem]}")
                del counts[elem]
        for elem in sorted(counts):
            parts.append(elem if counts[elem] == 1 else f"{elem}{counts[elem]}")
        return "".join(parts)
