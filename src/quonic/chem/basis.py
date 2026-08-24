"""Basis set registry and validation.

Provides a lightweight registry of common quantum chemistry basis sets.
No external dependencies required.

Example::

    from quonic.chem import validate_basis, list_bases

    basis = validate_basis("sto-3g")   # "sto-3g"
    all_bases = list_bases()           # sorted list of known basis names
"""

from __future__ import annotations

from .._i18n import tr

COMMON_BASES: dict[str, str] = {
    # Minimal basis sets
    "sto-3g": "sto-3g",
    "sto-6g": "sto-6g",
    # Pople split-valence
    "3-21g": "3-21g",
    "6-31g": "6-31g",
    "6-31g*": "6-31g*",
    "6-31g**": "6-31g**",
    "6-31+g*": "6-31+g*",
    "6-31+g**": "6-31+g**",
    "6-311g": "6-311g",
    "6-311g*": "6-311g*",
    "6-311g**": "6-311g**",
    "6-311+g*": "6-311+g*",
    "6-311+g**": "6-311+g**",
    "6-311++g**": "6-311++g**",
    "6-311++g(2d,2p)": "6-311++g(2d,2p)",
    # Dunning correlation-consistent
    "cc-pvdz": "cc-pvdz",
    "cc-pvtz": "cc-pvtz",
    "cc-pvqz": "cc-pvqz",
    "cc-pv5z": "cc-pv5z",
    "aug-cc-pvdz": "aug-cc-pvdz",
    "aug-cc-pvtz": "aug-cc-pvtz",
    "aug-cc-pvqz": "aug-cc-pvqz",
    # Def2 family (Ahlrichs)
    "def2-svp": "def2-svp",
    "def2-svpd": "def2-svpd",
    "def2-tzvp": "def2-tzvp",
    "def2-tzvpd": "def2-tzvpd",
    "def2-tzvpp": "def2-tzvpp",
    "def2-qzvp": "def2-qzvp",
    "def2-qzvpp": "def2-qzvpp",
    # LANL2DZ (ECP)
    "lanl2dz": "lanl2dz",
}


def validate_basis(basis: str) -> str:
    """Validate and normalize a basis set name.

    Args:
        basis: Basis set name (case-insensitive).

    Returns:
        Canonical lowercase basis name.

    Raises:
        ValueError: If the basis set is not in the known registry.
    """
    key = basis.strip().lower()
    if key not in COMMON_BASES:
        raise ValueError(tr("err.chem.unknown_basis", basis=basis))
    return COMMON_BASES[key]


def list_bases() -> list[str]:
    """Return sorted list of known basis set names."""
    return sorted(COMMON_BASES.keys())
