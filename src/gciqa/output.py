"""Output format utilities for GCIQA results.

Converts GCIQA conformations to standard molecular file formats
for visualization, analysis, and downstream tool integration.

Supported formats:
- PDB: Visualization, MD, docking validation
- JSON: Automated analysis, pipelines

Usage::

    from gciqa.output import to_pdb, to_json

    # Export to PDB
    pdb_str = to_pdb(conformation, atoms, filename="output.pdb")

    # Export to JSON with metadata
    json_str = to_json(conformation, atoms, report=report)
"""

from __future__ import annotations

import json as json_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report import ConstraintReport


def to_pdb(
    conformation: dict[str, tuple[float, float, float]],
    atoms: list[str],
    atom_names: list[str] | None = None,
    residue_names: list[str] | None = None,
    residue_numbers: list[int] | None = None,
    chain_ids: list[str] | None = None,
    filename: str | None = None,
) -> str:
    """Convert a conformation to PDB format.

    Args:
        conformation: Super-atom positions {index: (x, y, z)}.
        atoms: Element symbols for each atom.
        atom_names: Atom names (e.g., "CA", "CB"). Defaults to element.
        residue_names: Residue names (e.g., "ALA"). Defaults to "UNK".
        residue_numbers: Residue numbers. Defaults to 1.
        chain_ids: Chain IDs. Defaults to "A".
        filename: If provided, write to this file.

    Returns:
        PDB format string.
    """
    lines = []
    lines.append("HEADER    GCIQA PREDICTION")
    lines.append("TITLE     GCIQA PREDICTED CONFORMATION")

    for i, (key, coord) in enumerate(sorted(conformation.items(), key=lambda x: int(x[0]))):
        idx = int(key)
        element = atoms[idx] if idx < len(atoms) else "C"
        atom_name = atom_names[idx] if atom_names and idx < len(atom_names) else element
        res_name = residue_names[idx] if residue_names and idx < len(residue_names) else "UNK"
        res_num = residue_numbers[idx] if residue_numbers and idx < len(residue_numbers) else 1
        chain = chain_ids[idx] if chain_ids and idx < len(chain_ids) else "A"

        x, y, z = coord
        # PDB ATOM record format
        line = (
            f"ATOM  {i+1:5d} {atom_name:<4s} {res_name:3s} {chain:1s}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}  "
        )
        lines.append(line)

    lines.append("END")

    pdb_str = "\n".join(lines) + "\n"

    if filename:
        with open(filename, "w") as f:
            f.write(pdb_str)

    return pdb_str


def to_json(
    conformation: dict[str, tuple[float, float, float]],
    atoms: list[str] | None = None,
    report: ConstraintReport | None = None,
    rmsd: float | None = None,
    metadata: dict | None = None,
    filename: str | None = None,
) -> str:
    """Convert a conformation to JSON with full metadata.

    Args:
        conformation: Super-atom positions {index: (x, y, z)}.
        atoms: Element symbols (optional).
        report: Constraint satisfaction report (optional).
        rmsd: RMSD vs crystal structure (optional).
        metadata: Additional metadata (optional).
        filename: If provided, write to this file.

    Returns:
        JSON string.
    """
    data = {
        "conformation": {
            key: list(coord) for key, coord in conformation.items()
        },
    }

    if atoms:
        data["atoms"] = atoms

    if report:
        data["constraint_report"] = {
            "overall_score": report.overall_score,
            "satisfied": report.satisfied_count,
            "partial": report.partial_count,
            "violated": report.violated_count,
            "details": [
                {
                    "constraint": str(ev.constraint),
                    "status": ev.status.value,
                    "actual_value": ev.actual_value,
                    "expected_range": list(ev.expected_range),
                    "deviation": ev.deviation,
                }
                for ev in report.constraints
            ],
        }

    if rmsd is not None:
        data["rmsd"] = {"vs_crystal": rmsd}

    if metadata:
        data["metadata"] = metadata

    json_str = json_module.dumps(data, indent=2)

    if filename:
        with open(filename, "w") as f:
            f.write(json_str)

    return json_str
