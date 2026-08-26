"""PDB file parsing for GCIQA protein structure analysis.

Provides parsing of PDB (Protein Data Bank) format files to extract
atomic coordinates, residue information, chain assignments, and metal ions.

This module is a pure Python implementation — no external PDB parser required.
It handles standard PDB v3.3 format files.

Usage:
    from quonic.gciqa.pdb import parse_pdb, parse_pdb_string

    # From file
    protein = parse_pdb("1abc.pdb")

    # From string
    protein = parse_pdb_string(pdb_text)

    # Access structure
    print(f"Atoms: {len(protein.atoms)}")
    print(f"Residues: {len(protein.residues)}")
    print(f"Metal ions: {len(protein.metal_ions)}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Common metal elements in proteins
METAL_ELEMENTS = {
    "ZN", "FE", "CU", "MG", "CA", "MN", "CO", "NI", "MO", "W",
    "V", "CR", "CD", "HG", "PB", "PT", "AG", "AU", "SR", "BA",
    "LI", "NA", "K", "RB", "CS", "BE", "AL", "TI", "ZR",
}


@dataclass
class ResidueInfo:
    """Information about a single residue."""
    name: str
    number: int
    chain: str
    insertion_code: str = ""
    atom_indices: list[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique identifier for this residue."""
        return f"{self.chain}:{self.name}:{self.number}{self.insertion_code}"


@dataclass
class MetalIon:
    """Information about a detected metal ion."""
    element: str
    coord: tuple[float, float, float]
    index: int
    residue_name: str = ""
    residue_number: int = 0
    chain: str = ""
    occupancy: float = 1.0
    b_factor: float = 0.0


@dataclass
class ProteinStructure:
    """Parsed protein structure from a PDB file."""
    atoms: list[str] = field(default_factory=list)
    coords: list[tuple[float, float, float]] = field(default_factory=list)
    residues: list[ResidueInfo] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    metal_ions: list[MetalIon] = field(default_factory=list)

    # Additional metadata
    atom_names: list[str] = field(default_factory=list)
    residue_names: list[str] = field(default_factory=list)
    residue_numbers: list[int] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)
    hetero_atoms: list[bool] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def n_chains(self) -> int:
        return len(set(self.chain_ids))

    def get_atoms_in_residue(self, residue_index: int) -> list[int]:
        """Get atom indices belonging to a residue."""
        if 0 <= residue_index < len(self.residues):
            return self.residues[residue_index].atom_indices
        return []

    def get_residue_for_atom(self, atom_index: int) -> ResidueInfo | None:
        """Get the residue containing a given atom."""
        for res in self.residues:
            if atom_index in res.atom_indices:
                return res
        return None

    def get_chain_atoms(self, chain_id: str) -> list[int]:
        """Get atom indices belonging to a chain."""
        return [i for i, c in enumerate(self.chain_ids) if c == chain_id]

    def get_metal_coordination_atoms(self, metal_index: int, max_dist: float = 2.5) -> list[int]:
        """Get atom indices within coordination distance of a metal ion."""
        import math
        if metal_index >= len(self.coords):
            return []
        mx, my, mz = self.coords[metal_index]
        result = []
        for i, (x, y, z) in enumerate(self.coords):
            if i == metal_index:
                continue
            dist = math.sqrt((x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2)
            if dist <= max_dist:
                result.append(i)
        return result


def _parse_element(atom_name: str, element_from_pdb: str = "", is_hetatm: bool = False) -> str:
    """Extract element symbol from atom name or PDB element column.

    In PDB format, the element symbol is in columns 77-78. When available,
    this is authoritative. Otherwise, we infer from the atom name.

    The ambiguity: "CA" in ATOM records = C-alpha (carbon),
    "CA" in HETATM records = Calcium. The PDB element column resolves this.

    Args:
        atom_name: 4-character atom name from PDB
        element_from_pdb: Element from columns 77-78 (if available)
        is_hetatm: Whether this is a HETATM record
    """
    # If PDB element column is available, use it (authoritative)
    if element_from_pdb:
        return element_from_pdb.strip().upper()

    atom_name = atom_name.strip()
    if not atom_name:
        return "C"

    first = atom_name[0].upper()

    # Two-letter elements that are unambiguous in atom names
    unambiguous_two = {"CL", "BR", "FE", "ZN", "CU", "MG", "MN", "CO", "NI",
                       "MO", "SE", "SI", "AL", "PB", "HG", "CD", "PT", "AG",
                       "AU", "SR", "BA", "TI", "CR", "V", "W", "LI", "RB",
                       "CS", "BE", "ZR"}

    if len(atom_name) >= 2:
        two = atom_name[:2].upper()
        if two in unambiguous_two:
            return two

    # Ambiguous: "CA" = C-alpha in ATOM, Calcium in HETATM
    # "NA" = N-alpha in ATOM, Sodium in HETATM
    # "K"  = Lysine in ATOM, Potassium in HETATM
    if atom_name.upper() == "CA":
        return "CA" if is_hetatm else "C"
    if atom_name.upper() == "NA":
        return "NA" if is_hetatm else "N"

    # Single-letter elements
    single_letter = {"C", "N", "O", "S", "P", "H", "F", "I", "B"}
    if first in single_letter:
        return first

    return first


def _is_metal(element: str) -> bool:
    """Check if an element is a metal."""
    return element.upper() in METAL_ELEMENTS


def _parse_hetatm_residue(residue_name: str) -> bool:
    """Check if a residue name indicates a heteroatom (non-standard residue)."""
    standard_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
        # Nucleic acids
        "A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU",
        # Common modifications
        "MSE", "SEP", "TPO", "PTR", "NEP", "MLY", "HYP",
    }
    return residue_name.strip().upper() not in standard_residues


def parse_pdb_string(pdb_string: str) -> ProteinStructure:
    """Parse a PDB format string into a ProteinStructure.

    Handles:
    - ATOM records (protein/nucleic acid atoms)
    - HETATM records (heteroatoms, ligands, metal ions)
    - MODEL/ENDMDL (uses first model only)

    Does NOT handle:
    - REMARK, CONECT, SEQRES, etc. (ignored)
    - Multi-model ensembles (uses first model)
    - Anisotropic temperature factors

    Args:
        pdb_string: PDB format text

    Returns:
        ProteinStructure with parsed data
    """
    protein = ProteinStructure()

    # Track residues by (chain, name, number, insertion_code)
    residue_map: dict[tuple[str, str, int, str], ResidueInfo] = {}

    for line in pdb_string.split("\n"):
        record_type = line[0:6].strip() if len(line) >= 6 else ""

        # Only process ATOM and HETATM records
        if record_type not in ("ATOM", "HETATM"):
            # Stop at ENDMDL (use first model only)
            if record_type == "ENDMDL":
                break
            continue

        is_hetatm = record_type == "HETATM"

        # Parse PDB fixed-width format (v3.3)
        # Columns: 1-6 record, 7-11 serial, 13-16 name, 17 altloc,
        #          18-20 resname, 22 chain, 23-26 resseq, 27 icode,
        #          31-38 x, 39-46 y, 47-54 z, 55-60 occupancy, 61-66 bfactor
        try:
            serial = int(line[6:11].strip()) if len(line) > 11 else 0
            atom_name = line[12:16].strip() if len(line) > 16 else ""
            alt_loc = line[16].strip() if len(line) > 16 else ""
            residue_name = line[17:20].strip() if len(line) > 20 else ""
            chain_id = line[21].strip() if len(line) > 21 else "A"
            res_seq = int(line[22:26].strip()) if len(line) > 26 else 0
            icode = line[26].strip() if len(line) > 26 else ""
            x = float(line[30:38].strip()) if len(line) > 38 else 0.0
            y = float(line[38:46].strip()) if len(line) > 46 else 0.0
            z = float(line[46:54].strip()) if len(line) > 54 else 0.0
            occupancy = float(line[54:60].strip()) if len(line) > 60 else 1.0
            b_factor = float(line[60:66].strip()) if len(line) > 66 else 0.0
            # Element symbol is in columns 77-78 (authoritative when present)
            element_from_pdb = line[76:78].strip() if len(line) > 76 else ""
        except (ValueError, IndexError):
            # Skip malformed lines
            continue

        # Skip alternate conformations (keep only A or first)
        if alt_loc and alt_loc != "A":
            continue

        # Determine element (use PDB column when available, else infer)
        element = _parse_element(atom_name, element_from_pdb, is_hetatm)

        # Add atom
        atom_index = len(protein.atoms)
        protein.atoms.append(element)
        protein.coords.append((x, y, z))
        protein.atom_names.append(atom_name)
        protein.residue_names.append(residue_name)
        protein.residue_numbers.append(res_seq)
        protein.chain_ids.append(chain_id)
        protein.hetero_atoms.append(is_hetatm)

        # Track chains
        if chain_id not in protein.chains:
            protein.chains.append(chain_id)

        # Track residues
        res_key = (chain_id, residue_name, res_seq, icode)
        if res_key not in residue_map:
            residue_map[res_key] = ResidueInfo(
                name=residue_name,
                number=res_seq,
                chain=chain_id,
                insertion_code=icode,
            )
        residue_map[res_key].atom_indices.append(atom_index)

        # Detect metal ions
        if _is_metal(element):
            protein.metal_ions.append(MetalIon(
                element=element.upper(),
                coord=(x, y, z),
                index=atom_index,
                residue_name=residue_name,
                residue_number=res_seq,
                chain=chain_id,
                occupancy=occupancy,
                b_factor=b_factor,
            ))

    # Convert residue map to list (sorted by chain, then residue number)
    protein.residues = sorted(
        residue_map.values(),
        key=lambda r: (r.chain, r.number, r.insertion_code),
    )

    return protein


def parse_pdb(pdb_path: str | Path) -> ProteinStructure:
    """Parse a PDB file into a ProteinStructure.

    Args:
        pdb_path: Path to PDB file

    Returns:
        ProteinStructure with parsed data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be parsed
    """
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_pdb_string(content)


def find_metal_ions(protein: ProteinStructure, element: str | None = None) -> list[MetalIon]:
    """Find metal ions in a protein structure.

    Args:
        protein: Parsed protein structure
        element: Filter by element (e.g., "ZN", "FE"). None for all metals.

    Returns:
        List of MetalIon objects matching the filter
    """
    if element is None:
        return protein.metal_ions
    element = element.upper()
    return [m for m in protein.metal_ions if m.element == element]


def get_residue_atoms(protein: ProteinStructure, residue_name: str) -> list[int]:
    """Get atom indices for all residues with a given name.

    Args:
        protein: Parsed protein structure
        residue_name: Residue name (e.g., "HIS", "CYS")

    Returns:
        List of atom indices
    """
    indices = []
    for res in protein.residues:
        if res.name.upper() == residue_name.upper():
            indices.extend(res.atom_indices)
    return indices


def get_nearby_residues(
    protein: ProteinStructure,
    center: tuple[float, float, float],
    radius: float,
) -> list[ResidueInfo]:
    """Get residues with any atom within radius of center.

    Args:
        protein: Parsed protein structure
        center: (x, y, z) center point
        radius: Distance cutoff in Angstroms

    Returns:
        List of ResidueInfo objects with at least one atom in range
    """
    import math
    cx, cy, cz = center
    nearby = []

    for res in protein.residues:
        for idx in res.atom_indices:
            x, y, z = protein.coords[idx]
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            if dist <= radius:
                nearby.append(res)
                break

    return nearby
