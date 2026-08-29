"""Tests for GCIQA PDB parsing module."""

import pytest
import tempfile
from pathlib import Path

from gciqa.pdb import (
    parse_pdb,
    parse_pdb_string,
    find_metal_ions,
    get_residue_atoms,
    get_nearby_residues,
    ProteinStructure,
    ResidueInfo,
    MetalIon,
    _parse_element,
    _is_metal,
)


# Minimal PDB string for testing
MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00 10.00           O
ATOM      5  CB  ALA A   1       1.500   2.500   3.500  1.00 10.00           C
ATOM      6  N   GLY A   2       5.000   6.000   7.000  1.00 10.00           N
ATOM      7  CA  GLY A   2       6.000   7.000   8.000  1.00 10.00           C
ATOM      8  C   GLY A   2       7.000   8.000   9.000  1.00 10.00           C
ATOM      9  O   GLY A   2       8.000   9.000  10.000  1.00 10.00           O
END
"""

# PDB with metal ion
METAL_PDB = """\
ATOM      1  N   HIS A  10       1.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  HIS A  10       2.000   0.000   0.000  1.00 10.00           C
ATOM      3  NE2 HIS A  10       3.000   0.000   0.000  1.00 10.00           N
ATOM      4  N   CYS A  20       5.000   0.000   0.000  1.00 10.00           N
ATOM      5  CA  CYS A  20       6.000   0.000   0.000  1.00 10.00           C
ATOM      6  SG  CYS A  20       7.000   0.000   0.000  1.00 10.00           S
HETATM    7  ZN  ZN  A 100       4.000   0.000   0.000  1.00 10.00          ZN
END
"""

# PDB with multiple chains
MULTI_CHAIN_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  N   GLY B   1       5.000   6.000   7.000  1.00 10.00           N
ATOM      4  CA  GLY B   1       6.000   7.000   8.000  1.00 10.00           C
END
"""

# PDB with alternate conformations (altloc in column 17)
ALT_LOC_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA AALA A   1       2.000   3.000   4.000  0.60 10.00           C
ATOM      3  CA BALA A   1       2.500   3.500   4.500  0.40 10.00           C
ATOM      4  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C
END
"""


class TestParseElement:
    def test_single_letter_elements(self):
        assert _parse_element("N") == "N"
        assert _parse_element("O") == "O"
        assert _parse_element("S") == "S"

    def test_two_letter_elements(self):
        assert _parse_element("ZN") == "ZN"
        assert _parse_element("FE") == "FE"
        assert _parse_element("MG") == "MG"

    def test_ambiguous_ca(self):
        # CA in ATOM = C-alpha (carbon), CA in HETATM = Calcium
        assert _parse_element("CA", is_hetatm=False) == "C"
        assert _parse_element("CA", is_hetatm=True) == "CA"

    def test_element_from_pdb_column(self):
        # PDB element column is authoritative
        assert _parse_element("CA", element_from_pdb="C") == "C"
        assert _parse_element("CA", element_from_pdb="CA") == "CA"

    def test_empty_name(self):
        assert _parse_element("") == "C"


class TestIsMetal:
    def test_common_metals(self):
        assert _is_metal("ZN") is True
        assert _is_metal("FE") is True
        assert _is_metal("CU") is True
        assert _is_metal("MG") is True
        assert _is_metal("CA") is True

    def test_non_metals(self):
        assert _is_metal("C") is False
        assert _is_metal("N") is False
        assert _is_metal("O") is False
        assert _is_metal("H") is False


class TestParsePdbString:
    def test_minimal_pdb(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        assert protein.n_atoms == 9
        assert protein.n_residues == 2
        assert len(protein.chains) == 1
        assert protein.chains[0] == "A"

    def test_atom_coords(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        assert protein.coords[0] == (1.0, 2.0, 3.0)
        assert protein.coords[1] == (2.0, 3.0, 4.0)

    def test_atom_elements(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        assert protein.atoms[0] == "N"
        assert protein.atoms[1] == "C"
        assert protein.atoms[3] == "O"

    def test_residue_info(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        assert protein.residues[0].name == "ALA"
        assert protein.residues[0].number == 1
        assert protein.residues[0].chain == "A"
        assert len(protein.residues[0].atom_indices) == 5

    def test_metal_detection(self):
        protein = parse_pdb_string(METAL_PDB)
        assert len(protein.metal_ions) == 1
        assert protein.metal_ions[0].element == "ZN"
        assert protein.metal_ions[0].coord == (4.0, 0.0, 0.0)

    def test_multi_chain(self):
        protein = parse_pdb_string(MULTI_CHAIN_PDB)
        assert protein.n_atoms == 4
        assert len(protein.chains) == 2
        assert "A" in protein.chains
        assert "B" in protein.chains

    def test_alt_loc_filtering(self):
        protein = parse_pdb_string(ALT_LOC_PDB)
        # Should keep only altloc A (or first if no altloc)
        assert protein.n_atoms == 3  # N, CA (altloc A), C

    def test_empty_string(self):
        protein = parse_pdb_string("")
        assert protein.n_atoms == 0
        assert protein.n_residues == 0

    def test_remark_lines_ignored(self):
        pdb = """\
REMARK   1 This is a remark
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
END
"""
        protein = parse_pdb_string(pdb)
        assert protein.n_atoms == 1


class TestParsePdbFile:
    def test_parse_file(self, tmp_path):
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(MINIMAL_PDB)
        protein = parse_pdb(pdb_file)
        assert protein.n_atoms == 9

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdb("/nonexistent/path.pdb")


class TestProteinStructure:
    def test_get_atoms_in_residue(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        atoms = protein.get_atoms_in_residue(0)
        assert len(atoms) == 5
        assert 0 in atoms

    def test_get_residue_for_atom(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        res = protein.get_residue_for_atom(0)
        assert res is not None
        assert res.name == "ALA"

    def test_get_chain_atoms(self):
        protein = parse_pdb_string(MULTI_CHAIN_PDB)
        chain_a = protein.get_chain_atoms("A")
        chain_b = protein.get_chain_atoms("B")
        assert len(chain_a) == 2
        assert len(chain_b) == 2

    def test_get_metal_coordination_atoms(self):
        protein = parse_pdb_string(METAL_PDB)
        # ZN is at index 6, NE2 (HIS) at index 2 is at distance 1.0
        coord_atoms = protein.get_metal_coordination_atoms(6, max_dist=2.0)
        assert 2 in coord_atoms  # NE2 at distance 1.0


class TestFindMetalIons:
    def test_find_all(self):
        protein = parse_pdb_string(METAL_PDB)
        metals = find_metal_ions(protein)
        assert len(metals) == 1

    def test_find_by_element(self):
        protein = parse_pdb_string(METAL_PDB)
        zn = find_metal_ions(protein, "ZN")
        assert len(zn) == 1
        assert zn[0].element == "ZN"

    def test_find_nonexistent(self):
        protein = parse_pdb_string(METAL_PDB)
        fe = find_metal_ions(protein, "FE")
        assert len(fe) == 0


class TestGetResidueAtoms:
    def test_by_name(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        ala_atoms = get_residue_atoms(protein, "ALA")
        assert len(ala_atoms) == 5

    def test_case_insensitive(self):
        protein = parse_pdb_string(MINIMAL_PDB)
        ala_atoms = get_residue_atoms(protein, "ala")
        assert len(ala_atoms) == 5


class TestGetNearbyResidues:
    def test_nearby(self):
        protein = parse_pdb_string(METAL_PDB)
        # ZN is at (4, 0, 0), HIS has atoms at (1-3, 0, 0), CYS at (5-7, 0, 0)
        nearby = get_nearby_residues(protein, (4.0, 0.0, 0.0), radius=2.0)
        names = [r.name for r in nearby]
        assert "HIS" in names  # NE2 at distance 1.0
        assert "CYS" in names  # N at distance 1.0

    def test_no_nearby(self):
        protein = parse_pdb_string(METAL_PDB)
        nearby = get_nearby_residues(protein, (100.0, 100.0, 100.0), radius=1.0)
        assert len(nearby) == 0
