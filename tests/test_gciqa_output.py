"""Tests for GCIQA output formats."""

import json
import pytest

from gciqa.output import to_pdb, to_json
from gciqa.report import generate_report, ConstraintReport
from gciqa.constraints import GeometricConstraint, ConstraintSet


class TestToPdb:
    def test_basic(self):
        conformation = {"0": (1.0, 2.0, 3.0), "1": (4.0, 5.0, 6.0)}
        atoms = ["C", "N"]
        pdb = to_pdb(conformation, atoms)

        assert "HEADER" in pdb
        assert "ATOM" in pdb
        assert "END" in pdb
        assert "1.000" in pdb
        assert "C" in pdb

    def test_with_metadata(self):
        conformation = {"0": (1.0, 2.0, 3.0)}
        atoms = ["C"]
        atom_names = ["CA"]
        residue_names = ["ALA"]
        residue_numbers = [42]
        chain_ids = ["B"]

        pdb = to_pdb(
            conformation, atoms,
            atom_names=atom_names,
            residue_names=residue_names,
            residue_numbers=residue_numbers,
            chain_ids=chain_ids,
        )

        assert "ALA" in pdb
        assert "B" in pdb

    def test_write_file(self, tmp_path):
        conformation = {"0": (1.0, 2.0, 3.0)}
        atoms = ["C"]
        outfile = tmp_path / "test.pdb"

        to_pdb(conformation, atoms, filename=str(outfile))
        assert outfile.exists()
        content = outfile.read_text()
        assert "ATOM" in content


class TestToJson:
    def test_basic(self):
        conformation = {"0": (1.0, 2.0, 3.0), "1": (4.0, 5.0, 6.0)}
        json_str = to_json(conformation)

        data = json.loads(json_str)
        assert "conformation" in data
        assert data["conformation"]["0"] == [1.0, 2.0, 3.0]

    def test_with_report(self):
        conformation = {"0": (0, 0, 0), "1": (2.5, 0, 0)}
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints)

        json_str = to_json(conformation, report=report)
        data = json.loads(json_str)

        assert "constraint_report" in data
        assert data["constraint_report"]["overall_score"] == 1.0

    def test_with_rmsd(self):
        conformation = {"0": (1.0, 2.0, 3.0)}
        json_str = to_json(conformation, rmsd=0.72)

        data = json.loads(json_str)
        assert data["rmsd"]["vs_crystal"] == 0.72

    def test_with_metadata(self):
        conformation = {"0": (1.0, 2.0, 3.0)}
        metadata = {"method": "gciqa", "iterations": 3}
        json_str = to_json(conformation, metadata=metadata)

        data = json.loads(json_str)
        assert data["metadata"]["method"] == "gciqa"

    def test_write_file(self, tmp_path):
        conformation = {"0": (1.0, 2.0, 3.0)}
        outfile = tmp_path / "test.json"

        to_json(conformation, filename=str(outfile))
        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert "conformation" in data
