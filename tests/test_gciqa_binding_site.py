"""Tests for GCIQA binding site detection."""

import pytest

from gciqa.binding_site import (
    BindingSite,
    MetalSiteDetector,
    PocketDetector,
    SiteDetector,
)
from gciqa.pdb import MetalIon, ProteinStructure, ResidueInfo


class TestBindingSite:
    def test_creation(self):
        site = BindingSite(center=(1, 2, 3), radius=5.0)
        assert site.center == (1, 2, 3)
        assert site.radius == 5.0
        assert site.site_type == "unknown"

    def test_defaults(self):
        site = BindingSite(center=(0, 0, 0), radius=1.0)
        assert site.residues == []
        assert site.atoms == []


class TestMetalSiteDetector:
    def _make_protein(self):
        """Create a protein with a Zn ion and nearby residues."""
        atoms = ["N", "C", "O", "N", "C", "O", "ZN", "C", "C", "C"]
        coords = [
            (0.0, 0.0, 0.0),   # Res 1 (near metal)
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),  # Res 2 (far from metal)
            (11.0, 0.0, 0.0),
            (12.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),   # Metal
            (4.0, 0.0, 0.0),   # Near metal
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
        ]
        residues = [
            ResidueInfo(name="HIS", number=1, chain="A", atom_indices=[0, 1, 2]),
            ResidueInfo(name="ALA", number=2, chain="A", atom_indices=[3, 4, 5]),
            ResidueInfo(name="CYS", number=3, chain="A", atom_indices=[7, 8, 9]),
        ]
        metal_ions = [MetalIon(element="ZN", coord=(3.0, 0.0, 0.0), index=6)]
        return ProteinStructure(
            atoms=atoms,
            coords=coords,
            residues=residues,
            metal_ions=metal_ions,
            chains=["A"],
            atom_names=["N", "CA", "O", "N", "CA", "O", "ZN", "CB", "CG", "CD"],
            residue_names=["HIS", "HIS", "HIS", "ALA", "ALA", "ALA", "ZN", "CYS", "CYS", "CYS"],
            residue_numbers=[1, 1, 1, 2, 2, 2, 0, 3, 3, 3],
            chain_ids=["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
        )

    def test_detect_with_explicit_metal(self):
        protein = self._make_protein()
        metal = protein.metal_ions[0]
        detector = MetalSiteDetector(radius=5.0)
        site = detector.detect(protein, metal_ion=metal)

        assert site.site_type == "metal"
        assert site.center == (3.0, 0.0, 0.0)
        assert site.radius == 5.0

    def test_detect_auto_metal(self):
        protein = self._make_protein()
        detector = MetalSiteDetector(radius=5.0)
        site = detector.detect(protein)

        assert site.site_type == "metal"
        assert site.center == (3.0, 0.0, 0.0)

    def test_nearby_residues(self):
        protein = self._make_protein()
        detector = MetalSiteDetector(radius=5.0)
        site = detector.detect(protein)

        # HIS (res 0) and CYS (res 2) are near metal, ALA (res 1) is far
        assert 0 in site.residues  # HIS
        assert 2 in site.residues  # CYS
        assert 1 not in site.residues  # ALA (too far)

    def test_nearby_atoms(self):
        protein = self._make_protein()
        detector = MetalSiteDetector(radius=5.0)
        site = detector.detect(protein)

        # Atoms 0-2 (HIS), 6 (metal), 7-9 (CYS) are within 5Å of metal
        assert 0 in site.atoms
        assert 6 in site.atoms
        assert 3 not in site.atoms  # ALA atom, too far

    def test_no_metal_raises(self):
        protein = ProteinStructure(
            atoms=["C", "N"],
            coords=[(0, 0, 0), (1, 0, 0)],
        )
        detector = MetalSiteDetector()
        with pytest.raises(ValueError, match="No metal ions"):
            detector.detect(protein)

    def test_is_site_detector(self):
        detector = MetalSiteDetector()
        assert isinstance(detector, SiteDetector)


class TestPocketDetector:
    def test_empty_protein(self):
        protein = ProteinStructure()
        detector = PocketDetector()
        site = detector.detect(protein)
        assert site.center == (0, 0, 0)

    def test_detect_pocket(self):
        # Create a protein with a cavity at (5, 5, 5)
        atoms = ["C"] * 20
        coords = []
        # Shell of atoms around (5, 5, 5)
        import math
        for i in range(20):
            angle = 2 * math.pi * i / 20
            r = 3.0
            coords.append((5 + r * math.cos(angle), 5 + r * math.sin(angle), 5))

        protein = ProteinStructure(
            atoms=atoms,
            coords=coords,
            residues=[],
        )
        detector = PocketDetector(grid_spacing=1.5)
        site = detector.detect(protein)

        # Site should be detected (non-zero radius)
        assert site.radius > 0
        assert site.site_type == "pocket"

    def test_is_site_detector(self):
        detector = PocketDetector()
        assert isinstance(detector, SiteDetector)
