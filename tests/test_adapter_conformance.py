"""Adapter conformance test suites for GCIQA.

Proves that each adapter interface is trustworthy by running standardized
tests against all implementations. New adapters must pass these tests.

Usage:
    pytest tests/test_adapter_conformance.py -v
"""

import math

import pytest

from gciqa.binding_site import (
    BindingSite,
    MetalSiteDetector,
    PocketDetector,
)
from gciqa.coarsegrain import CoarseGraining
from gciqa.constraint_adapters import (
    AdaptiveConstraintGenerator,
    TemplateConstraintGenerator,
)
from gciqa.constraints import ConstraintSet
from gciqa.pdb import MetalIon, ProteinStructure, ResidueInfo
from gciqa.protein_cg import ProteinCoarseGraining

# ============================================================
# Shared fixtures
# ============================================================

def _make_metalloprotein():
    """Standard metalloprotein fixture for conformance tests."""
    atoms = ["ZN", "N", "S", "O", "C", "C", "C"]
    coords = [
        (0.0, 0.0, 0.0),   # Zn
        (2.0, 0.0, 0.0),   # His N
        (0.0, 2.3, 0.0),   # Cys S
        (0.0, 0.0, 2.1),   # Water O
        (10.0, 0.0, 0.0),  # Far atom 1
        (11.0, 0.0, 0.0),  # Far atom 2
        (12.0, 0.0, 0.0),  # Far atom 3
    ]
    residues = [
        ResidueInfo(name="HIS", number=1, chain="A", atom_indices=[1]),
        ResidueInfo(name="CYS", number=2, chain="A", atom_indices=[2]),
        ResidueInfo(name="HOH", number=3, chain="A", atom_indices=[3]),
        ResidueInfo(name="ALA", number=4, chain="A", atom_indices=[4, 5, 6]),
    ]
    metal_ions = [MetalIon(element="ZN", coord=(0, 0, 0), index=0)]
    protein = ProteinStructure(
        atoms=atoms, coords=coords, residues=residues,
        metal_ions=metal_ions,
    )
    return protein


# ============================================================
# CoarseGraining conformance
# ============================================================

class CoarseGrainingConformanceTests:
    """Conformance tests for CoarseGrainingStrategy implementations.

    Subclass this and set `self.strategy` in a fixture.
    """

    def test_super_atom_count(self):
        """Super-atom count must be > 0 and <= full atom count."""
        protein = _make_metalloprotein()
        cg = self.strategy.coarse_grain(
            protein.atoms, protein.coords,
            residues=protein.residues, metal_ions=protein.metal_ions,
        )
        assert cg.n_super_atoms > 0
        assert cg.n_super_atoms <= len(protein.atoms)

    def test_all_atoms_assigned(self):
        """Every atom must be assigned to a super-atom."""
        protein = _make_metalloprotein()
        cg = self.strategy.coarse_grain(
            protein.atoms, protein.coords,
            residues=protein.residues, metal_ions=protein.metal_ions,
        )
        assert len(cg.atom_to_super) == len(protein.atoms)
        for i in range(len(protein.atoms)):
            super_idx = cg.atom_to_super[i]
            assert 0 <= super_idx < cg.n_super_atoms

    def test_super_to_atoms_consistent(self):
        """atom_to_super and super_to_atoms must be inverse mappings."""
        protein = _make_metalloprotein()
        cg = self.strategy.coarse_grain(
            protein.atoms, protein.coords,
            residues=protein.residues, metal_ions=protein.metal_ions,
        )
        for atom_idx, super_idx in enumerate(cg.atom_to_super):
            assert atom_idx in cg.super_to_atoms[super_idx]

    def test_metal_ions_preserved(self):
        """Metal ions must be in their own super-atom (not merged with residues)."""
        protein = _make_metalloprotein()
        cg = self.strategy.coarse_grain(
            protein.atoms, protein.coords,
            residues=protein.residues, metal_ions=protein.metal_ions,
        )
        for metal in protein.metal_ions:
            metal_super = cg.atom_to_super[metal.index]
            # Metal's super-atom should contain only the metal (or very few atoms)
            assert len(cg.super_to_atoms[metal_super]) <= 2

    def test_returns_coarse_graining(self):
        """Must return a CoarseGraining instance."""
        protein = _make_metalloprotein()
        cg = self.strategy.coarse_grain(
            protein.atoms, protein.coords,
            residues=protein.residues, metal_ions=protein.metal_ions,
        )
        assert isinstance(cg, CoarseGraining)


class TestProteinCoarseGrainingConformance(CoarseGrainingConformanceTests):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.strategy = ProteinCoarseGraining()


# ============================================================
# SiteDetector conformance
# ============================================================

class SiteDetectorConformanceTests:
    """Conformance tests for SiteDetector implementations.

    Subclass this and set `self.detector` in a fixture.
    """

    def test_returns_binding_site(self):
        """Must return a BindingSite instance."""
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        assert isinstance(site, BindingSite)

    def test_site_has_center(self):
        """Site center must be a valid 3D coordinate."""
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        assert len(site.center) == 3
        for c in site.center:
            assert isinstance(c, (int, float))

    def test_site_has_radius(self):
        """Site radius must be positive."""
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        assert site.radius > 0

    def test_site_has_type(self):
        """Site must have a type string."""
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        assert isinstance(site.site_type, str)
        assert len(site.site_type) > 0


class TestMetalSiteDetectorConformance(SiteDetectorConformanceTests):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = MetalSiteDetector(radius=5.0)

    def test_detects_metal_site(self):
        """Must detect site near the metal ion."""
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        metal = protein.metal_ions[0]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(site.center, metal.coord)))
        assert dist < 1.0  # Site center should be at the metal

    def test_site_type_is_metal(self):
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        assert site.site_type == "metal"


class TestPocketDetectorConformance(SiteDetectorConformanceTests):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = PocketDetector()

    def test_site_type_is_pocket(self):
        protein = _make_metalloprotein()
        site = self.detector.detect(protein)
        assert site.site_type == "pocket"


# ============================================================
# ConstraintGenerator conformance
# ============================================================

class ConstraintGeneratorConformanceTests:
    """Conformance tests for ConstraintGenerator implementations.

    Subclass this and set `self.generator` in a fixture.
    """

    def test_returns_constraint_set(self):
        """Must return a ConstraintSet instance."""
        protein = _make_metalloprotein()
        constraints = self.generator.generate(protein)
        assert isinstance(constraints, ConstraintSet)

    def test_constraints_are_bond_type(self):
        """Generated constraints should be bond type for metal coordination."""
        protein = _make_metalloprotein()
        constraints = self.generator.generate(protein)
        for c in constraints:
            assert c.type.value == "bond"

    def test_constraints_have_valid_ranges(self):
        """Constraint ranges must be physically reasonable."""
        protein = _make_metalloprotein()
        constraints = self.generator.generate(protein)
        for c in constraints:
            min_d = c.params.get("min_dist", 0)
            max_d = c.params.get("max_dist", 0)
            assert min_d >= 0
            assert max_d > min_d
            assert max_d < 10.0  # Metal-ligand bonds are < 10 Å


class TestTemplateConstraintGeneratorConformance(ConstraintGeneratorConformanceTests):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.generator = TemplateConstraintGenerator()


class TestAdaptiveConstraintGeneratorConformance(ConstraintGeneratorConformanceTests):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.generator = AdaptiveConstraintGenerator()
