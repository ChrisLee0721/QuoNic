"""Tests for GCIQA metal coordination templates."""

import pytest

from gciqa.metal_templates import (
    get_metal_template,
    auto_detect_geometry,
    generate_metal_constraints,
    get_available_metals,
    get_available_geometries,
    MetalTemplate,
    METAL_COORDINATION,
    DEFAULT_GEOMETRY,
)
from gciqa.pdb import MetalIon, ProteinStructure


class TestGetMetalTemplate:
    def test_zn_tetrahedral(self):
        template = get_metal_template("ZN", "tetrahedral")
        assert template.element == "ZN"
        assert template.geometry == "tetrahedral"
        assert template.coordination_number == 4
        assert "N" in template.distances
        assert "O" in template.distances
        assert "S" in template.distances

    def test_zn_auto(self):
        template = get_metal_template("ZN")
        assert template.geometry == "tetrahedral"  # default

    def test_fe_octahedral(self):
        template = get_metal_template("FE", "octahedral")
        assert template.element == "FE"
        assert template.coordination_number == 6

    def test_unsupported_metal(self):
        with pytest.raises(ValueError, match="Unsupported metal"):
            get_metal_template("XX")

    def test_unsupported_geometry_fallback(self):
        # Unsupported geometry falls back to default (tetrahedral for ZN)
        template = get_metal_template("ZN", "cubic")
        assert template.element == "ZN"
        assert template.geometry == "tetrahedral"

    def test_all_metals_have_default(self):
        for metal in METAL_COORDINATION:
            template = get_metal_template(metal)
            assert template.element == metal


class TestAutoDetectGeometry:
    def _make_protein(self, metal_coord, ligand_coords):
        """Helper to create a minimal protein for testing."""
        atoms = ["ZN"]
        coords = [metal_coord]
        for i, c in enumerate(ligand_coords):
            atoms.append("N")
            coords.append(c)
        return ProteinStructure(atoms=atoms, coords=coords)

    def test_tetrahedral(self):
        # 4 ligands at tetrahedral positions
        metal = MetalIon(element="ZN", coord=(0, 0, 0), index=0)
        protein = self._make_protein(
            (0, 0, 0),
            [
                (1.0, 1.0, 1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (1.0, -1.0, -1.0),
            ],
        )
        geom = auto_detect_geometry(metal, protein, max_dist=2.5)
        assert geom == "tetrahedral"

    def test_octahedral(self):
        # 6 ligands at octahedral positions
        metal = MetalIon(element="FE", coord=(0, 0, 0), index=0)
        protein = self._make_protein(
            (0, 0, 0),
            [
                (2.0, 0, 0), (-2.0, 0, 0),
                (0, 2.0, 0), (0, -2.0, 0),
                (0, 0, 2.0), (0, 0, -2.0),
            ],
        )
        geom = auto_detect_geometry(metal, protein, max_dist=2.5)
        assert geom == "octahedral"

    def test_linear(self):
        # 2 ligands
        metal = MetalIon(element="CU", coord=(0, 0, 0), index=0)
        protein = self._make_protein(
            (0, 0, 0),
            [(2.0, 0, 0), (-2.0, 0, 0)],
        )
        geom = auto_detect_geometry(metal, protein, max_dist=2.5)
        assert geom == "linear"

    def test_no_ligands(self):
        metal = MetalIon(element="ZN", coord=(0, 0, 0), index=0)
        protein = self._make_protein((0, 0, 0), [(100, 100, 100)])
        geom = auto_detect_geometry(metal, protein, max_dist=2.5)
        # No ligands → default for ZN is tetrahedral
        assert geom == "tetrahedral"


class TestGenerateMetalConstraints:
    def test_zn_with_his_cys(self):
        # Zn with 1 His (N) and 1 Cys (S) coordinating
        metal = MetalIon(element="ZN", coord=(0, 0, 0), index=0)
        protein = ProteinStructure(
            atoms=["ZN", "N", "S"],
            coords=[(0, 0, 0), (2.0, 0, 0), (0, 2.3, 0)],
        )
        template = get_metal_template("ZN", "tetrahedral")
        constraints = generate_metal_constraints(metal, protein, template, max_dist=2.5)

        assert len(constraints.constraints) == 2
        # Check that constraints are bond type
        for c in constraints.constraints:
            assert c.type.value == "bond"

    def test_no_coordinators(self):
        metal = MetalIon(element="ZN", coord=(0, 0, 0), index=0)
        protein = ProteinStructure(
            atoms=["ZN", "C"],
            coords=[(0, 0, 0), (100, 100, 100)],
        )
        template = get_metal_template("ZN", "tetrahedral")
        constraints = generate_metal_constraints(metal, protein, template, max_dist=2.5)
        assert len(constraints.constraints) == 0


class TestGetAvailableMetals:
    def test_returns_list(self):
        metals = get_available_metals()
        assert isinstance(metals, list)
        assert "ZN" in metals
        assert "FE" in metals
        assert "CU" in metals


class TestGetAvailableGeometries:
    def test_zn_geometries(self):
        geoms = get_available_geometries("ZN")
        assert "tetrahedral" in geoms
        assert "octahedral" in geoms

    def test_unknown_metal(self):
        geoms = get_available_geometries("XX")
        assert geoms == []
