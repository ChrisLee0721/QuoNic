"""Tests for ΔG calculation pipeline."""

import pytest

pyscf = pytest.importorskip("pyscf")

from quonic.chem import Molecule, compute_dg, compute_dg_barrier


@pytest.fixture
def h2_molecule():
    """H2 molecule."""
    return Molecule.from_xyz("""
    2
    H2
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    """)


@pytest.fixture
def h2o_molecule():
    """H2O molecule."""
    return Molecule.from_xyz("""
    3
    H2O
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    """)


class TestComputeDG:
    """Test ΔG calculation."""

    def test_simple_reaction(self, h2_molecule, h2o_molecule):
        """Test ΔG calculation for a simple reaction."""
        # H2 + O -> H2O (simplified)
        o_atom = Molecule.from_xyz("""
        1
        O atom
        O  0.0  0.0  0.0
        """)
        result = compute_dg(
            reaction={
                "reactants": [h2_molecule, o_atom],
                "products": [h2o_molecule],
            },
            method="hf",
            basis="sto-3g",
            optimize=False,  # Skip optimization for speed
        )
        assert hasattr(result, "dg")
        assert hasattr(result, "d_electronic")
        assert hasattr(result, "d_zpe")
        assert hasattr(result, "d_thermal")
        assert hasattr(result, "d_solvation")
        assert isinstance(result.dg, float)

    def test_dg_components_sum(self, h2_molecule, h2o_molecule):
        """Test that ΔG components sum to total."""
        o_atom = Molecule.from_xyz("""
        1
        O atom
        O  0.0  0.0  0.0
        """)
        result = compute_dg(
            reaction={
                "reactants": [h2_molecule, o_atom],
                "products": [h2o_molecule],
            },
            method="hf",
            basis="sto-3g",
            optimize=False,
        )
        # Components should sum to total (within rounding)
        total = result.d_electronic + result.d_zpe + result.d_thermal + result.d_solvation
        assert abs(result.dg - total) < 0.01

    def test_dg_with_stoichiometry(self, h2_molecule):
        """Test ΔG with stoichiometric coefficients."""
        # H2 + H2 -> H2 + H2 (should be ~0)
        result = compute_dg(
            reaction={
                "reactants": [h2_molecule, h2_molecule],
                "products": [h2_molecule, h2_molecule],
            },
            method="hf",
            basis="sto-3g",
            optimize=False,
        )
        # ΔG should be approximately 0 for this trivial reaction
        assert abs(result.dg) < 0.1

    def test_dg_result_structure(self, h2_molecule, h2o_molecule):
        """Test DGResult has expected fields."""
        o_atom = Molecule.from_xyz("""
        1
        O atom
        O  0.0  0.0  0.0
        """)
        result = compute_dg(
            reaction={
                "reactants": [h2_molecule, o_atom],
                "products": [h2o_molecule],
            },
            method="hf",
            basis="sto-3g",
            optimize=False,
        )
        assert hasattr(result, "dg")
        assert hasattr(result, "d_electronic")
        assert hasattr(result, "d_zpe")
        assert hasattr(result, "d_thermal")
        assert hasattr(result, "d_solvation")
        assert hasattr(result, "temperature")
        assert hasattr(result, "solvent")
        assert hasattr(result, "reactant_thermo")
        assert hasattr(result, "product_thermo")

    def test_dg_empty_reaction_raises(self):
        """Test that empty reaction raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            compute_dg(
                reaction={"reactants": [], "products": []},
                method="hf",
                basis="sto-3g",
            )


class TestComputeDGBarrier:
    """Test activation energy calculation."""

    def test_barrier_calculation(self):
        """Test ΔG‡ calculation runs without error."""
        reactant = Molecule.from_xyz("""
        3
        H2O reactant
        O  0.0  0.0  0.0
        H  0.0  0.0  0.96
        H  0.0  0.96  0.0
        """)
        ts = Molecule.from_xyz("""
        3
        H2O TS
        O  0.0  0.0  0.0
        H  0.0  0.0  1.2
        H  0.0  1.2  0.0
        """)
        barrier = compute_dg_barrier(
            ts, reactant,
            method="hf",
            basis="sto-3g",
            optimize=False,
        )
        assert isinstance(barrier, float)
