"""Tests for geometry optimization module."""

import pytest

pyscf = pytest.importorskip("pyscf")

from quonic.chem import Molecule, optimize_geometry, optimize_transition_state


@pytest.fixture
def water_molecule():
    """Water molecule at near-equilibrium geometry."""
    return Molecule.from_xyz("""
    3
    H2O near equilibrium
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    """)


@pytest.fixture
def h2_molecule():
    """H2 molecule at near-equilibrium geometry."""
    return Molecule.from_xyz("""
    2
    H2 near equilibrium
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    """)


class TestOptimizeGeometry:
    """Test geometry optimization."""

    def test_water_optimization(self, water_molecule):
        """Test water molecule optimization converges."""
        result = optimize_geometry(
            water_molecule,
            method="hf",
            basis="sto-3g",
            max_steps=20,
        )
        assert result.converged
        assert result.gradient_norm < 1e-4
        assert result.energy < 0  # Should be negative for bound system

    def test_h2_optimization(self, h2_molecule):
        """Test H2 molecule optimization."""
        result = optimize_geometry(
            h2_molecule,
            method="hf",
            basis="sto-3g",
            max_steps=20,
        )
        assert result.converged
        assert result.molecule.n_atoms == 2

    def test_optimization_preserves_atoms(self, water_molecule):
        """Test that optimization preserves atom types."""
        result = optimize_geometry(
            water_molecule,
            method="hf",
            basis="sto-3g",
        )
        assert result.molecule.atoms == water_molecule.atoms

    def test_optimization_with_dft(self, water_molecule):
        """Test optimization with DFT method."""
        result = optimize_geometry(
            water_molecule,
            method="b3lyp",
            basis="sto-3g",
        )
        assert result.converged
        assert result.energy < 0

    def test_optimization_result_structure(self, h2_molecule):
        """Test OptimizationResult has expected fields."""
        result = optimize_geometry(h2_molecule, method="hf", basis="sto-3g")
        assert hasattr(result, "molecule")
        assert hasattr(result, "energy")
        assert hasattr(result, "converged")
        assert hasattr(result, "n_steps")
        assert hasattr(result, "gradient_norm")


class TestOptimizeTransitionState:
    """Test transition state optimization."""

    def test_ts_search_runs(self):
        """Test that TS search runs without error."""
        reactant = Molecule.from_xyz("""
        3
        H2O reactant
        O  0.0  0.0  0.0
        H  0.0  0.0  0.96
        H  0.0  0.96  0.0
        """)
        product = Molecule.from_xyz("""
        3
        H2O product
        O  0.0  0.0  0.0
        H  0.0  0.0  1.1
        H  0.0  1.1  0.0
        """)
        result = optimize_transition_state(
            reactant, product,
            method="hf",
            basis="sto-3g",
        )
        assert result.molecule.n_atoms == 3
        assert result.energy < 0

    def test_ts_rejects_mismatched_atoms(self):
        """Test that TS search rejects mismatched atoms."""
        reactant = Molecule.from_xyz("""
        2
        H2
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74
        """)
        product = Molecule.from_xyz("""
        3
        H2O
        O  0.0  0.0  0.0
        H  0.0  0.0  0.96
        H  0.0  0.96  0.0
        """)
        with pytest.raises(ValueError, match="same atoms"):
            optimize_transition_state(reactant, product)
