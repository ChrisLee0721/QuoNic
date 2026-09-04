"""Tests for thermochemistry module."""

import pytest

pyscf = pytest.importorskip("pyscf")

from quonic.chem import Molecule, gibbs_free_energy, thermochemistry


@pytest.fixture
def water_molecule():
    """Water molecule at equilibrium geometry."""
    return Molecule.from_xyz("""
    3
    H2O
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    """)


@pytest.fixture
def h2_molecule():
    """H2 molecule at equilibrium geometry."""
    return Molecule.from_xyz("""
    2
    H2
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    """)


class TestThermochemistry:
    """Test thermochemistry calculations."""

    def test_water_thermochemistry(self, water_molecule):
        """Test water thermochemistry returns valid results."""
        result = thermochemistry(
            water_molecule,
            method="hf",
            basis="sto-3g",
            temperature=298.15,
        )
        assert result.zpe > 0  # ZPE should be positive
        assert result.electronic_energy < 0  # Electronic energy should be negative
        assert result.entropy > 0  # Entropy should be positive
        assert result.temperature == 298.15
        assert result.n_imaginary == 0  # Should have no imaginary frequencies

    def test_h2_thermochemistry(self, h2_molecule):
        """Test H2 thermochemistry."""
        result = thermochemistry(h2_molecule, method="hf", basis="sto-3g")
        assert result.zpe > 0
        assert result.n_imaginary == 0  # Should have no imaginary frequencies

    def test_thermochemistry_at_different_temperatures(self, water_molecule):
        """Test thermochemistry at different temperatures."""
        result_298 = thermochemistry(
            water_molecule, method="hf", basis="sto-3g", temperature=298.15
        )
        result_373 = thermochemistry(
            water_molecule, method="hf", basis="sto-3g", temperature=373.15
        )
        # Higher temperature should have higher entropy contribution
        assert result_373.entropy > result_298.entropy

    def test_thermoresult_structure(self, h2_molecule):
        """Test ThermoResult has expected fields."""
        result = thermochemistry(h2_molecule, method="hf", basis="sto-3g")
        assert hasattr(result, "zpe")
        assert hasattr(result, "enthalpy")
        assert hasattr(result, "entropy")
        assert hasattr(result, "gibbs")
        assert hasattr(result, "electronic_energy")
        assert hasattr(result, "temperature")
        assert hasattr(result, "pressure")
        assert hasattr(result, "frequencies")
        assert hasattr(result, "n_imaginary")

    def test_frequencies_are_positive(self, water_molecule):
        """Test that frequencies are positive for stable molecule."""
        result = thermochemistry(water_molecule, method="hf", basis="sto-3g")
        for freq in result.frequencies:
            assert freq > 0


class TestGibbsFreeEnergy:
    """Test Gibbs free energy convenience function."""

    def test_gibbs_free_energy(self, water_molecule):
        """Test gibbs_free_energy returns a float."""
        g = gibbs_free_energy(water_molecule, method="hf", basis="sto-3g")
        assert isinstance(g, float)

    def test_gibbs_matches_thermochemistry(self, h2_molecule):
        """Test that gibbs_free_energy matches thermochemistry result."""
        g = gibbs_free_energy(h2_molecule, method="hf", basis="sto-3g")
        thermo = thermochemistry(h2_molecule, method="hf", basis="sto-3g")
        assert abs(g - thermo.gibbs) < 1e-10
