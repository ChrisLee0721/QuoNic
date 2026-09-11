"""
Tests for the decision module.
"""

import pytest
import numpy as np

from .platforms import PlatformParams, PLATFORMS, get_platform
from .analyzer import CircuitAnalysis, analyze_circuit
from .boundary import Strategy, DecisionResult, calculate_decision, calculate_advantage
from .selector import select_strategy, compare_platforms


class TestPlatformParams:
    """Test PlatformParams class."""

    def test_superconducting_platform(self):
        """Test superconducting platform parameters."""
        platform = PlatformParams(
            name="Test Platform",
            F_gate=0.995,
            F_meas=0.99,
            T_fb=34,
            T2=200,
            platform_type="superconducting",
        )

        assert platform.name == "Test Platform"
        assert platform.F_gate == 0.995

        # T = F_meas * exp(-T_fb/T2)
        expected_T = 0.99 * np.exp(-34 / 200)
        assert abs(platform.T - expected_T) < 1e-6

        # S = T for superconducting
        assert platform.S == platform.T

        # threshold = |ln S| / |ln F_eff|
        expected_threshold = abs(np.log(platform.T)) / abs(np.log(0.995))
        assert abs(platform.threshold - expected_threshold) < 1e-6

    def test_photonic_platform(self):
        """Test photonic platform parameters."""
        platform = PlatformParams(
            name="Test Photonic",
            eta_det=0.999,
            eta_mode=0.995,
            eta_mem=1.0,
            platform_type="photonic",
        )

        assert platform.platform_type == "photonic"
        assert platform.T is None  # T not defined for photonic
        assert platform.S == 0.999 * 0.995 * 1.0

    def test_platform_database(self):
        """Test platform database."""
        # Test known platforms
        assert "ibm_heron" in PLATFORMS
        assert "iqm_garnet" in PLATFORMS
        assert "quandela_belenos" in PLATFORMS

        # Test get_platform
        platform = get_platform("ibm_heron")
        assert platform.name == "IBM Heron (generic)"

        # Test unknown platform
        with pytest.raises(ValueError):
            get_platform("unknown_platform")


class TestCircuitAnalysis:
    """Test CircuitAnalysis class."""

    def test_analysis_creation(self):
        """Test creating CircuitAnalysis."""
        analysis = CircuitAnalysis(
            n_qubits=3,
            depth=100,
            n_conditional=10,
            c=1.0,
            d=1,
            has_midcircuit_measurement=False,
        )

        assert analysis.n_qubits == 3
        assert analysis.depth == 100
        assert analysis.n_conditional == 10
        assert analysis.c == 1.0
        assert analysis.d == 1

    def test_cd_property(self):
        """Test (c-1)*d calculation."""
        analysis = CircuitAnalysis(
            n_qubits=3,
            depth=100,
            n_conditional=10,
            c=2.0,
            d=5,
            has_midcircuit_measurement=False,
        )

        # (c-1)*d = (2-1)*5 = 5
        assert analysis.cd == 5.0


class TestDecisionBoundary:
    """Test decision boundary calculation."""

    def test_unitary_preferred(self):
        """Test case where unitary is preferred."""
        platform = PlatformParams(
            name="Test",
            F_gate=0.995,
            F_meas=0.99,
            T_fb=34,
            T2=200,
        )

        # Small (c-1)*d
        analysis = CircuitAnalysis(
            n_qubits=3,
            depth=100,
            n_conditional=10,
            c=1.0,
            d=1,
            has_midcircuit_measurement=False,
        )

        decision = calculate_decision(analysis, platform)

        assert decision.strategy == Strategy.UNITARY
        assert decision.advantage > 1.0
        assert decision.cd < decision.threshold

    def test_dynamic_preferred(self):
        """Test case where dynamic is preferred."""
        platform = PlatformParams(
            name="Test",
            F_gate=0.995,
            F_meas=0.99,
            T_fb=34,
            T2=200,
        )

        # Large (c-1)*d
        analysis = CircuitAnalysis(
            n_qubits=3,
            depth=1000,
            n_conditional=100,
            c=10.0,
            d=100,
            has_midcircuit_measurement=True,
        )

        decision = calculate_decision(analysis, platform)

        assert decision.strategy == Strategy.DYNAMIC
        assert decision.cd > decision.threshold

    def test_advantage_calculation(self):
        """Test advantage calculation."""
        platform = PlatformParams(
            name="Test",
            F_gate=0.995,
            F_meas=0.99,
            T_fb=34,
            T2=200,
        )

        analysis = CircuitAnalysis(
            n_qubits=3,
            depth=100,
            n_conditional=10,
            c=1.0,
            d=1,
            has_midcircuit_measurement=False,
        )

        advantage = calculate_advantage(analysis, platform)

        # Advantage should be > 1 for unitary
        assert advantage > 1.0


class TestSelector:
    """Test strategy selector."""

    def test_select_strategy(self):
        """Test strategy selection."""
        # Create a mock circuit
        class MockCircuit:
            num_qubits = 3
            depth = 100
            data = []

            def size(self):
                return 50

            def count_ops(self):
                return {"cx": 30, "h": 20}

        circuit = MockCircuit()

        # Test with string platform name
        strategy, decision = select_strategy(circuit, "ibm_heron")

        assert isinstance(strategy, Strategy)
        assert isinstance(decision, DecisionResult)

    def test_compare_platforms(self):
        """Test platform comparison."""
        class MockCircuit:
            num_qubits = 3
            depth = 100
            data = []

            def size(self):
                return 50

            def count_ops(self):
                return {"cx": 30, "h": 20}

        circuit = MockCircuit()

        results = compare_platforms(
            circuit,
            ["ibm_heron", "iqm_garnet", "qi_tuna17"]
        )

        assert len(results) == 3
        for r in results:
            assert "platform" in r
            assert "strategy" in r


def test_decision_boundary_values():
    """Test decision boundary with known values from paper."""
    # IBM Heron: T ≈ 0.83, F_gate ≈ 0.995
    # Threshold ≈ |ln(0.83)| / |ln(0.995)| ≈ 0.186 / 0.005 ≈ 37
    platform = get_platform("ibm_heron")

    T = platform.T
    threshold = platform.threshold

    # Check T is approximately 0.83
    assert 0.8 < T < 0.85, f"T = {T}, expected ~0.83"

    # Check threshold is approximately 37
    assert 30 < threshold < 45, f"threshold = {threshold}, expected ~37"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
