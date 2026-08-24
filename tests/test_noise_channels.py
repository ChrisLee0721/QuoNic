"""Tests for extended noise models: amplitude damping, phase damping, thermal relaxation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quonic.noise import (
    NoiseModel,
    amplitude_damping,
    depolarizing,
    phase_damping,
    resolve_noise,
    thermal_relaxation,
)

# ── NoiseModel construction ─────────────────────────────────────

def test_amplitude_damping_factory():
    nm = amplitude_damping(0.05)
    assert nm.amplitude_damping == 0.05
    assert nm.phase_damping == 0.0
    assert nm.single == 0.0
    assert nm.enabled


def test_phase_damping_factory():
    nm = phase_damping(0.03)
    assert nm.phase_damping == 0.03
    assert nm.amplitude_damping == 0.0
    assert nm.enabled


def test_thermal_relaxation_factory():
    nm = thermal_relaxation(t1=50e-6, t2=70e-6, gate_time=20e-9)
    assert nm.amplitude_damping > 0.0
    assert nm.phase_damping >= 0.0
    assert nm.enabled


def test_thermal_relaxation_t2_constraint():
    with pytest.raises(ValueError, match="T2"):
        thermal_relaxation(t1=50e-6, t2=110e-6, gate_time=20e-9)


def test_thermal_relaxation_positive_params():
    with pytest.raises(ValueError):
        thermal_relaxation(t1=-1.0, t2=1.0, gate_time=1.0)
    with pytest.raises(ValueError):
        thermal_relaxation(t1=1.0, t2=-1.0, gate_time=1.0)
    with pytest.raises(ValueError):
        thermal_relaxation(t1=1.0, t2=1.0, gate_time=-1.0)


def test_thermal_relaxation_formulas():
    t1, t2, gt = 50e-6, 30e-6, 20e-9
    nm = thermal_relaxation(t1, t2, gt)
    expected_amp = 1.0 - math.exp(-gt / t1)
    expected_phase = 1.0 - math.exp(-gt / t2) - expected_amp / 2.0
    assert abs(nm.amplitude_damping - expected_amp) < 1e-15
    assert abs(nm.phase_damping - max(0.0, expected_phase)) < 1e-15


def test_combined_noise_model():
    nm = NoiseModel(single=0.01, amplitude_damping=0.02, phase_damping=0.03)
    assert nm.enabled
    assert nm.single == 0.01
    assert nm.amplitude_damping == 0.02
    assert nm.phase_damping == 0.03


def test_noise_model_validation():
    with pytest.raises(ValueError):
        NoiseModel(amplitude_damping=1.5)
    with pytest.raises(ValueError):
        NoiseModel(phase_damping=-0.1)


def test_resolve_noise_still_works():
    assert resolve_noise(None) == NoiseModel()
    assert resolve_noise(0.05) == depolarizing(0.05)
    nm = amplitude_damping(0.01)
    assert resolve_noise(nm) is nm


# ── Density matrix engine integration ───────────────────────────

def test_amplitude_damping_engine():
    """Amplitude damping on |1> should decay towards |0>."""
    from quonic.simulators._density import DensityMatrixEngine

    eng = DensityMatrixEngine(1, noise=amplitude_damping(1.0))
    eng.apply("x", [0])  # prepare |1>
    # After gamma=1 amplitude damping, state should be |0>
    probs = np.real(np.diag(eng.rho))
    assert abs(probs[0] - 1.0) < 1e-10
    assert abs(probs[1] - 0.0) < 1e-10


def test_phase_damping_preserves_populations():
    """Phase damping should not change diagonal (populations)."""
    from quonic.simulators._density import DensityMatrixEngine

    eng = DensityMatrixEngine(1, noise=phase_damping(0.5))
    eng.apply("h", [0])  # prepare |+>
    probs = np.real(np.diag(eng.rho))
    assert abs(probs[0] - 0.5) < 1e-10
    assert abs(probs[1] - 0.5) < 1e-10


def test_phase_damping_reduces_coherence():
    """Phase damping should reduce off-diagonal elements."""
    from quonic.simulators._density import DensityMatrixEngine

    eng = DensityMatrixEngine(1, noise=phase_damping(1.0))
    eng.apply("h", [0])  # prepare |+>, full coherence
    # After gamma=1 phase damping, off-diag should be ~0
    assert abs(eng.rho[0, 1]) < 1e-10
    assert abs(eng.rho[1, 0]) < 1e-10


def test_no_noise_passthrough():
    """With no noise, density matrix should be pure."""
    from quonic.simulators._density import DensityMatrixEngine

    eng = DensityMatrixEngine(2)
    eng.apply("h", [0])
    eng.apply("cx", [0, 1])
    # Bell state: should have coherence
    assert abs(eng.rho[0, 3]) > 0.4
