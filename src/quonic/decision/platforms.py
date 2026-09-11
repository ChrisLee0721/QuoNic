"""
Platform-specific parameters for the decision boundary framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class PlatformParams:
    """Platform-specific parameters for decision boundary calculation.

    Attributes:
        name: Human-readable platform name
        F_gate: Gate fidelity (per CZ or CNOT)
        F_meas: Measurement fidelity
        T_fb: Feedback latency in microseconds
        T2: Decoherence time in microseconds
        F_eff: Effective path fidelity (defaults to F_gate if None)
        platform_type: "superconducting", "trapped_ion", "neutral_atom", or "photonic"
    """

    name: str
    F_gate: Optional[float] = None
    F_meas: Optional[float] = None
    T_fb: Optional[float] = None
    T2: Optional[float] = None
    F_eff: Optional[float] = None
    platform_type: str = "superconducting"

    # Photonic-specific parameters
    eta_det: Optional[float] = None  # Detection efficiency
    eta_mode: Optional[float] = None  # Mode coupling efficiency
    eta_mem: Optional[float] = None  # Memory/storage efficiency
    p_dark: Optional[float] = None  # Dark count probability
    tau_loss: Optional[float] = None  # Photon lifetime

    @property
    def T(self) -> float:
        """Measurement tax: T = F_meas * exp(-T_fb/T2).

        For superconducting/trapped ion platforms.
        Returns None for photonic platforms (use F_photonic instead).
        """
        if self.platform_type == "photonic":
            return None
        if self.F_meas is None or self.T_fb is None or self.T2 is None:
            return None
        return self.F_meas * np.exp(-self.T_fb / self.T2)

    @property
    def S(self) -> float:
        """Per-step success probability.

        For superconducting: S = F_meas * exp(-T_fb/T2)
        For photonic: S = eta_det * eta_mode * eta_mem * (1-p_dark)^m * exp(-t_fb/tau_loss)
        """
        if self.platform_type == "photonic":
            # Simplified photonic model
            if all(x is not None for x in [self.eta_det, self.eta_mode, self.eta_mem]):
                # Assume m=1 for single-photon, t_fb=0 for no feedback
                return self.eta_det * self.eta_mode * self.eta_mem
            return None
        return self.T

    @property
    def effective_F(self) -> Optional[float]:
        """Effective fidelity for decision boundary."""
        return self.F_eff or self.F_gate

    @property
    def threshold(self) -> Optional[float]:
        """Decision boundary threshold: |ln S| / |ln F_eff|.

        Returns None if parameters are insufficient.
        """
        S = self.S
        F_eff = self.effective_F

        if S is None or F_eff is None:
            return None

        # Avoid division by zero
        if F_eff >= 1.0:
            return float('inf')

        return abs(np.log(S)) / abs(np.log(F_eff))

    def summary(self) -> str:
        """Return a summary string of platform parameters."""
        lines = [f"Platform: {self.name}"]
        lines.append(f"  Type: {self.platform_type}")

        if self.platform_type == "photonic":
            if self.eta_det:
                lines.append(f"  eta_det: {self.eta_det:.4f}")
            if self.eta_mode:
                lines.append(f"  eta_mode: {self.eta_mode:.4f}")
            if self.eta_mem:
                lines.append(f"  eta_mem: {self.eta_mem:.4f}")
        else:
            if self.F_gate:
                lines.append(f"  F_gate: {self.F_gate:.5f}")
            if self.F_meas:
                lines.append(f"  F_meas: {self.F_meas:.4f}")
            if self.T_fb:
                lines.append(f"  T_fb: {self.T_fb} us")
            if self.T2:
                lines.append(f"  T2: {self.T2} us")

        T = self.T
        if T is not None:
            lines.append(f"  Measurement tax T: {T:.4f}")

        threshold = self.threshold
        if threshold is not None:
            lines.append(f"  Decision threshold: {threshold:.1f}")

        return "\n".join(lines)


# Platform database
# Values from experimental measurements and hardware calibration data

PLATFORMS: Dict[str, PlatformParams] = {
    # IBM Heron processors
    "ibm_heron": PlatformParams(
        name="IBM Heron (generic)",
        F_gate=0.995,
        F_meas=0.99,
        T_fb=34,
        T2=200,
        platform_type="superconducting",
    ),
    "ibm_marrakesh": PlatformParams(
        name="IBM ibm_marrakesh",
        F_gate=0.995,
        F_meas=0.99,
        T_fb=34,
        T2=200,
        platform_type="superconducting",
    ),
    "ibm_kingston": PlatformParams(
        name="IBM ibm_kingston",
        F_gate=0.995,
        F_meas=0.99,
        T_fb=34,
        T2=200,
        platform_type="superconducting",
    ),

    # IQM processors (via AWS Braket)
    "iqm_garnet": PlatformParams(
        name="IQM Garnet",
        F_gate=0.9989,
        F_meas=0.98,  # Estimated
        T_fb=50,      # Estimated
        T2=100,       # Estimated
        platform_type="superconducting",
    ),
    "iqm_emerald": PlatformParams(
        name="IQM Emerald",
        F_gate=0.9995,
        F_meas=0.98,
        T_fb=50,
        T2=100,
        platform_type="superconducting",
    ),

    # Quantum Inspire
    "qi_tuna17": PlatformParams(
        name="Quantum Inspire Tuna-17",
        F_gate=0.99763,
        F_meas=0.95,
        T_fb=100,
        T2=50,
        platform_type="superconducting",
    ),

    # Quandela Belenos (photonic)
    "quandela_belenos": PlatformParams(
        name="Quandela Belenos",
        eta_det=0.9992,    # Single-photon survival (mode 0)
        eta_mode=0.9954,   # Single-photon survival (mode 1)
        eta_mem=1.0,       # No memory in current experiments
        p_dark=0.0,        # Not measurable (min 1 photon required)
        tau_loss=None,     # Not measured
        platform_type="photonic",
    ),
}


def get_platform(name: str) -> PlatformParams:
    """Get platform parameters by name.

    Args:
        name: Platform identifier (e.g., "ibm_heron")

    Returns:
        PlatformParams instance

    Raises:
        ValueError: If platform not found
    """
    if name not in PLATFORMS:
        available = ", ".join(PLATFORMS.keys())
        raise ValueError(
            f"Unknown platform: {name}. Available: {available}"
        )
    return PLATFORMS[name]


def register_platform(params: PlatformParams) -> None:
    """Register a new platform.

    Args:
        params: PlatformParams instance to register
    """
    PLATFORMS[params.name.lower().replace(" ", "_")] = params


def list_platforms() -> list:
    """List all available platform names."""
    return list(PLATFORMS.keys())
