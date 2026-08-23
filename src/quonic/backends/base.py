"""Base class for backend adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result


class Backend(ABC):
    name: str = "base"
    # Set of method names this backend supports (overridden by subclasses). The scheduler uses this for capability matching and fallback.
    methods: frozenset[str] = frozenset({"statevector"})

    # Capability matrix — subclasses override to declare support.
    _CAPABILITIES: dict[str, bool] = {
        "noise": False,
        "ctrl": False,
        "mid_measure": False,
        "gpu": False,
    }

    def supports(self, method: str) -> bool:
        """Whether this backend supports the given simulation method."""
        return method in self.methods

    @abstractmethod
    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
    ) -> Result:
        """Run a circuit and return a Result with kind="counts".

        noise may be a NoiseModel, a probability value in [0, 1], or None (no noise).
        method is the simulation method (e.g. "statevector" / "stabilizer" / "matrix_product_state"),
        which only matters for backends supporting multiple methods (such as Qiskit Aer); other backends ignore it.
        """
        raise NotImplementedError
