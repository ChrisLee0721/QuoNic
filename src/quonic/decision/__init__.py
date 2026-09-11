"""
QuoNic Decision Module
======================

Implements the decision boundary framework for automatic compilation
strategy selection based on the measurement tax.

Reference:
    "Static Unitary Compilation Outperforms Dynamic Feedback on NISQ Hardware"
    Decision boundary: (c-1)d < |ln T| / |ln F_eff|
    Measurement tax: T = F_meas * exp(-T_fb/T2)
"""

from .platforms import PlatformParams, PLATFORMS, get_platform
from .analyzer import CircuitAnalysis, analyze_circuit
from .boundary import Strategy, DecisionResult, calculate_decision, compare_strategies
from .selector import select_strategy, compare_platforms, get_optimal_platform
from .quonic_integration import (
    QuoNicCircuitAnalysis,
    analyze_quonic_circuit,
    select_strategy_for_quonic,
)

__all__ = [
    "PlatformParams",
    "PLATFORMS",
    "get_platform",
    "CircuitAnalysis",
    "analyze_circuit",
    "Strategy",
    "DecisionResult",
    "calculate_decision",
    "compare_strategies",
    "select_strategy",
    "compare_platforms",
    "get_optimal_platform",
    "QuoNicCircuitAnalysis",
    "analyze_quonic_circuit",
    "select_strategy_for_quonic",
]
