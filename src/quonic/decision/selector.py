"""
Strategy selector for automatic compilation strategy selection.
"""

from typing import Optional, Any, Tuple

from .platforms import PlatformParams, get_platform, PLATFORMS
from .analyzer import CircuitAnalysis, analyze_circuit
from .boundary import Strategy, DecisionResult, calculate_decision


def select_strategy(
    circuit: Any,
    platform: Any,
    backend: Any = None,
    user_override: Optional[Strategy] = None,
    transpiled: bool = False,
) -> Tuple[Strategy, DecisionResult]:
    """Select optimal compilation strategy for a circuit.

    Args:
        circuit: Quantum circuit (Qiskit QuantumCircuit or similar)
        platform: Platform name (str) or PlatformParams instance
        backend: Optional backend for transpilation analysis
        user_override: Optional user-specified strategy
        transpiled: Whether circuit is already transpiled

    Returns:
        Tuple of (Strategy, DecisionResult)

    Example:
        >>> strategy, decision = select_strategy(circuit, "ibm_heron")
        >>> print(f"Recommended: {strategy.value}")
        >>> print(f"Advantage: {decision.advantage:.2f}x")
    """
    # Get platform parameters
    if isinstance(platform, str):
        platform_params = get_platform(platform)
    elif isinstance(platform, PlatformParams):
        platform_params = platform
    else:
        raise ValueError(
            f"platform must be str or PlatformParams, got {type(platform)}"
        )

    # Analyze circuit - detect QuoNic Circuit vs generic
    from ..ir import Circuit as QuoNicCircuit
    if isinstance(circuit, QuoNicCircuit):
        from .quonic_integration import analyze_quonic_circuit
        quonic_analysis = analyze_quonic_circuit(circuit, backend)
        analysis = CircuitAnalysis(
            n_qubits=quonic_analysis.n_qubits,
            depth=quonic_analysis.depth,
            n_conditional=quonic_analysis.n_conditional,
            c=quonic_analysis.c,
            d=quonic_analysis.d,
            has_midcircuit_measurement=quonic_analysis.has_midcircuit_measurement,
            control_structures=quonic_analysis.control_structures,
            n_gates=quonic_analysis.n_gates,
            gate_counts=quonic_analysis.gate_counts,
        )
    else:
        analysis = analyze_circuit(circuit, backend, transpiled)

    # Calculate decision
    decision = calculate_decision(analysis, platform_params)

    # Apply user override if specified
    if user_override is not None:
        decision = DecisionResult(
            strategy=user_override,
            threshold=decision.threshold,
            cd=decision.cd,
            advantage=decision.advantage,
            confidence=decision.confidence,
            explanation=(
                f"{decision.explanation} "
                f"(User override: {user_override.value})"
            ),
            T=decision.T,
            S=decision.S,
        )

    return decision.strategy, decision


def compare_platforms(
    circuit: Any,
    platform_names: list,
    backend: Any = None,
) -> list:
    """Compare compilation strategies across multiple platforms.

    Args:
        circuit: Quantum circuit
        platform_names: List of platform names
        backend: Optional backend for transpilation analysis

    Returns:
        List of dictionaries with comparison results

    Example:
        >>> results = compare_platforms(circuit, ["ibm_heron", "iqm_garnet"])
        >>> for r in results:
        ...     print(f"{r['platform']}: {r['strategy']} ({r['advantage']:.2f}x)")
    """
    results = []

    for name in platform_names:
        try:
            strategy, decision = select_strategy(circuit, name, backend)
            platform = get_platform(name)

            results.append({
                "platform": platform.name,
                "platform_id": name,
                "strategy": strategy.value,
                "threshold": decision.threshold,
                "cd": decision.cd,
                "advantage": decision.advantage,
                "confidence": decision.confidence,
                "T": decision.T,
                "S": decision.S,
                "F_eff": platform.effective_F,
            })
        except Exception as e:
            results.append({
                "platform": name,
                "platform_id": name,
                "strategy": "error",
                "error": str(e),
            })

    return results


def get_optimal_platform(
    circuit: Any,
    platform_names: list,
    backend: Any = None,
) -> Tuple[str, DecisionResult]:
    """Find the platform with the highest unitary advantage.

    Args:
        circuit: Quantum circuit
        platform_names: List of platform names
        backend: Optional backend for transpilation analysis

    Returns:
        Tuple of (platform_name, DecisionResult) for the best platform

    Example:
        >>> best_platform, decision = get_optimal_platform(
        ...     circuit, ["ibm_heron", "iqm_garnet", "qi_tuna17"]
        ... )
        >>> print(f"Best platform: {best_platform}")
        >>> print(f"Advantage: {decision.advantage:.2f}x")
    """
    results = compare_platforms(circuit, platform_names, backend)

    # Filter out errors
    valid_results = [r for r in results if r["strategy"] != "error"]

    if not valid_results:
        raise ValueError("No valid platform results")

    # Find platform with highest advantage for unitary
    # Or if all recommend dynamic, find the one with least disadvantage
    best = max(valid_results, key=lambda x: x["advantage"])

    # Get the DecisionResult for the best platform
    _, decision = select_strategy(circuit, best["platform_id"], backend)

    return best["platform_id"], decision
