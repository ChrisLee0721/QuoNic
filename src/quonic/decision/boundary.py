"""
Decision boundary calculation for compilation strategy selection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from .platforms import PlatformParams
from .analyzer import CircuitAnalysis


class Strategy(Enum):
    """Compilation strategy."""
    UNITARY = "unitary"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


@dataclass
class DecisionResult:
    """Decision boundary calculation result.

    Attributes:
        strategy: Recommended compilation strategy
        threshold: Decision boundary threshold
        cd: (c-1)*d value
        advantage: Expected fidelity advantage of unitary over dynamic
        confidence: Confidence level ("high", "medium", "low")
        explanation: Human-readable explanation
        T: Measurement tax (if applicable)
        S: Per-step success probability
    """

    strategy: Strategy
    threshold: Optional[float]
    cd: float
    advantage: float
    confidence: str
    explanation: str
    T: Optional[float] = None
    S: Optional[float] = None

    def summary(self) -> str:
        """Return a summary string of the decision."""
        lines = [
            f"Decision: {self.strategy.value.upper()}",
            f"  Threshold: {self.threshold:.1f}" if self.threshold else "  Threshold: N/A",
            f"  (c-1)*d: {self.cd:.1f}",
            f"  Expected advantage: {self.advantage:.2f}x",
            f"  Confidence: {self.confidence}",
        ]
        if self.T is not None:
            lines.append(f"  Measurement tax T: {self.T:.4f}")
        if self.S is not None:
            lines.append(f"  Per-step success S: {self.S:.4f}")
        lines.append(f"  {self.explanation}")
        return "\n".join(lines)


def calculate_decision(
    analysis: CircuitAnalysis,
    platform: PlatformParams,
) -> DecisionResult:
    """Calculate optimal compilation strategy using decision boundary.

    The decision boundary is:
        Unitary wins when (c-1)*d < |ln S| / |ln F_eff|

    where:
        S = per-step success probability (measurement tax T for superconducting)
        F_eff = effective path fidelity
        c = compilation overhead factor
        d = bare unitary depth

    Args:
        analysis: Circuit analysis results
        platform: Platform parameters

    Returns:
        DecisionResult with recommended strategy
    """
    # Get platform parameters
    S = platform.S
    T = platform.T
    F_eff = platform.effective_F
    threshold = platform.threshold

    # Calculate (c-1)*d
    cd = analysis.cd

    # Handle missing parameters
    if threshold is None or F_eff is None:
        return DecisionResult(
            strategy=Strategy.HYBRID,
            threshold=None,
            cd=cd,
            advantage=1.0,
            confidence="low",
            explanation=(
                "Insufficient platform parameters for decision. "
                "Using hybrid strategy as default."
            ),
            T=T,
            S=S,
        )

    # Edge case: no conditional operations
    if analysis.n_conditional == 0:
        return DecisionResult(
            strategy=Strategy.UNITARY,
            threshold=threshold,
            cd=cd,
            advantage=1.0,
            confidence="high",
            explanation=(
                "No conditional operations detected. "
                "Pure unitary execution is always optimal."
            ),
            T=T,
            S=S,
        )

    # Decision boundary
    if cd < threshold:
        # Unitary wins
        advantage = calculate_advantage(analysis, platform)

        # Confidence based on how far below threshold
        if cd < threshold * 0.5:
            confidence = "high"
        elif cd < threshold * 0.8:
            confidence = "medium"
        else:
            confidence = "low"

        explanation = (
            f"Unitary compilation recommended. "
            f"(c-1)*d = {cd:.1f} < threshold = {threshold:.1f}. "
            f"Expected advantage: {advantage:.2f}x"
        )

        return DecisionResult(
            strategy=Strategy.UNITARY,
            threshold=threshold,
            cd=cd,
            advantage=advantage,
            confidence=confidence,
            explanation=explanation,
            T=T,
            S=S,
        )
    else:
        # Dynamic wins (or comparable)
        # Calculate how much worse unitary would be
        disadvantage = calculate_disadvantage(analysis, platform)

        # Confidence based on how far above threshold
        if cd > threshold * 2:
            confidence = "high"
        elif cd > threshold * 1.2:
            confidence = "medium"
        else:
            confidence = "low"

        explanation = (
            f"Dynamic execution recommended. "
            f"(c-1)*d = {cd:.1f} > threshold = {threshold:.1f}. "
            f"Unitary would be {disadvantage:.2f}x worse."
        )

        return DecisionResult(
            strategy=Strategy.DYNAMIC,
            threshold=threshold,
            cd=cd,
            advantage=1.0 / disadvantage if disadvantage > 0 else 1.0,
            confidence=confidence,
            explanation=explanation,
            T=T,
            S=S,
        )


def calculate_advantage(
    analysis: CircuitAnalysis,
    platform: PlatformParams,
) -> float:
    """Calculate expected fidelity advantage of unitary over dynamic.

    R_total = (F_eff^((c-1)*d) / S)^N

    Args:
        analysis: Circuit analysis results
        platform: Platform parameters

    Returns:
        Expected advantage ratio (>1 means unitary is better)
    """
    N = analysis.n_conditional
    S = platform.S
    F_eff = platform.effective_F

    if S is None or F_eff is None or S == 0:
        return 1.0

    # R_total = (F_eff^((c-1)*d) / S)^N
    # For unitary: F_total = F_eff^(c*d*N)
    # For dynamic: F_total = S^N * F_eff^(d*N)
    # Ratio = F_eff^((c-1)*d*N) / S^N

    cd = analysis.cd
    advantage = (F_eff ** cd / S) ** N

    return advantage


def calculate_disadvantage(
    analysis: CircuitAnalysis,
    platform: PlatformParams,
) -> float:
    """Calculate how much worse unitary would be compared to dynamic.

    Args:
        analysis: Circuit analysis results
        platform: Platform parameters

    Returns:
        Disadvantage ratio (>1 means unitary is worse)
    """
    advantage = calculate_advantage(analysis, platform)
    if advantage > 0:
        return 1.0 / advantage
    return float('inf')


def compare_strategies(
    analysis: CircuitAnalysis,
    platform: PlatformParams,
) -> dict:
    """Compare both strategies and return detailed metrics.

    Args:
        analysis: Circuit analysis results
        platform: Platform parameters

    Returns:
        Dictionary with comparison metrics
    """
    decision = calculate_decision(analysis, platform)

    N = analysis.n_conditional
    S = platform.S
    F_eff = platform.effective_F
    T = platform.T

    # Calculate fidelities for both strategies
    if S is not None and F_eff is not None:
        # Dynamic: F = S^N (simplified, ignoring bare U fidelity)
        F_dynamic = S ** N

        # Unitary: F = F_eff^(c*d*N)
        F_unitary = F_eff ** (analysis.c * analysis.d * N)
    else:
        F_dynamic = None
        F_unitary = None

    return {
        "decision": decision,
        "N": N,
        "T": T,
        "S": S,
        "F_eff": F_eff,
        "threshold": decision.threshold,
        "cd": analysis.cd,
        "F_dynamic": F_dynamic,
        "F_unitary": F_unitary,
        "advantage": decision.advantage,
        "recommended": decision.strategy.value,
    }
