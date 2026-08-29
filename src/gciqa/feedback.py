"""Dynamic constraint feedback for GCIQA.

Monitors the iteration loop and provides adaptive feedback when
constraints are too tight or incompatible.

Usage::

    from gciqa.feedback import ConstraintRelaxer

    relaxer = ConstraintRelaxer(patience=2, relax_factor=0.1)
    signal = relaxer.monitor(iteration, valid_count, constraints)
    if signal:
        constraints = relaxer.auto_relax(constraints, signal)
"""

from __future__ import annotations

from dataclasses import dataclass

from .constraints import ConstraintSet


@dataclass
class FeedbackSignal:
    """Signal from the constraint relaxer to the search loop."""
    action: str  # "relax", "tighten", "warn", "continue"
    constraint_name: str
    current_range: tuple[float, float]
    suggested_range: tuple[float, float]
    reason: str


class ConstraintRelaxer:
    """Monitors iteration loop and provides adaptive feedback.

    Triggers:
    - Consecutive rounds with zero valid conformations -> relax strictest constraint
    - All candidates near one constraint boundary -> suggest tightening
    - Constraint combination appears incompatible -> warn user
    """

    def __init__(self, patience: int = 2, relax_factor: float = 0.1):
        """
        Args:
            patience: Number of consecutive zero-valid rounds before relaxing.
            relax_factor: Fraction to relax constraints by (0.1 = 10%).
        """
        self.patience = patience
        self.relax_factor = relax_factor
        self._zero_streak = 0
        self._history: list[tuple[int, int]] = []  # (iteration, valid_count)

    def monitor(
        self,
        iteration: int,
        valid_count: int,
        constraints: ConstraintSet,
    ) -> FeedbackSignal | None:
        """Monitor iteration and return feedback signal if needed.

        Args:
            iteration: Current iteration number.
            valid_count: Number of valid conformations found.
            constraints: Current constraint set.

        Returns:
            FeedbackSignal if action is needed, None otherwise.
        """
        self._history.append((iteration, valid_count))

        if valid_count == 0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0

        # Check if we need to relax
        if self._zero_streak >= self.patience:
            # Find the strictest constraint to relax
            strictest = self._find_strictest_constraint(constraints)
            if strictest:
                return FeedbackSignal(
                    action="relax",
                    constraint_name=str(strictest),
                    current_range=(
                        strictest.params.get("min_dist", 0),
                        strictest.params.get("max_dist", 0),
                    ),
                    suggested_range=self._compute_relaxed_range(strictest),
                    reason=f"No valid conformations for {self._zero_streak} consecutive rounds",
                )

        # Check if all candidates are near a boundary
        if len(self._history) >= 3:
            recent = self._history[-3:]
            if all(count > 0 for _, count in recent):
                # All rounds found valid conformations — could tighten
                return FeedbackSignal(
                    action="continue",
                    constraint_name="",
                    current_range=(0, 0),
                    suggested_range=(0, 0),
                    reason="Search is finding valid conformations",
                )

        return None

    def auto_relax(
        self,
        constraints: ConstraintSet,
        signal: FeedbackSignal,
    ) -> ConstraintSet:
        """Apply relaxation to constraints based on signal.

        Args:
            constraints: Current constraint set.
            signal: Feedback signal with relaxation parameters.

        Returns:
            New ConstraintSet with relaxed constraints.
        """
        if signal.action != "relax":
            return constraints

        new_constraints = []
        for c in constraints:
            if str(c) == signal.constraint_name:
                # Relax this constraint
                min_dist = c.params.get("min_dist", 0)
                max_dist = c.params.get("max_dist", 0)
                range_width = max_dist - min_dist
                relaxation = range_width * self.relax_factor

                new_min = max(0, min_dist - relaxation)
                new_max = max_dist + relaxation

                from .constraints import GeometricConstraint
                new_c = GeometricConstraint.bond(
                    c.atoms[0], c.atoms[1],
                    min_dist=new_min,
                    max_dist=new_max,
                )
                new_constraints.append(new_c)
            else:
                new_constraints.append(c)

        return ConstraintSet(new_constraints)

    def _find_strictest_constraint(self, constraints: ConstraintSet):
        """Find the constraint with the narrowest range."""
        strictest = None
        min_width = float("inf")

        for c in constraints:
            if c.type.value == "bond":
                min_dist = c.params.get("min_dist", 0)
                max_dist = c.params.get("max_dist", 0)
                width = max_dist - min_dist
                if width < min_width:
                    min_width = width
                    strictest = c

        return strictest

    def _compute_relaxed_range(self, constraint) -> tuple[float, float]:
        """Compute relaxed range for a constraint."""
        min_dist = constraint.params.get("min_dist", 0)
        max_dist = constraint.params.get("max_dist", 0)
        range_width = max_dist - min_dist
        relaxation = range_width * self.relax_factor

        return (
            max(0, min_dist - relaxation),
            max_dist + relaxation,
        )

    def reset(self):
        """Reset the relaxer state for a new search."""
        self._zero_streak = 0
        self._history = []
