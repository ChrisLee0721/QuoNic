"""Failure analysis for GCIQA results.

Provides structured failure reports that turn "failure" into "diagnosis"
when GCIQA finds no valid conformations.

Usage::

    from gciqa.failure import diagnose_failure

    report = diagnose_failure(gciqa_result, constraints)
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .constraints import ConstraintSet
from .report import ConstraintReport, generate_report

if TYPE_CHECKING:
    from .iterative import GCIQAResult


class FailureMode(str, Enum):
    """Classification of GCIQA failure modes."""
    NO_SOLUTION = "no_solution"           # No conformations satisfy all constraints
    PARTIAL_CONVERGENCE = "partial"       # Only soft constraints satisfied
    LOCAL_MINIMUM = "local_minimum"       # Converged but to local region
    RESOURCE_LIMIT = "resource_limit"     # Time/iteration limit reached


@dataclass
class FailureReport:
    """Structured failure diagnosis."""
    failure_mode: FailureMode
    best_conformation: dict | None        # Closest to satisfying all constraints
    best_report: ConstraintReport | None  # Constraint report for best
    iterations_attempted: int
    closest_candidates: list[dict]        # Top-3 closest conformations
    unsatisfiable_constraints: list[str]  # Constraints that cannot be simultaneously satisfied
    convergence_path: list[float]         # Valid ratio per iteration
    suggestion: str                       # Human-readable suggestion

    def __str__(self) -> str:
        """Human-readable failure report."""
        lines = ["=== GCIQA Failure Diagnosis ==="]
        lines.append("")
        lines.append(f"Failure Mode: {self.failure_mode.value}")
        lines.append(f"Iterations Attempted: {self.iterations_attempted}")
        lines.append("")

        if self.best_conformation:
            lines.append("Best Conformation Found:")
            for key, coord in sorted(self.best_conformation.items()):
                lines.append(f"  SA{key}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
            lines.append("")

        if self.best_report:
            lines.append(f"Constraint Score: {self.best_report.overall_score:.2f}")
            lines.append(f"  Satisfied: {self.best_report.satisfied_count}")
            lines.append(f"  Partial: {self.best_report.partial_count}")
            lines.append(f"  Violated: {self.best_report.violated_count}")
            lines.append("")

        if self.closest_candidates:
            lines.append(f"Closest Candidates: {len(self.closest_candidates)}")
            lines.append("")

        if self.unsatisfiable_constraints:
            lines.append("Potentially Unsatisfiable Constraints:")
            for c in self.unsatisfiable_constraints:
                lines.append(f"  - {c}")
            lines.append("")

        if self.convergence_path:
            lines.append("Convergence Path (valid ratio per iteration):")
            for i, ratio in enumerate(self.convergence_path):
                lines.append(f"  Iteration {i}: {100*ratio:.1f}%")
            lines.append("")

        lines.append(f"Suggestion: {self.suggestion}")

        return "\n".join(lines)


def diagnose_failure(
    result: GCIQAResult,
    constraints: ConstraintSet,
) -> FailureReport:
    """Diagnose why GCIQA failed and suggest fixes.

    Args:
        result: GCIQA result (should be a failed/converged result).
        constraints: Constraint set used in the search.

    Returns:
        FailureReport with diagnosis and suggestions.
    """
    # Determine failure mode
    if result.n_iterations == 0:
        failure_mode = FailureMode.NO_SOLUTION
    elif not result.converged:
        failure_mode = FailureMode.RESOURCE_LIMIT
    elif result.best_conformation:
        # Check if all constraints are satisfied
        report = generate_report(result.best_conformation, constraints)
        if report.violated_count > 0:
            failure_mode = FailureMode.PARTIAL_CONVERGENCE
        else:
            failure_mode = FailureMode.LOCAL_MINIMUM
    else:
        failure_mode = FailureMode.NO_SOLUTION

    # Generate report for best conformation
    best_report = None
    if result.best_conformation:
        best_report = generate_report(result.best_conformation, constraints)

    # Find closest candidates (from convergence history)
    closest_candidates = []
    if result.best_conformation:
        closest_candidates.append(result.best_conformation)

    # Identify unsatisfiable constraints
    unsatisfiable = []
    if best_report:
        for ev in best_report.constraints:
            if ev.status.value == "violated":
                unsatisfiable.append(str(ev.constraint))

    # Convergence path
    convergence_path = []
    if result.convergence_history:
        convergence_path = [ratio for _, ratio in result.convergence_history]

    # Generate suggestion
    suggestion = _generate_suggestion(failure_mode, best_report, constraints)

    return FailureReport(
        failure_mode=failure_mode,
        best_conformation=result.best_conformation,
        best_report=best_report,
        iterations_attempted=result.n_iterations,
        closest_candidates=closest_candidates,
        unsatisfiable_constraints=unsatisfiable,
        convergence_path=convergence_path,
        suggestion=suggestion,
    )


def _generate_suggestion(
    failure_mode: FailureMode,
    best_report: ConstraintReport | None,
    constraints: ConstraintSet,
) -> str:
    """Generate a human-readable suggestion based on failure mode."""
    if failure_mode == FailureMode.NO_SOLUTION:
        return (
            "No valid conformations found. Consider: "
            "(1) relaxing constraint ranges, "
            "(2) reducing the number of constraints, "
            "(3) checking if constraints are physically compatible, "
            "(4) increasing max_iterations."
        )

    if failure_mode == FailureMode.RESOURCE_LIMIT:
        return (
            "Search reached iteration/time limit without converging. Consider: "
            "(1) increasing max_iterations, "
            "(2) relaxing convergence_threshold, "
            "(3) using hierarchical search for large systems."
        )

    if failure_mode == FailureMode.PARTIAL_CONVERGENCE:
        if best_report and best_report.violated_count > 0:
            violated_names = [
                str(ev.constraint)
                for ev in best_report.constraints
                if ev.status.value == "violated"
            ]
            return (
                f"Only partial convergence achieved. "
                f"Violated constraints: {', '.join(violated_names)}. "
                f"Consider relaxing these constraints or checking if they are "
                f"physically compatible with each other."
            )
        return "Partial convergence achieved. Consider relaxing constraints."

    if failure_mode == FailureMode.LOCAL_MINIMUM:
        return (
            "Converged to a local minimum. Consider: "
            "(1) using different initial constraints, "
            "(2) running multiple searches with different random seeds, "
            "(3) using quantum Grover search for better global exploration."
        )

    return "Unknown failure mode. Check input parameters."
