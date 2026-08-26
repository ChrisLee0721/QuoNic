"""Constraint satisfaction report for GCIQA results.

Generates human-readable reports explaining why a conformation was selected
and how well it satisfies the geometric constraints.

Usage::

    from quonic.gciqa.report import generate_report

    report = generate_report(conformation, constraints)
    print(report)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .constraints import GeometricConstraint, ConstraintSet


class ConstraintStatus(str, Enum):
    """Status of a single constraint evaluation."""
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    VIOLATED = "violated"


@dataclass
class ConstraintEvaluation:
    """Evaluation of a single constraint against a conformation."""
    constraint: GeometricConstraint
    status: ConstraintStatus
    actual_value: float
    expected_range: tuple[float, float]
    deviation: float  # 0 if satisfied, positive if violated


@dataclass
class ConstraintReport:
    """Evidence chain connecting computation results to user trust."""
    conformation: dict[str, tuple[float, float, float]]
    constraints: list[ConstraintEvaluation]
    overall_score: float  # 0.0 (all violated) to 1.0 (all satisfied)
    satisfied_count: int
    partial_count: int
    violated_count: int

    def __str__(self) -> str:
        """Human-readable report."""
        lines = ["=== Constraint Satisfaction Report ==="]
        lines.append("")

        # Conformation summary
        lines.append("Conformation:")
        for key, coord in sorted(self.conformation.items()):
            lines.append(f"  SA{key}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
        lines.append("")

        # Individual constraints
        for i, ev in enumerate(self.constraints, 1):
            status_icon = {
                ConstraintStatus.SATISFIED: "OK",
                ConstraintStatus.PARTIAL: "~~",
                ConstraintStatus.VIOLATED: "XX",
            }[ev.status]

            lines.append(f"Constraint {i}: {ev.constraint}")
            lines.append(f"  Status: [{status_icon}] {ev.status.value.upper()}")

            if ev.constraint.type.value == "bond":
                lo, hi = ev.expected_range
                lines.append(f"  Actual: {ev.actual_value:.2f} A, Range: [{lo:.2f}, {hi:.2f}] A")
                if ev.deviation > 0:
                    lines.append(f"  Deviation: {ev.deviation:.2f} A")
            elif ev.constraint.type.value == "pocket":
                lines.append(f"  Distance to center: {ev.actual_value:.2f} A")
                lines.append(f"  Pocket radius: {ev.expected_range[1]:.2f} A")
            lines.append("")

        # Summary
        lines.append(f"Overall: {self.satisfied_count}/{len(self.constraints)} satisfied, "
                     f"{self.partial_count}/{len(self.constraints)} partial, "
                     f"{self.violated_count}/{len(self.constraints)} violated")
        lines.append(f"Score: {self.overall_score:.2f}")

        return "\n".join(lines)


def generate_report(
    conformation: dict[str, tuple[float, float, float]],
    constraints: ConstraintSet,
    partial_tolerance: float = 0.5,
) -> ConstraintReport:
    """Generate a constraint satisfaction report for a conformation.

    Args:
        conformation: Super-atom positions {index: (x, y, z)}.
        constraints: Constraint set to evaluate against.
        partial_tolerance: Distance beyond constraint range to count as
            "partial" instead of "violated" (Å).

    Returns:
        ConstraintReport with evaluation details.
    """
    evaluations = []

    for constraint in constraints:
        ev = _evaluate_constraint(conformation, constraint, partial_tolerance)
        evaluations.append(ev)

    # Compute overall score
    n = len(evaluations)
    if n == 0:
        score = 1.0
    else:
        # Satisfied = 1.0, Partial = 0.5, Violated = 0.0
        total = sum(
            1.0 if e.status == ConstraintStatus.SATISFIED
            else 0.5 if e.status == ConstraintStatus.PARTIAL
            else 0.0
            for e in evaluations
        )
        score = total / n

    satisfied = sum(1 for e in evaluations if e.status == ConstraintStatus.SATISFIED)
    partial = sum(1 for e in evaluations if e.status == ConstraintStatus.PARTIAL)
    violated = sum(1 for e in evaluations if e.status == ConstraintStatus.VIOLATED)

    return ConstraintReport(
        conformation=conformation,
        constraints=evaluations,
        overall_score=score,
        satisfied_count=satisfied,
        partial_count=partial,
        violated_count=violated,
    )


def _evaluate_constraint(
    conformation: dict[str, tuple[float, float, float]],
    constraint: GeometricConstraint,
    partial_tolerance: float,
) -> ConstraintEvaluation:
    """Evaluate a single constraint against a conformation."""
    ctype = constraint.type.value

    if ctype == "bond":
        return _evaluate_bond(conformation, constraint, partial_tolerance)
    elif ctype == "pocket":
        return _evaluate_pocket(conformation, constraint, partial_tolerance)
    else:
        # For unsupported constraint types, mark as satisfied
        return ConstraintEvaluation(
            constraint=constraint,
            status=ConstraintStatus.SATISFIED,
            actual_value=0.0,
            expected_range=(0.0, 0.0),
            deviation=0.0,
        )


def _evaluate_bond(
    conformation: dict[str, tuple[float, float, float]],
    constraint: GeometricConstraint,
    partial_tolerance: float,
) -> ConstraintEvaluation:
    """Evaluate a bond distance constraint."""
    atom1 = constraint.atoms[0]
    atom2 = constraint.atoms[1]

    if atom1 not in conformation or atom2 not in conformation:
        return ConstraintEvaluation(
            constraint=constraint,
            status=ConstraintStatus.VIOLATED,
            actual_value=0.0,
            expected_range=(constraint.params.get("min_dist", 0), constraint.params.get("max_dist", 0)),
            deviation=float("inf"),
        )

    c1 = conformation[atom1]
    c2 = conformation[atom2]
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    min_dist = constraint.params.get("min_dist", 0)
    max_dist = constraint.params.get("max_dist", float("inf"))

    if min_dist <= dist <= max_dist:
        status = ConstraintStatus.SATISFIED
        deviation = 0.0
    elif (min_dist - partial_tolerance <= dist < min_dist) or \
         (max_dist < dist <= max_dist + partial_tolerance):
        status = ConstraintStatus.PARTIAL
        deviation = max(0, min_dist - dist, dist - max_dist)
    else:
        status = ConstraintStatus.VIOLATED
        deviation = max(0, min_dist - dist, dist - max_dist)

    return ConstraintEvaluation(
        constraint=constraint,
        status=status,
        actual_value=dist,
        expected_range=(min_dist, max_dist),
        deviation=deviation,
    )


def _evaluate_pocket(
    conformation: dict[str, tuple[float, float, float]],
    constraint: GeometricConstraint,
    partial_tolerance: float,
) -> ConstraintEvaluation:
    """Evaluate a pocket constraint."""
    cx = constraint.params.get("cx", 0)
    cy = constraint.params.get("cy", 0)
    cz = constraint.params.get("cz", 0)
    radius = constraint.params.get("radius", 0)

    # Check if any super-atom is inside the pocket
    min_dist = float("inf")
    for key, coord in conformation.items():
        dist = math.sqrt(
            (coord[0] - cx) ** 2 + (coord[1] - cy) ** 2 + (coord[2] - cz) ** 2
        )
        if dist < min_dist:
            min_dist = dist

    if min_dist <= radius:
        status = ConstraintStatus.SATISFIED
        deviation = 0.0
    elif min_dist <= radius + partial_tolerance:
        status = ConstraintStatus.PARTIAL
        deviation = min_dist - radius
    else:
        status = ConstraintStatus.VIOLATED
        deviation = min_dist - radius

    return ConstraintEvaluation(
        constraint=constraint,
        status=status,
        actual_value=min_dist,
        expected_range=(0, radius),
        deviation=deviation,
    )
