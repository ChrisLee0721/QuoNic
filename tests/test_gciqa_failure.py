"""Tests for GCIQA failure analysis."""

import pytest

from quonic.gciqa.failure import (
    diagnose_failure,
    FailureMode,
    FailureReport,
)
from quonic.gciqa.constraints import GeometricConstraint, ConstraintSet
from quonic.gciqa.iterative import GCIQAResult


def _make_result(
    converged=True,
    n_iterations=3,
    best_conformation=None,
    convergence_history=None,
):
    """Helper to create a GCIQAResult for testing."""
    return GCIQAResult(
        best_conformation=best_conformation or {},
        convergence_history=convergence_history or [],
        cluster_history=[],
        n_iterations=n_iterations,
        converged=converged,
        total_time=1.0,
    )


class TestFailureMode:
    def test_values(self):
        assert FailureMode.NO_SOLUTION == "no_solution"
        assert FailureMode.PARTIAL_CONVERGENCE == "partial"
        assert FailureMode.LOCAL_MINIMUM == "local_minimum"
        assert FailureMode.RESOURCE_LIMIT == "resource_limit"


class TestDiagnoseFailure:
    def test_no_solution(self):
        result = _make_result(best_conformation=None, n_iterations=0)
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = diagnose_failure(result, constraints)

        assert report.failure_mode == FailureMode.NO_SOLUTION
        assert "No valid conformations" in report.suggestion

    def test_resource_limit(self):
        result = _make_result(converged=False, n_iterations=10)
        constraints = ConstraintSet([])
        report = diagnose_failure(result, constraints)

        assert report.failure_mode == FailureMode.RESOURCE_LIMIT
        assert "iteration" in report.suggestion.lower()

    def test_partial_convergence(self):
        # Conformation that violates a constraint
        result = _make_result(
            converged=True,
            best_conformation={"0": (0, 0, 0), "1": (10, 0, 0)},
            convergence_history=[(0, 0.5), (1, 0.3), (2, 0.1)],
        )
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = diagnose_failure(result, constraints)

        assert report.failure_mode == FailureMode.PARTIAL_CONVERGENCE
        assert report.best_report is not None
        assert report.best_report.violated_count > 0

    def test_str_output(self):
        result = _make_result(
            converged=True,
            best_conformation={"0": (0, 0, 0), "1": (10, 0, 0)},
        )
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = diagnose_failure(result, constraints)
        text = str(report)

        assert "Failure Diagnosis" in text
        assert "Failure Mode" in text
        assert "Suggestion" in text

    def test_convergence_path(self):
        result = _make_result(
            converged=False,
            n_iterations=3,
            convergence_history=[(0, 0.8), (1, 0.5), (2, 0.2)],
        )
        constraints = ConstraintSet([])
        report = diagnose_failure(result, constraints)

        assert len(report.convergence_path) == 3
        assert report.convergence_path[0] == 0.8
