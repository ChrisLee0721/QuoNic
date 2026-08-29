"""Tests for GCIQA constraint report."""

import pytest

from gciqa.report import (
    generate_report,
    ConstraintReport,
    ConstraintEvaluation,
    ConstraintStatus,
)
from gciqa.constraints import GeometricConstraint, ConstraintSet


class TestConstraintStatus:
    def test_values(self):
        assert ConstraintStatus.SATISFIED == "satisfied"
        assert ConstraintStatus.PARTIAL == "partial"
        assert ConstraintStatus.VIOLATED == "violated"


class TestGenerateReport:
    def test_all_satisfied(self):
        conformation = {"0": (0, 0, 0), "1": (2.5, 0, 0)}
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints)

        assert report.satisfied_count == 1
        assert report.violated_count == 0
        assert report.overall_score == 1.0

    def test_violated(self):
        conformation = {"0": (0, 0, 0), "1": (10.0, 0, 0)}
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints)

        assert report.satisfied_count == 0
        assert report.violated_count == 1
        assert report.overall_score == 0.0

    def test_partial(self):
        conformation = {"0": (0, 0, 0), "1": (3.3, 0, 0)}
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints, partial_tolerance=0.5)

        assert report.partial_count == 1
        assert report.overall_score == 0.5

    def test_multiple_constraints(self):
        conformation = {"0": (0, 0, 0), "1": (2.5, 0, 0), "2": (0, 2.5, 0)}
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
            GeometricConstraint.bond("0", "2", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints)

        assert report.satisfied_count == 2
        assert report.overall_score == 1.0

    def test_empty_constraints(self):
        conformation = {"0": (0, 0, 0)}
        constraints = ConstraintSet([])
        report = generate_report(conformation, constraints)

        assert report.overall_score == 1.0
        assert report.satisfied_count == 0

    def test_str_output(self):
        conformation = {"0": (0, 0, 0), "1": (2.5, 0, 0)}
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints)
        text = str(report)

        assert "Constraint Satisfaction Report" in text
        assert "SA0:" in text
        assert "SA1:" in text
        assert "satisfied" in text.lower()

    def test_missing_atom_violated(self):
        conformation = {"0": (0, 0, 0)}  # Missing atom "1"
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
        ])
        report = generate_report(conformation, constraints)

        assert report.violated_count == 1
