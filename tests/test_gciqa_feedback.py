"""Tests for GCIQA constraint feedback."""

import pytest

from quonic.gciqa.feedback import ConstraintRelaxer, FeedbackSignal
from quonic.gciqa.constraints import GeometricConstraint, ConstraintSet


class TestFeedbackSignal:
    def test_creation(self):
        signal = FeedbackSignal(
            action="relax",
            constraint_name="bond(0, 1)",
            current_range=(2.0, 3.0),
            suggested_range=(1.8, 3.2),
            reason="No valid conformations",
        )
        assert signal.action == "relax"
        assert signal.constraint_name == "bond(0, 1)"


class TestConstraintRelaxer:
    def _make_constraints(self):
        return ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=3.0),
            GeometricConstraint.bond("0", "2", min_dist=1.5, max_dist=2.5),
        ])

    def test_no_signal_initially(self):
        relaxer = ConstraintRelaxer(patience=2)
        constraints = self._make_constraints()
        signal = relaxer.monitor(0, 10, constraints)
        assert signal is None

    def test_no_signal_with_valid(self):
        relaxer = ConstraintRelaxer(patience=2)
        constraints = self._make_constraints()
        relaxer.monitor(0, 10, constraints)
        signal = relaxer.monitor(1, 5, constraints)
        assert signal is None

    def test_relax_after_patience(self):
        relaxer = ConstraintRelaxer(patience=2)
        constraints = self._make_constraints()

        # Two consecutive zero-valid rounds
        relaxer.monitor(0, 0, constraints)
        signal = relaxer.monitor(1, 0, constraints)

        assert signal is not None
        assert signal.action == "relax"
        assert "No valid conformations" in signal.reason

    def test_streak_resets_on_valid(self):
        relaxer = ConstraintRelaxer(patience=3)
        constraints = self._make_constraints()

        relaxer.monitor(0, 0, constraints)
        relaxer.monitor(1, 5, constraints)  # Reset streak
        relaxer.monitor(2, 0, constraints)
        relaxer.monitor(3, 0, constraints)
        signal = relaxer.monitor(4, 0, constraints)

        # 3 consecutive zeros after reset → triggers
        assert signal is not None

    def test_auto_relax(self):
        relaxer = ConstraintRelaxer(patience=2, relax_factor=0.2)
        constraints = self._make_constraints()

        relaxer.monitor(0, 0, constraints)
        signal = relaxer.monitor(1, 0, constraints)

        relaxed = relaxer.auto_relax(constraints, signal)

        # The strictest constraint (width=1.0) should be relaxed
        # New width = 1.0 + 2 * (1.0 * 0.2) = 1.4
        for c in relaxed:
            if str(c) == signal.constraint_name:
                width = c.params["max_dist"] - c.params["min_dist"]
                assert width > 1.0  # Original width

    def test_auto_relax_no_op(self):
        relaxer = ConstraintRelaxer(patience=2)
        constraints = self._make_constraints()

        signal = FeedbackSignal(
            action="continue",
            constraint_name="",
            current_range=(0, 0),
            suggested_range=(0, 0),
            reason="ok",
        )
        result = relaxer.auto_relax(constraints, signal)
        assert len(result.constraints) == len(constraints)

    def test_reset(self):
        relaxer = ConstraintRelaxer(patience=3)
        constraints = self._make_constraints()

        relaxer.monitor(0, 0, constraints)
        relaxer.reset()
        relaxer.monitor(1, 0, constraints)
        relaxer.monitor(2, 0, constraints)
        signal = relaxer.monitor(3, 0, constraints)

        # After reset, 3 consecutive zeros → triggers
        assert signal is not None

    def test_strictest_constraint_found(self):
        relaxer = ConstraintRelaxer(patience=1)
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", min_dist=2.0, max_dist=5.0),  # width=3
            GeometricConstraint.bond("0", "2", min_dist=1.5, max_dist=2.0),  # width=0.5 (strictest)
        ])

        signal = relaxer.monitor(0, 0, constraints)
        assert signal is not None
        # Should target the narrowest constraint
        assert "0" in signal.constraint_name and "2" in signal.constraint_name
