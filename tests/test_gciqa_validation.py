"""Tests for GCIQA validation utilities."""

import pytest

from gciqa.validation import (
    compute_rmsd,
    validate_binding_site,
    batch_validate,
    ValidationResult,
    BatchValidationResult,
)
from gciqa.binding_site import BindingSite


class TestComputeRmsd:
    def test_identical(self):
        coords = [(1, 2, 3), (4, 5, 6)]
        assert compute_rmsd(coords, coords) == 0.0

    def test_known_rmsd(self):
        predicted = [(0, 0, 0), (1, 0, 0)]
        reference = [(0, 0, 0), (2, 0, 0)]
        rmsd = compute_rmsd(predicted, reference)
        # RMSD = sqrt((0+1)/2) = sqrt(0.5) ≈ 0.707
        assert abs(rmsd - 0.7071) < 0.001

    def test_with_mapping(self):
        predicted = [(0, 0, 0), (1, 0, 0)]
        reference = [(2, 0, 0), (0, 0, 0)]
        mapping = {0: 1, 1: 0}  # Swap order
        rmsd = compute_rmsd(predicted, reference, atom_mapping=mapping)
        assert abs(rmsd - 0.7071) < 0.001

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            compute_rmsd([(0, 0, 0)], [(0, 0, 0), (1, 0, 0)])

    def test_empty(self):
        assert compute_rmsd([], []) == 0.0


class TestValidateBindingSite:
    def test_success(self):
        predicted = BindingSite(center=(1.0, 2.0, 3.0), radius=5.0)
        crystal = (1.0, 2.0, 3.0)
        result = validate_binding_site(predicted, crystal, threshold=1.0)
        assert result.success is True
        assert result.rmsd == 0.0

    def test_failure(self):
        predicted = BindingSite(center=(10.0, 0.0, 0.0), radius=5.0)
        crystal = (0.0, 0.0, 0.0)
        result = validate_binding_site(predicted, crystal, threshold=1.0)
        assert result.success is False
        assert result.rmsd == 10.0

    def test_threshold(self):
        predicted = BindingSite(center=(1.5, 0.0, 0.0), radius=5.0)
        crystal = (0.0, 0.0, 0.0)
        result = validate_binding_site(predicted, crystal, threshold=2.0)
        assert result.success is True
        assert result.rmsd == 1.5


class TestBatchValidate:
    def test_all_success(self):
        predictions = [
            BindingSite(center=(0, 0, 0), radius=5),
            BindingSite(center=(1, 1, 1), radius=5),
        ]
        references = [(0, 0, 0), (1, 1, 1)]
        result = batch_validate(predictions, references, threshold=1.0)
        assert result.success_rate == 1.0
        assert result.n_success == 2

    def test_partial_success(self):
        predictions = [
            BindingSite(center=(0, 0, 0), radius=5),
            BindingSite(center=(10, 0, 0), radius=5),
        ]
        references = [(0, 0, 0), (0, 0, 0)]
        result = batch_validate(predictions, references, threshold=1.0)
        assert result.success_rate == 0.5
        assert result.n_success == 1

    def test_length_mismatch(self):
        predictions = [BindingSite(center=(0, 0, 0), radius=5)]
        references = [(0, 0, 0), (1, 0, 0)]
        with pytest.raises(ValueError, match="Number of predictions"):
            batch_validate(predictions, references)
