"""Validation utilities for GCIQA results.

Provides RMSD computation and binding site validation for comparing
GCIQA predictions against crystal structures or reference data.

Usage::

    from quonic.gciqa.validation import compute_rmsd, validate_binding_site

    # Compute RMSD between two conformations
    rmsd = compute_rmsd(predicted_coords, reference_coords)

    # Validate binding site prediction
    result = validate_binding_site(predicted_site, crystal_site)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .binding_site import BindingSite


@dataclass
class ValidationResult:
    """Result of validating a prediction against a reference."""
    rmsd: float
    success: bool
    threshold: float
    details: dict


@dataclass
class BatchValidationResult:
    """Result of validating multiple predictions."""
    success_rate: float
    mean_rmsd: float
    std_rmsd: float
    n_success: int
    n_total: int
    results: list[ValidationResult]


def compute_rmsd(
    predicted: list[tuple[float, float, float]],
    reference: list[tuple[float, float, float]],
    atom_mapping: dict[int, int] | None = None,
) -> float:
    """Compute RMSD between two sets of coordinates.

    Args:
        predicted: Predicted coordinates.
        reference: Reference coordinates.
        atom_mapping: Optional mapping from predicted index to reference index.
            If None, assumes same ordering.

    Returns:
        RMSD in Angstroms.

    Raises:
        ValueError: If coordinate sets have different lengths (without mapping).
    """
    if atom_mapping is None:
        if len(predicted) != len(reference):
            raise ValueError(
                f"Coordinate sets have different lengths: "
                f"{len(predicted)} vs {len(reference)}"
            )
        pairs = list(zip(predicted, reference))
    else:
        pairs = []
        for pred_idx, ref_idx in atom_mapping.items():
            if pred_idx < len(predicted) and ref_idx < len(reference):
                pairs.append((predicted[pred_idx], reference[ref_idx]))

    if not pairs:
        return 0.0

    sum_sq = 0.0
    for (px, py, pz), (rx, ry, rz) in pairs:
        sum_sq += (px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2

    return math.sqrt(sum_sq / len(pairs))


def validate_binding_site(
    predicted_site: BindingSite,
    crystal_site: tuple[float, float, float],
    threshold: float = 1.0,
) -> ValidationResult:
    """Validate a predicted binding site against a crystal structure.

    Args:
        predicted_site: Predicted binding site.
        crystal_site: Crystal structure site center (x, y, z).
        threshold: RMSD threshold for success (Å).

    Returns:
        ValidationResult with RMSD and success flag.
    """
    rmsd = _distance(predicted_site.center, crystal_site)
    success = rmsd <= threshold

    return ValidationResult(
        rmsd=rmsd,
        success=success,
        threshold=threshold,
        details={
            "predicted_center": predicted_site.center,
            "crystal_center": crystal_site,
            "predicted_radius": predicted_site.radius,
            "site_type": predicted_site.site_type,
        },
    )


def batch_validate(
    predictions: list[BindingSite],
    references: list[tuple[float, float, float]],
    threshold: float = 1.0,
) -> BatchValidationResult:
    """Validate multiple predictions against references.

    Args:
        predictions: List of predicted binding sites.
        references: List of reference site centers.
        threshold: RMSD threshold for success (Å).

    Returns:
        BatchValidationResult with aggregate statistics.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) != "
            f"number of references ({len(references)})"
        )

    results = []
    for pred, ref in zip(predictions, references):
        results.append(validate_binding_site(pred, ref, threshold))

    rmsds = [r.rmsd for r in results]
    n_success = sum(1 for r in results if r.success)
    n_total = len(results)

    mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else 0.0
    variance = sum((r - mean_rmsd) ** 2 for r in rmsds) / len(rmsds) if rmsds else 0.0
    std_rmsd = math.sqrt(variance)

    return BatchValidationResult(
        success_rate=n_success / n_total if n_total > 0 else 0.0,
        mean_rmsd=mean_rmsd,
        std_rmsd=std_rmsd,
        n_success=n_success,
        n_total=n_total,
        results=results,
    )


def _distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
