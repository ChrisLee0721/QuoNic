"""GCIQA: Geometric Constraint Iterative Quantum Amplitude Amplification.

Quantum conformation search using Grover search with geometric constraints
and iterative clustering feedback.

Example::

    from quonic.gciqa import GCIQA, GeometricConstraint

    # Define constraints
    constraints = [
        GeometricConstraint.bond("C1", "N2", min_dist=1.3, max_dist=1.5),
        GeometricConstraint.pocket(center=(10, 20, 30), radius=8.0),
    ]

    # Run search
    gciqa = GCIQA(n_super_atoms=50, constraints=constraints)
    result = gciqa.run(max_iterations=5)
"""

from .constraints import GeometricConstraint, ConstraintSet
from .coarsegrain import CoarseGraining, coarse_grain, binding_site_super_atoms
from .oracle import GroverOracle
from .search import grover_search
from .clustering import geometric_clustering
from .iterative import GCIQA, GCIQAResult
from .pdb import (
    ProteinStructure,
    ResidueInfo,
    MetalIon,
    parse_pdb,
    parse_pdb_string,
    find_metal_ions,
    get_residue_atoms,
    get_nearby_residues,
)
from .metal_templates import (
    MetalTemplate,
    get_metal_template,
    auto_detect_geometry,
    generate_metal_constraints,
    get_available_metals,
    get_available_geometries,
)
from .coarsegrain_adapters import CoarseGrainingStrategy
from .protein_cg import ProteinCoarseGraining
from .binding_site import BindingSite, SiteDetector, MetalSiteDetector, PocketDetector
from .constraint_adapters import ConstraintGenerator, TemplateConstraintGenerator, AdaptiveConstraintGenerator
from .validation import compute_rmsd, validate_binding_site, batch_validate, ValidationResult, BatchValidationResult
from .report import generate_report, ConstraintReport, ConstraintEvaluation, ConstraintStatus
from .failure import diagnose_failure, FailureMode, FailureReport
from .feedback import ConstraintRelaxer, FeedbackSignal
from .output import to_pdb, to_json

__all__ = [
    "CoarseGraining",
    "GeometricConstraint",
    "ConstraintSet",
    "GroverOracle",
    "binding_site_super_atoms",
    "coarse_grain",
    "grover_search",
    "geometric_clustering",
    "GCIQA",
    "GCIQAResult",
    "ProteinStructure",
    "ResidueInfo",
    "MetalIon",
    "parse_pdb",
    "parse_pdb_string",
    "find_metal_ions",
    "get_residue_atoms",
    "get_nearby_residues",
    "MetalTemplate",
    "get_metal_template",
    "auto_detect_geometry",
    "generate_metal_constraints",
    "get_available_metals",
    "get_available_geometries",
    "CoarseGrainingStrategy",
    "ProteinCoarseGraining",
    "BindingSite",
    "SiteDetector",
    "MetalSiteDetector",
    "PocketDetector",
    "ConstraintGenerator",
    "TemplateConstraintGenerator",
    "AdaptiveConstraintGenerator",
    "compute_rmsd",
    "validate_binding_site",
    "batch_validate",
    "ValidationResult",
    "BatchValidationResult",
    "generate_report",
    "ConstraintReport",
    "ConstraintEvaluation",
    "ConstraintStatus",
    "diagnose_failure",
    "FailureMode",
    "FailureReport",
    "ConstraintRelaxer",
    "FeedbackSignal",
    "to_pdb",
    "to_json",
]
