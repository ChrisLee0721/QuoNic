"""GCIQA: Geometric Constraint Iterative Quantum Amplitude Amplification.

Quantum conformation search using Grover search with geometric constraints
and iterative clustering feedback.

Example::

    from gciqa import GCIQA, GeometricConstraint

    # Define constraints
    constraints = [
        GeometricConstraint.bond("C1", "N2", min_dist=1.3, max_dist=1.5),
        GeometricConstraint.pocket(center=(10, 20, 30), radius=8.0),
    ]

    # Run search
    gciqa = GCIQA(n_super_atoms=50, constraints=constraints)
    result = gciqa.run(max_iterations=5)
"""

from .binding_site import BindingSite, MetalSiteDetector, PocketDetector, SiteDetector
from .clustering import geometric_clustering
from .coarsegrain import CoarseGraining, binding_site_super_atoms, coarse_grain
from .coarsegrain_adapters import CoarseGrainingStrategy
from .constraint_adapters import (
    AdaptiveConstraintGenerator,
    ConstraintGenerator,
    TemplateConstraintGenerator,
)
from .constraints import ConstraintSet, GeometricConstraint
from .failure import FailureMode, FailureReport, diagnose_failure
from .feedback import ConstraintRelaxer, FeedbackSignal
from .iterative import GCIQA, GCIQAResult
from .metal_templates import (
    MetalTemplate,
    auto_detect_geometry,
    generate_metal_constraints,
    get_available_geometries,
    get_available_metals,
    get_metal_template,
)
from .oracle import GroverOracle
from .output import to_json, to_pdb
from .pdb import (
    MetalIon,
    ProteinStructure,
    ResidueInfo,
    find_metal_ions,
    get_nearby_residues,
    get_residue_atoms,
    parse_pdb,
    parse_pdb_string,
)
from .protein_cg import ProteinCoarseGraining
from .report import ConstraintEvaluation, ConstraintReport, ConstraintStatus, generate_report
from .search import grover_search
from .validation import (
    BatchValidationResult,
    ValidationResult,
    batch_validate,
    compute_rmsd,
    validate_binding_site,
)

__all__ = [
    "GCIQA",
    "AdaptiveConstraintGenerator",
    "BatchValidationResult",
    "BindingSite",
    "CoarseGraining",
    "CoarseGrainingStrategy",
    "ConstraintEvaluation",
    "ConstraintGenerator",
    "ConstraintRelaxer",
    "ConstraintReport",
    "ConstraintSet",
    "ConstraintStatus",
    "FailureMode",
    "FailureReport",
    "FeedbackSignal",
    "GCIQAResult",
    "GeometricConstraint",
    "GroverOracle",
    "MetalIon",
    "MetalSiteDetector",
    "MetalTemplate",
    "PocketDetector",
    "ProteinCoarseGraining",
    "ProteinStructure",
    "ResidueInfo",
    "SiteDetector",
    "TemplateConstraintGenerator",
    "ValidationResult",
    "auto_detect_geometry",
    "batch_validate",
    "binding_site_super_atoms",
    "coarse_grain",
    "compute_rmsd",
    "diagnose_failure",
    "find_metal_ions",
    "generate_metal_constraints",
    "generate_report",
    "geometric_clustering",
    "get_available_geometries",
    "get_available_metals",
    "get_metal_template",
    "get_nearby_residues",
    "get_residue_atoms",
    "grover_search",
    "parse_pdb",
    "parse_pdb_string",
    "to_json",
    "to_pdb",
    "validate_binding_site",
]
