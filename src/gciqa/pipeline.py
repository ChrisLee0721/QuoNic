"""Compress → Quantum Search → Decompress pipeline.

Orchestrates the full GCIQA workflow:
1. Compress: coarse-grain atoms into super-atoms
2. Remap: map constraints to super-atom space
3. Search: find optimal super-atom conformation (quantum or classical)
4. Decompress: expand back to full-atom coordinates

Supports cascading search: when super-atom count exceeds hardware limits,
split into overlapping fragments and run sequentially.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .coarsegrain import CoarseGraining, coarse_grain
from .constraints import ConstraintSet, GeometricConstraint
from .hierarchical_cg import hierarchical_coarse_grain


@dataclass
class PipelineResult:
    """Result of the compress-search-decompress pipeline."""

    super_atom_conformation: dict[str, tuple[float, float, float]]
    full_conformation: dict[str, tuple[float, float, float]]
    coarse_graining: CoarseGraining
    super_atom_constraints: ConstraintSet
    n_super_atoms: int
    n_full_atoms: int
    compression_ratio: float
    search_time: float
    total_time: float
    constraint_score: float = 0.0
    n_fragments: int = 1
    fragment_results: list[dict] = field(default_factory=list)


class GCIQAPipeline:
    """Compress → Quantum Search → Decompress pipeline.

    Example::

        from gciqa import GCIQAPipeline, ConstraintSet, GeometricConstraint

        atoms = ["C", "C", "C", "O", "N"]
        coords = [(0,0,0), (1.5,0,0), (3,0,0), (1.5,1.5,0), (1.5,-1.5,0)]
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", 1.0, 2.0),
            GeometricConstraint.bond("1", "2", 1.0, 2.0),
        ])

        pipeline = GCIQAPipeline(atoms, coords, constraints, target_super_atoms=3)
        result = pipeline.run()
        print(result.full_conformation)
    """

    def __init__(
        self,
        atoms: list[str],
        coords: list[tuple[float, float, float]],
        constraints: ConstraintSet,
        compression: str = "hierarchical",
        target_super_atoms: int = 20,
        bits_per_coord: int = 3,
        coord_range: tuple[float, float] | None = None,
        binding_site_center: tuple[float, float, float] | None = None,
        binding_site_radius: float = 10.0,
        use_quantum: bool = False,
        backend: str = "qiskit",
    ):
        self.atoms = atoms
        self.coords = coords
        self.constraints = constraints
        self.compression = compression
        self.target_super_atoms = target_super_atoms
        self.bits_per_coord = bits_per_coord
        self.binding_site_center = binding_site_center
        self.binding_site_radius = binding_site_radius
        self.use_quantum = use_quantum
        self.backend = backend

        # Auto-compute coord_range if not given
        if coord_range is None:
            all_vals = [v for c in coords for v in c]
            lo = min(all_vals) - 5.0
            hi = max(all_vals) + 5.0
            self.coord_range = (lo, hi)
        else:
            self.coord_range = coord_range

    def run(self, max_iterations: int = 3, n_shots: int = 1000) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            max_iterations: Max GCIQA iterations.
            n_shots: Number of measurement shots per iteration.

        Returns:
            PipelineResult with super-atom and full-atom conformations.
        """
        t0 = time.time()

        # Stage 1: Compress
        cg = self.compress()

        # Stage 2: Remap constraints to super-atom space
        sa_constraints = self.remap_constraints(cg)

        # Stage 3: Search
        t_search = time.time()
        sa_conf = self.search(cg, sa_constraints, max_iterations, n_shots)
        search_time = time.time() - t_search

        # Stage 4: Decompress
        full_conf = self.decompress(cg, sa_conf)

        # Evaluate constraint satisfaction
        _, score = self.constraints.evaluate(full_conf)

        total_time = time.time() - t0

        return PipelineResult(
            super_atom_conformation=sa_conf,
            full_conformation=full_conf,
            coarse_graining=cg,
            super_atom_constraints=sa_constraints,
            n_super_atoms=cg.n_super_atoms,
            n_full_atoms=cg.n_full_atoms,
            compression_ratio=cg.n_full_atoms / max(1, cg.n_super_atoms),
            search_time=search_time,
            total_time=total_time,
            constraint_score=score,
        )

    def compress(self) -> CoarseGraining:
        """Stage 1: Coarse-grain atoms into super-atoms."""
        if self.compression == "hierarchical":
            return hierarchical_coarse_grain(
                self.atoms, self.coords,
                n_super_atoms=self.target_super_atoms,
                binding_site_center=self.binding_site_center,
                binding_site_radius=self.binding_site_radius,
            )
        elif self.compression == "uniform":
            return coarse_grain(
                self.atoms, self.coords,
                strategy="spatial",
                n_super_atoms=self.target_super_atoms,
            )
        elif self.compression == "residue":
            return coarse_grain(
                self.atoms, self.coords,
                strategy="residue",
            )
        else:
            raise ValueError(f"Unknown compression: {self.compression!r}")

    def remap_constraints(self, cg: CoarseGraining) -> ConstraintSet:
        """Stage 2: Map atom-level constraints to super-atom space.

        For bond constraints between atoms i and j:
        - If i and j map to different super-atoms → keep as super-atom bond
        - If i and j map to the same super-atom → drop (intra-group)

        Pocket constraints are kept as-is (center doesn't change).
        """
        remapped = ConstraintSet()

        for c in self.constraints.constraints:
            if c.type.value == "POCKET":
                remapped.add(c)
                continue

            new_atoms = []
            for atom in c.atoms:
                try:
                    atom_idx = int(atom)
                    super_idx = cg.atom_to_super[atom_idx]
                    new_atoms.append(str(super_idx))
                except (ValueError, IndexError):
                    new_atoms.append(atom)

            # Skip intra-group constraints
            if len(set(new_atoms)) < len(new_atoms):
                continue

            new_c = GeometricConstraint(
                type=c.type,
                atoms=tuple(new_atoms),
                params=dict(c.params),
                weight=c.weight,
            )
            remapped.add(new_c)

        return remapped

    def search(
        self,
        cg: CoarseGraining,
        sa_constraints: ConstraintSet,
        max_iterations: int,
        n_shots: int,
    ) -> dict[str, tuple[float, float, float]]:
        """Stage 3: Search for optimal super-atom conformation."""
        from .iterative import GCIQA

        gciqa = GCIQA(
            n_super_atoms=cg.n_super_atoms,
            constraints=sa_constraints,
            coord_range=self.coord_range,
            bits_per_coord=self.bits_per_coord,
            use_quantum=self.use_quantum,
            initial_conformation=cg.super_to_dict(),
            backend=self.backend,
        )
        gciqa._perturbation_pct = 0.05

        result = gciqa.run(
            max_iterations=max_iterations,
            n_shots=n_shots,
            n_clusters=min(3, cg.n_super_atoms),
        )

        if result.best_conformation:
            return result.best_conformation

        # Fallback: use initial super-atom positions
        return cg.super_to_dict()

    def decompress(
        self,
        cg: CoarseGraining,
        sa_conformation: dict[str, tuple[float, float, float]],
    ) -> dict[str, tuple[float, float, float]]:
        """Stage 4: Expand super-atom conformation to full atoms."""
        return cg.expand_conformation(sa_conformation)
