"""Main GCIQA iterative loop.

Implements the full 5-stage pipeline:
0. Preprocessing (coarse-graining)
1. Coarse global scan (quantum)
2. Local fine sampling (quantum)
3. Geometric clustering (classical)
4. Oracle update & iteration

Example::

    from quonic.gciqa import GCIQA, GeometricConstraint, ConstraintSet

    constraints = ConstraintSet([
        GeometricConstraint.pocket(center=(10, 20, 30), radius=8.0),
        GeometricConstraint.bond("C1", "N2", 1.3, 1.5),
    ])

    gciqa = GCIQA(
        n_super_atoms=50,
        constraints=constraints,
    )
    result = gciqa.run(max_iterations=5)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .constraints import ConstraintSet, GeometricConstraint
from .clustering import geometric_clustering, ClusterResult, compute_rmsd
from .coarsegrain import CoarseGraining, coarse_grain


@dataclass
class GCIQAResult:
    """Result of GCIQA iterative search.

    Attributes:
        best_conformation: Final best conformation (atom -> coords).
        convergence_history: List of (iteration, convergence_radius) pairs.
        cluster_history: Cluster results from each iteration.
        n_iterations: Total iterations performed.
        converged: Whether convergence was achieved.
        total_time: Total wall-clock time in seconds.
        coarse_graining: Coarse-graining mapping (if used).
    """

    best_conformation: dict[str, tuple[float, float, float]]
    convergence_history: list[tuple[int, float]]
    cluster_history: list[ClusterResult]
    n_iterations: int
    converged: bool
    total_time: float
    coarse_graining: CoarseGraining | None = None


class GCIQA:
    """GCIQA iterative conformation search.

    Attributes:
        n_super_atoms: Number of coarse-grained super-atoms.
        constraints: Initial geometric constraints.
        coord_range: Physical coordinate range (Angstrom).
        bits_per_coord: Bits per coordinate dimension.
        alpha: Convergence radius shrinkage factor (0 < alpha < 1).
        convergence_threshold: Convergence threshold (Angstrom).
        use_quantum: If True, use quantum Grover search. If False,
            use classical simulation for testing.
        atoms: Full atom element symbols (for coarse-graining).
        coords: Full atom coordinates (for coarse-graining).
        cg_strategy: Coarse-graining strategy ("spatial", "residue", "fragment").
    """

    def __init__(
        self,
        n_super_atoms: int,
        constraints: ConstraintSet | None = None,
        coord_range: tuple[float, float] = (-50.0, 50.0),
        bits_per_coord: int = 10,
        alpha: float = 0.7,
        convergence_threshold: float = 0.1,
        use_quantum: bool = False,
        atoms: list[str] | None = None,
        coords: list[tuple[float, float, float]] | None = None,
        cg_strategy: str = "spatial",
    ):
        self.n_super_atoms = n_super_atoms
        self.constraints = constraints or ConstraintSet()
        self.coord_range = coord_range
        self.bits_per_coord = bits_per_coord
        self.alpha = alpha
        self.convergence_threshold = convergence_threshold
        self.use_quantum = use_quantum
        self.atoms = atoms
        self.coords = coords
        self.cg_strategy = cg_strategy
        self._cg: CoarseGraining | None = None

    def run(
        self,
        max_iterations: int = 5,
        n_shots: int = 1000,
        n_clusters: int = 5,
    ) -> GCIQAResult:
        """Run the full GCIQA iterative search.

        Args:
            max_iterations: Maximum number of iterations.
            n_shots: Number of quantum measurement shots per iteration.
            n_clusters: Number of clusters for K-means.

        Returns:
            GCIQAResult with best conformation and convergence history.
        """
        t_start = time.time()

        # Stage 0: Coarse-graining
        if self.atoms and self.coords:
            self._cg = coarse_grain(
                atoms=self.atoms,
                coords=self.coords,
                strategy=self.cg_strategy,
                n_super_atoms=self.n_super_atoms,
            )
            # Remap constraints to super-atom indices
            current_constraints = self._remap_constraints()
        else:
            current_constraints = ConstraintSet(
                list(self.constraints.constraints)
            )

        convergence_history = []
        cluster_history = []

        best_conformation = None
        converged = False

        for iteration in range(max_iterations):
            # Stage 1/2: Quantum search
            conformations = self._search(
                constraints=current_constraints, n_shots=n_shots
            )

            if not conformations:
                # No valid conformations found — constraints may be too tight
                # Keep best from previous iteration and stop
                break

            # Stage 3: Geometric clustering
            cluster_result = geometric_clustering(
                conformations=conformations,
                n_clusters=n_clusters,
            )
            cluster_history.append(cluster_result)

            # Get best conformation (center of largest cluster)
            best_conformation = cluster_result.centers[cluster_result.largest_cluster]
            conv_radius = cluster_result.convergence_radius
            convergence_history.append((iteration, conv_radius))

            # Stage 4: Check convergence
            if conv_radius < self.convergence_threshold:
                converged = True
                break

            # Stage 4: Update constraints
            # Tighten bond constraints based on actual distances
            current_constraints = self._tighten_bond_constraints(
                current_constraints, best_conformation
            )

        total_time = time.time() - t_start

        if best_conformation is None:
            best_conformation = {}

        # Expand back to full coordinates if coarse-graining was used
        full_conformation = best_conformation
        if self._cg and best_conformation:
            full_conformation = self._cg.expand_conformation(best_conformation)

        return GCIQAResult(
            best_conformation=full_conformation,
            convergence_history=convergence_history,
            cluster_history=cluster_history,
            n_iterations=len(convergence_history),
            converged=converged,
            total_time=total_time,
            coarse_graining=self._cg,
        )

    def _remap_constraints(self) -> ConstraintSet:
        """Remap constraints from full atom names to super-atom indices."""
        if not self._cg:
            return self.constraints

        remapped = ConstraintSet()
        for c in self.constraints.constraints:
            # Try to map atom names to super-atom indices
            new_atoms = []
            for atom in c.atoms:
                try:
                    atom_idx = int(atom)
                    super_idx = self._cg.atom_to_super[atom_idx]
                    new_atoms.append(str(super_idx))
                except (ValueError, IndexError):
                    new_atoms.append(atom)

            new_c = GeometricConstraint(
                type=c.type,
                atoms=new_atoms,
                params=dict(c.params),
                weight=c.weight,
            )
            remapped.add(new_c)

        return remapped

    def _tighten_bond_constraints(
        self,
        constraints: ConstraintSet,
        conformation: dict[str, tuple[float, float, float]],
    ) -> ConstraintSet:
        """Tighten bond constraints based on actual distances.

        Instead of centering the new range on the single best conformation,
        we shrink the existing range by factor alpha while keeping the
        midpoint of the old range. This is more robust against outlier
        conformations from random sampling.

        Args:
            constraints: Current constraint set.
            conformation: Best conformation from this iteration.

        Returns:
            New constraint set with tightened ranges.
        """
        import math

        new_constraints = ConstraintSet()

        for c in constraints.constraints:
            if c.type.value == "bond":
                a1, a2 = c.atoms[0], c.atoms[1]
                if a1 in conformation and a2 in conformation:
                    p1 = conformation[a1]
                    p2 = conformation[a2]
                    actual_dist = math.sqrt(
                        sum((x - y) ** 2 for x, y in zip(p1, p2))
                    )

                    old_min = c.params["min_dist"]
                    old_max = c.params["max_dist"]
                    old_mid = (old_min + old_max) / 2
                    old_range = old_max - old_min

                    # Shrink range, biased toward actual distance
                    # Blend: 50% old midpoint, 50% actual distance
                    target = 0.5 * old_mid + 0.5 * actual_dist
                    new_range = self.alpha * old_range
                    new_min = max(0.0, target - new_range / 2)
                    new_max = target + new_range / 2

                    new_params = dict(c.params)
                    new_params["min_dist"] = new_min
                    new_params["max_dist"] = new_max

                    new_c = GeometricConstraint(
                        type=c.type,
                        atoms=c.atoms,
                        params=new_params,
                        weight=c.weight,
                    )
                    new_constraints.add(new_c)
                else:
                    new_constraints.add(c)
            elif c.type.value == "no_clash":
                a1, a2 = c.atoms[0], c.atoms[1]
                if a1 in conformation and a2 in conformation:
                    p1 = conformation[a1]
                    p2 = conformation[a2]
                    actual_dist = math.sqrt(
                        sum((x - y) ** 2 for x, y in zip(p1, p2))
                    )
                    old_min = c.params["min_dist"]
                    new_min = old_min + self.alpha * (actual_dist - old_min) * 0.3
                    new_params = dict(c.params)
                    new_params["min_dist"] = new_min
                    new_c = GeometricConstraint(
                        type=c.type,
                        atoms=c.atoms,
                        params=new_params,
                        weight=c.weight,
                    )
                    new_constraints.add(new_c)
                else:
                    new_constraints.add(c)
            else:
                new_constraints.add(c)

        return new_constraints

    def _search(
        self,
        constraints: ConstraintSet,
        n_shots: int,
    ) -> list[dict[str, tuple[float, float, float]]]:
        """Run quantum or classical search."""
        if self.use_quantum:
            return self._quantum_search(constraints, n_shots)
        else:
            return self._classical_search(constraints, n_shots)

    def _quantum_search(
        self,
        constraints: ConstraintSet,
        n_shots: int,
    ) -> list[dict[str, tuple[float, float, float]]]:
        """Quantum Grover search."""
        from .oracle import GroverOracle
        from .search import grover_search

        n_qubits = self.n_super_atoms * 3 * self.bits_per_coord

        oracle = GroverOracle(
            n_qubits=n_qubits,
            constraints=constraints,
            bits_per_coord=self.bits_per_coord,
        )

        result = grover_search(
            oracle=oracle,
            n_qubits=n_qubits,
            n_shots=n_shots,
        )

        return result.decode_coordinates(
            n_atoms=self.n_super_atoms,
            bits_per_coord=self.bits_per_coord,
            coord_range=self.coord_range,
        )

    def _classical_search(
        self,
        constraints: ConstraintSet,
        n_shots: int,
    ) -> list[dict[str, tuple[float, float, float]]]:
        """Classical simulation of search (for testing).

        If pocket constraints exist, generates coordinates near the pocket
        center for better hit rate.
        """
        import random

        lo, hi = self.coord_range

        # Find pocket center/radius for smarter sampling
        pocket_center = None
        pocket_radius = None
        for c in constraints.constraints:
            if c.type.value == "pocket" and c.atoms[0] == "*":
                pocket_center = (c.params["cx"], c.params["cy"], c.params["cz"])
                pocket_radius = c.params["radius"]
                break

        if pocket_center and pocket_radius:
            # Sample near pocket center with tight spread for high hit rate
            spread = pocket_radius * 0.3
            cx, cy, cz = pocket_center
            def gen_coord():
                return random.gauss(0, spread)
            center = (cx, cy, cz)
        else:
            def gen_coord():
                return random.uniform(lo, hi)
            center = (0, 0, 0)

        conformations = []
        for _ in range(n_shots * 10):
            conf = {}
            for i in range(self.n_super_atoms):
                x = center[0] + gen_coord()
                y = center[1] + gen_coord()
                z = center[2] + gen_coord()
                conf[f"{i}"] = (x, y, z)

            satisfied, score = constraints.evaluate(conf)
            if satisfied:
                conformations.append(conf)

            if len(conformations) >= n_shots:
                break

        return conformations

    def preprocess_pocket(
        self,
        pocket_center: tuple[float, float, float],
        pocket_radius: float,
    ) -> ConstraintSet:
        """Stage 0: Create initial constraints from pocket definition."""
        cs = ConstraintSet(list(self.constraints.constraints))
        cs.add(
            GeometricConstraint.pocket(
                center=pocket_center,
                radius=pocket_radius,
            )
        )
        return cs
