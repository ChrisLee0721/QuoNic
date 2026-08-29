"""Tests for GCIQA module."""

import pytest
import math

from gciqa import (
    GeometricConstraint,
    ConstraintSet,
    GroverOracle,
    grover_search,
    geometric_clustering,
    GCIQA,
    GCIQAResult,
)
from gciqa.clustering import compute_rmsd


class TestGeometricConstraint:
    """Test geometric constraint definitions."""

    def test_bond_constraint_satisfied(self):
        """Test bond constraint with valid distance."""
        c = GeometricConstraint.bond("A", "B", min_dist=1.0, max_dist=2.0)
        coords = {"A": (0, 0, 0), "B": (1.5, 0, 0)}
        assert c.evaluate(coords) is True

    def test_bond_constraint_too_short(self):
        """Test bond constraint with distance too short."""
        c = GeometricConstraint.bond("A", "B", min_dist=1.0, max_dist=2.0)
        coords = {"A": (0, 0, 0), "B": (0.5, 0, 0)}
        assert c.evaluate(coords) is False

    def test_bond_constraint_too_long(self):
        """Test bond constraint with distance too long."""
        c = GeometricConstraint.bond("A", "B", min_dist=1.0, max_dist=2.0)
        coords = {"A": (0, 0, 0), "B": (3.0, 0, 0)}
        assert c.evaluate(coords) is False

    def test_angle_constraint(self):
        """Test angle constraint."""
        c = GeometricConstraint.angle("A", "B", "C", min_deg=90, max_deg=120)
        # 90-degree angle: A at (0,1,0), B at (0,0,0), C at (1,0,0)
        coords = {"A": (0, 1, 0), "B": (0, 0, 0), "C": (1, 0, 0)}
        assert c.evaluate(coords) is True

    def test_pocket_constraint(self):
        """Test pocket constraint."""
        c = GeometricConstraint.pocket(center=(0, 0, 0), radius=5.0)
        coords = {"A": (1, 1, 1)}  # Inside pocket
        assert c.evaluate(coords) is True
        coords = {"A": (10, 0, 0)}  # Outside pocket
        assert c.evaluate(coords) is False

    def test_no_clash_constraint(self):
        """Test steric clash constraint."""
        c = GeometricConstraint.no_clash("A", "B", min_dist=2.0)
        coords = {"A": (0, 0, 0), "B": (3, 0, 0)}  # No clash
        assert c.evaluate(coords) is True
        coords = {"A": (0, 0, 0), "B": (1, 0, 0)}  # Clash
        assert c.evaluate(coords) is False

    def test_missing_atom(self):
        """Test constraint with missing atom returns False."""
        c = GeometricConstraint.bond("A", "B", min_dist=1.0, max_dist=2.0)
        coords = {"A": (0, 0, 0)}  # B is missing
        assert c.evaluate(coords) is False


class TestConstraintSet:
    """Test constraint set operations."""

    def test_all_satisfied(self):
        """Test when all constraints are satisfied."""
        cs = ConstraintSet([
            GeometricConstraint.bond("A", "B", 1.0, 2.0),
            GeometricConstraint.no_clash("A", "C", 1.5),
        ])
        coords = {"A": (0, 0, 0), "B": (1.5, 0, 0), "C": (2, 0, 0)}
        satisfied, score = cs.evaluate(coords)
        assert satisfied is True
        assert score == 2.0

    def test_one_violated(self):
        """Test when one constraint is violated."""
        cs = ConstraintSet([
            GeometricConstraint.bond("A", "B", 1.0, 2.0),
            GeometricConstraint.no_clash("A", "C", 1.5),
        ])
        coords = {"A": (0, 0, 0), "B": (1.5, 0, 0), "C": (0.5, 0, 0)}  # C clashes
        satisfied, score = cs.evaluate(coords)
        assert satisfied is False
        assert score == 1.0  # Only bond satisfied

    def test_weighted_score(self):
        """Test weighted scoring."""
        cs = ConstraintSet([
            GeometricConstraint.bond("A", "B", 1.0, 2.0, weight=2.0),
            GeometricConstraint.no_clash("A", "C", 1.5, weight=0.5),
        ])
        coords = {"A": (0, 0, 0), "B": (1.5, 0, 0), "C": (2, 0, 0)}
        _, score = cs.evaluate(coords)
        assert score == 2.5


class TestClustering:
    """Test geometric clustering."""

    def test_identical_conformations(self):
        """Test clustering identical conformations gives one cluster."""
        conf = {"A": (1.0, 2.0, 3.0), "B": (4.0, 5.0, 6.0)}
        conformations = [conf] * 10
        result = geometric_clustering(conformations, n_clusters=1)
        assert result.cluster_sizes[0] == 10
        assert result.convergence_radius < 1e-6

    def test_two_distinct_groups(self):
        """Test clustering two distinct groups."""
        group1 = [{"A": (0, 0, 0), "B": (1, 0, 0)} for _ in range(5)]
        group2 = [{"A": (10, 10, 10), "B": (11, 10, 10)} for _ in range(5)]
        conformations = group1 + group2
        result = geometric_clustering(conformations, n_clusters=2)
        # Should find 2 clusters with 5 members each
        assert sorted(result.cluster_sizes) == [5, 5]

    def test_rmsd_identical(self):
        """Test RMSD of identical conformations is 0."""
        conf = {"A": (1.0, 2.0, 3.0)}
        assert compute_rmsd(conf, conf) < 1e-10

    def test_rmsd_known_value(self):
        """Test RMSD with known value."""
        conf1 = {"A": (0, 0, 0)}
        conf2 = {"A": (1, 0, 0)}
        rmsd = compute_rmsd(conf1, conf2)
        assert abs(rmsd - 1.0) < 1e-10

    def test_too_few_conformations(self):
        """Test error when too few conformations."""
        conformations = [{"A": (0, 0, 0)}]
        with pytest.raises(ValueError, match="at least"):
            geometric_clustering(conformations, n_clusters=5)


class TestGCIQA:
    """Test the main GCIQA iterative loop."""

    def test_classical_search_finds_valid(self):
        """Test classical search finds conformations satisfying constraints."""
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=20.0),
        ])
        gciqa = GCIQA(
            n_super_atoms=3,
            constraints=constraints,
            coord_range=(-10.0, 10.0),
            use_quantum=False,
        )
        result = gciqa.run(max_iterations=1, n_shots=50, n_clusters=1)
        assert result.best_conformation
        assert result.n_iterations >= 1

    def test_convergence_with_tight_pocket(self):
        """Test convergence with a tight pocket constraint."""
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=5.0),
        ])
        gciqa = GCIQA(
            n_super_atoms=2,
            constraints=constraints,
            coord_range=(-10.0, 10.0),
            alpha=0.5,
            convergence_threshold=2.0,
            use_quantum=False,
        )
        result = gciqa.run(max_iterations=5, n_shots=200, n_clusters=1)
        assert result.n_iterations >= 1
        assert len(result.convergence_history) >= 1

    def test_result_structure(self):
        """Test GCIQAResult has expected fields."""
        gciqa = GCIQA(
            n_super_atoms=2,
            coord_range=(-5.0, 5.0),
            use_quantum=False,
        )
        result = gciqa.run(max_iterations=1, n_shots=20, n_clusters=1)
        assert hasattr(result, "best_conformation")
        assert hasattr(result, "convergence_history")
        assert hasattr(result, "cluster_history")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "converged")
        assert hasattr(result, "total_time")
        assert result.total_time > 0

    def test_preprocess_pocket(self):
        """Test pocket preprocessing creates constraints."""
        gciqa = GCIQA(n_super_atoms=5)
        cs = gciqa.preprocess_pocket(pocket_center=(10, 20, 30), pocket_radius=8.0)
        assert len(cs) == 1
        assert cs.constraints[0].params["radius"] == 8.0


class TestGroverOracle:
    """Test Grover oracle construction and correctness."""

    def test_oracle_builds_small(self):
        """Test oracle circuit builds for small search space."""
        pytest.importorskip("qiskit")
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=40.0),
        ])
        oracle = GroverOracle(n_qubits=6, constraints=constraints, bits_per_coord=2)
        qc = oracle.build()
        assert qc.num_qubits == 6

    def test_classical_oracle_pocket(self):
        """Test classical oracle correctly identifies valid states."""
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=40.0),
        ])
        oracle = GroverOracle(n_qubits=6, constraints=constraints, bits_per_coord=2)
        # Count valid states
        valid = sum(1 for s in range(64) if oracle.classical_oracle_fn(format(s, '06b')))
        assert valid == 8  # Known: 8/64 states within radius=40

    def test_classical_oracle_bond(self):
        """Test classical oracle with bond constraint."""
        # 2 atoms, 2 bits/coord = 12 qubits, coord_range=(-5, 5)
        # With 2 bits: values are -5, -1.67, 1.67, 5
        # Some pairs will have distance in [0.5, 4.0]
        constraints = ConstraintSet([
            GeometricConstraint.bond("0", "1", 0.5, 4.0),
        ])
        oracle = GroverOracle(
            n_qubits=12, constraints=constraints, bits_per_coord=2,
            coord_range=(-5.0, 5.0),
        )
        valid = sum(1 for s in range(2**12) if oracle.classical_oracle_fn(format(s, '012b')))
        assert valid > 0
        assert valid < 2**12

    def test_oracle_amplification(self):
        """Test Grover search amplifies valid states."""
        pytest.importorskip("qiskit")
        import math
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=40.0),
        ])
        oracle = GroverOracle(n_qubits=6, constraints=constraints, bits_per_coord=2)

        from gciqa.search import grover_search
        result = grover_search(oracle=oracle, n_qubits=6, n_shots=200, n_iterations=1)

        # Count valid measurements
        valid_count = sum(
            cnt for bs, cnt in result.top_states
            if oracle.classical_oracle_fn(bs)
        )
        # Should be significantly higher than 12.5% (classical)
        assert valid_count / result.n_shots > 0.3

    def test_oracle_no_valid_states(self):
        """Test oracle with no valid states gives uniform distribution."""
        pytest.importorskip("qiskit")
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=0.01),
        ])
        oracle = GroverOracle(n_qubits=6, constraints=constraints, bits_per_coord=2)
        valid = sum(1 for s in range(64) if oracle.classical_oracle_fn(format(s, '06b')))
        assert valid == 0

    def test_oracle_all_valid(self):
        """Test oracle where all states are valid."""
        pytest.importorskip("qiskit")
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(0, 0, 0), radius=100.0),
        ])
        oracle = GroverOracle(n_qubits=6, constraints=constraints, bits_per_coord=2)
        valid = sum(1 for s in range(64) if oracle.classical_oracle_fn(format(s, '06b')))
        assert valid == 64

    def test_estimate_qubits(self):
        """Test qubit estimation."""
        from gciqa.oracle import estimate_oracle_qubits
        n = estimate_oracle_qubits(n_atoms=2, bits_per_coord=4, n_constraints=1)
        assert n > 2 * 3 * 4  # More than data qubits alone


class TestCoarseGraining:
    """Test coarse-graining module."""

    def test_spatial_strategy(self):
        """Test spatial coarse-graining."""
        from gciqa.coarsegrain import coarse_grain
        # 6 atoms in 2 groups
        atoms = ["C", "C", "O", "H", "H", "H"]
        coords = [(0, 0, 0), (0.5, 0, 0), (1, 0, 0),
                  (10, 0, 0), (10.5, 0, 0), (11, 0, 0)]
        cg = coarse_grain(atoms, coords, strategy="spatial", n_super_atoms=2)
        assert cg.n_super_atoms == 2
        assert cg.n_full_atoms == 6
        # First 3 atoms should map to one super-atom, last 3 to another
        assert cg.atom_to_super[0] == cg.atom_to_super[1]
        assert cg.atom_to_super[3] == cg.atom_to_super[4]
        assert cg.atom_to_super[0] != cg.atom_to_super[3]

    def test_residue_strategy(self):
        """Test residue-based coarse-graining."""
        from gciqa.coarsegrain import coarse_grain
        atoms = ["C", "C", "O", "N"]
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        residue_ids = [0, 0, 1, 1]
        cg = coarse_grain(atoms, coords, strategy="residue", residue_ids=residue_ids)
        assert cg.n_super_atoms == 2
        assert cg.atom_to_super == [0, 0, 1, 1]

    def test_fragment_strategy(self):
        """Test fragment-based coarse-graining."""
        from gciqa.coarsegrain import coarse_grain
        # Two disconnected fragments
        atoms = ["C", "C", "C", "C"]
        coords = [(0, 0, 0), (1, 0, 0), (10, 0, 0), (11, 0, 0)]
        cg = coarse_grain(atoms, coords, strategy="fragment", cutoff=2.0)
        assert cg.n_super_atoms == 2

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        from gciqa.coarsegrain import coarse_grain
        atoms = ["C", "C"]
        coords = [(0, 0, 0), (2, 0, 0)]
        cg = coarse_grain(atoms, coords, strategy="spatial", n_super_atoms=1)
        assert cg.n_super_atoms == 1
        # Center of mass should be at (1, 0, 0) for equal masses
        assert abs(cg.super_coords[0][0] - 1.0) < 0.01

    def test_expand_conformation(self):
        """Test expanding super-atom conformation to full atoms."""
        from gciqa.coarsegrain import coarse_grain
        atoms = ["C", "C", "O", "O"]
        coords = [(0, 0, 0), (1, 0, 0), (10, 0, 0), (11, 0, 0)]
        cg = coarse_grain(atoms, coords, strategy="spatial", n_super_atoms=2)
        super_conf = {"0": (5, 5, 5), "1": (15, 15, 15)}
        full = cg.expand_conformation(super_conf)
        assert len(full) == 4
        # Atoms 0,1 should be at super-atom 0's position
        assert full["0"] == (5, 5, 5)
        assert full["1"] == (5, 5, 5)

    def test_binding_site_super_atoms(self):
        """Test finding super-atoms near binding site."""
        from gciqa.coarsegrain import coarse_grain, binding_site_super_atoms
        atoms = ["C", "C", "C"]
        coords = [(0, 0, 0), (1, 0, 0), (20, 0, 0)]
        cg = coarse_grain(atoms, coords, strategy="spatial", n_super_atoms=2)
        nearby = binding_site_super_atoms(cg, pocket_center=(0, 0, 0), pocket_radius=5.0)
        assert len(nearby) >= 1

    def test_gciqa_with_coarse_graining(self):
        """Test GCIQA with full molecular system coarse-graining."""
        from gciqa import GCIQA, GeometricConstraint, ConstraintSet
        atoms = ["C"] * 20
        coords = [(i * 1.5, 0, 0) for i in range(20)]
        # Use a pocket large enough to encompass the whole system
        constraints = ConstraintSet([
            GeometricConstraint.pocket(center=(15, 0, 0), radius=50.0),
        ])
        gciqa = GCIQA(
            n_super_atoms=5,
            constraints=constraints,
            coord_range=(-50.0, 50.0),
            use_quantum=False,
            atoms=atoms,
            coords=coords,
            cg_strategy="spatial",
        )
        result = gciqa.run(max_iterations=1, n_shots=20, n_clusters=1)
        assert result.coarse_graining is not None
        assert result.coarse_graining.n_super_atoms == 5
        # Result should have full atom coordinates
        assert len(result.best_conformation) == 20
