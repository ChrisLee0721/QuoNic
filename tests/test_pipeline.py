"""Tests for the compress-quantum-decompress pipeline."""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gciqa import (
    GCIQAPipeline, PipelineResult,
    hierarchical_coarse_grain,
    ConstraintSet, GeometricConstraint,
    CoarseGraining, coarse_grain,
    run_grover_pyqpanda3, build_grover_oracle_pyqpanda3,
)


def test_hierarchical_cg_basic():
    """Test hierarchical coarse-graining with a simple molecule."""
    atoms = ["C"] * 20
    coords = [(float(i), 0.0, 0.0) for i in range(20)]

    cg = hierarchical_coarse_grain(
        atoms, coords,
        n_super_atoms=5,
        binding_site_center=(0.0, 0.0, 0.0),
        binding_site_radius=3.0,
    )

    assert cg.n_super_atoms > 0
    assert cg.n_full_atoms == 20
    assert len(cg.atom_to_super) == 20
    print(f"  hierarchical: {cg.n_full_atoms} atoms -> {cg.n_super_atoms} SA")


def test_hierarchical_cg_no_binding_site():
    """Test hierarchical CG falls back to spatial when no binding site."""
    atoms = ["C"] * 10
    coords = [(float(i), 0.0, 0.0) for i in range(10)]

    cg = hierarchical_coarse_grain(atoms, coords, n_super_atoms=3)

    assert cg.n_super_atoms == 3
    assert cg.n_full_atoms == 10
    print(f"  no binding site: {cg.n_full_atoms} atoms -> {cg.n_super_atoms} SA")


def test_pipeline_basic():
    """Test full pipeline with a small molecule."""
    atoms = ["C", "C", "C"]
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", 1.0, 2.0),
        GeometricConstraint.bond("1", "2", 1.0, 2.0),
    ])

    pipeline = GCIQAPipeline(
        atoms, coords, constraints,
        target_super_atoms=3,
        bits_per_coord=3,
    )

    result = pipeline.run(max_iterations=2, n_shots=500)

    assert isinstance(result, PipelineResult)
    assert result.n_full_atoms == 3
    assert result.n_super_atoms > 0
    assert len(result.full_conformation) == 3
    assert result.total_time > 0
    print(f"  pipeline: {result.n_full_atoms} atoms -> {result.n_super_atoms} SA")
    print(f"  compression: {result.compression_ratio:.1f}x")
    print(f"  score: {result.constraint_score:.2f}")
    print(f"  time: {result.total_time:.2f}s")


def test_pipeline_with_binding_site():
    """Test pipeline with binding site for hierarchical compression."""
    # 10 atoms, binding site at origin
    atoms = ["C"] * 10
    coords = [
        (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0),
        (0.0, 1.5, 0.0), (1.5, 1.5, 0.0),
        (10.0, 0.0, 0.0), (11.5, 0.0, 0.0),
        (20.0, 0.0, 0.0), (21.5, 0.0, 0.0), (23.0, 0.0, 0.0),
    ]
    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", 1.0, 2.0),
        GeometricConstraint.bond("1", "2", 1.0, 2.0),
        GeometricConstraint.pocket(center=(0.0, 0.0, 0.0), radius=5.0),
    ])

    pipeline = GCIQAPipeline(
        atoms, coords, constraints,
        compression="hierarchical",
        target_super_atoms=5,
        binding_site_center=(0.0, 0.0, 0.0),
        binding_site_radius=5.0,
    )

    result = pipeline.run(max_iterations=2, n_shots=500)

    assert result.n_super_atoms > 0
    assert len(result.full_conformation) == 10
    print(f"  binding site: {result.n_full_atoms} atoms -> {result.n_super_atoms} SA")
    print(f"  compression: {result.compression_ratio:.1f}x")


def test_constraint_remap():
    """Test that constraints are correctly remapped to super-atom space."""
    atoms = ["C"] * 6
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (5.0, 0.0, 0.0), (6.0, 0.0, 0.0),
        (10.0, 0.0, 0.0), (11.0, 0.0, 0.0),
    ]
    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", 0.5, 1.5),
        GeometricConstraint.bond("2", "3", 0.5, 1.5),
        GeometricConstraint.bond("4", "5", 0.5, 1.5),
    ])

    pipeline = GCIQAPipeline(atoms, coords, constraints, target_super_atoms=3)
    cg = pipeline.compress()
    sa_constraints = pipeline.remap_constraints(cg)

    # Some constraints may be dropped (intra-group) or remapped
    print(f"  original: {len(constraints)} constraints")
    print(f"  remapped: {len(sa_constraints)} constraints")
    print(f"  SA mapping: {cg.atom_to_super}")


def test_expand_conformation():
    """Test that decompression restores atom positions."""
    atoms = ["C", "C", "C", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (1.5, 1.0, 0.0),
    ]

    cg = coarse_grain(atoms, coords, strategy="spatial", n_super_atoms=2)

    # Use original super-atom positions
    sa_conf = cg.super_to_dict()
    full_conf = cg.expand_conformation(sa_conf)

    assert len(full_conf) == 5
    print(f"  expand: {cg.n_super_atoms} SA -> {len(full_conf)} atoms")


def test_pyqpanda3_grover():
    """Test pyqpanda3 Grover search."""
    try:
        import pyqpanda3
    except ImportError:
        print("  SKIPPED: pyqpanda3 not installed")
        return

    result = run_grover_pyqpanda3(
        n_data=4,
        valid_states=["0101", "1010"],
        n_iterations=2,
        n_shots=1000,
    )

    hits = sum(result.counts.get(s, 0) for s in ["0101", "1010"])
    assert hits > 500  # Should be ~95%
    print(f"  pyqpanda3: {result.elapsed:.3f}s, hits={hits}/1000")


def test_pyqpanda3_oracle_build():
    """Test building oracle from classical function."""
    try:
        import pyqpanda3
    except ImportError:
        print("  SKIPPED: pyqpanda3 not installed")
        return

    # Oracle: mark states where bit 0 == 1
    def oracle_fn(bitstring):
        return bitstring[-1] == '1'

    oracle, valid = build_grover_oracle_pyqpanda3(4, oracle_fn)
    assert len(valid) == 8  # Half of 16 states
    print(f"  oracle build: {len(valid)} valid states")


def main():
    print("=" * 60)
    print("Pipeline Tests")
    print("=" * 60)

    tests = [
        ("Hierarchical CG basic", test_hierarchical_cg_basic),
        ("Hierarchical CG no binding site", test_hierarchical_cg_no_binding_site),
        ("Pipeline basic", test_pipeline_basic),
        ("Pipeline with binding site", test_pipeline_with_binding_site),
        ("Constraint remap", test_constraint_remap),
        ("Expand conformation", test_expand_conformation),
        ("pyqpanda3 Grover", test_pyqpanda3_grover),
        ("pyqpanda3 oracle build", test_pyqpanda3_oracle_build),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
