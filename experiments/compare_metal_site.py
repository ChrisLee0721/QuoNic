"""Compare GCIQA vs Distance Geometry for metal site reconstruction.

Problem: Given a metal ion and N ligands with known distances,
reconstruct the 3D coordinates.

Methods:
1. GCIQA: Direct computation with 4-bit quantization
2. Distance Geometry: Metric matrix embedding
3. Distance Geometry: Random embed + optimization

Metrics:
- Coordinate RMSD to true structure
- Distance error (quantization error for GCIQA)
- Speed
- Success rate
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Result:
    method: str
    n_ligands: int
    coord_rmsd: float
    dist_error: float
    time_us: float
    success: bool


def generate_metal_site(n_ligands: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic metal coordination site.

    Returns:
        metal_coord: (3,) array
        ligand_coords: (N, 3) array
    """
    # Metal at origin
    metal = np.array([0.0, 0.0, 0.0])

    # Ligands at typical coordination distances (1.5-3.0 A)
    ligands = []
    for i in range(n_ligands):
        # Random direction
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(1.8, 2.5)  # Typical metal-ligand distance

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        ligands.append([x, y, z])

    return metal, np.array(ligands)


def compute_distances(metal: np.ndarray, ligands: np.ndarray) -> np.ndarray:
    """Compute metal-ligand distances."""
    return np.array([np.linalg.norm(metal - l) for l in ligands])


def gciqa_quantize(distances: np.ndarray, bits: int = 4) -> tuple[np.ndarray, float]:
    """GCIQA: quantize distances to grid points.

    This is the O(n) direct computation.
    """
    step = 5.0 / (2 ** bits)  # 0.3125 A for 4-bit
    quantized = np.zeros_like(distances)

    for i, d in enumerate(distances):
        k = round(d / step - 0.5)
        k = max(0, min(k, (1 << bits) - 1))
        quantized[i] = (k + 0.5) * step

    error = np.mean(np.abs(quantized - distances))
    return quantized, error


def reconstruct_from_distances(
    distances: np.ndarray,
    n_trials: int = 100,
) -> tuple[Optional[np.ndarray], float]:
    """Reconstruct ligand coordinates from distances using optimization.

    This is what distance geometry does.
    """
    n = len(distances)
    best_coords = None
    best_error = float('inf')

    for trial in range(n_trials):
        # Random initial positions on sphere
        coords = np.random.randn(n, 3)
        coords = coords / np.linalg.norm(coords, axis=1, keepdims=True) * 2.0

        # Gradient descent
        for iteration in range(500):
            # Current distances
            current = np.linalg.norm(coords, axis=1)

            # Error
            error = np.sum((current - distances) ** 2)

            if error < best_error:
                best_error = error
                best_coords = coords.copy()

            if error < 1e-8:
                break

            # Gradient: d/dx_i (||x_i|| - d_i)^2 = 2(||x_i|| - d_i) * x_i/||x_i||
            grad = np.zeros_like(coords)
            for i in range(n):
                r = current[i]
                if r > 1e-10:
                    grad[i] = 2 * (r - distances[i]) * coords[i] / r

            coords -= 0.01 * grad

    return best_coords, np.sqrt(best_error / n)


def coord_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    # Center
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    # SVD for optimal rotation
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    c2_aligned = c2 @ R.T
    return np.sqrt(np.mean(np.sum((c1 - c2_aligned) ** 2, axis=1)))


def run_comparison(n_ligands: int, n_trials: int = 50) -> list[Result]:
    """Run comparison for a given number of ligands."""
    results = []

    for trial in range(n_trials):
        # Generate site
        metal, true_ligands = generate_metal_site(n_ligands)
        true_dists = compute_distances(metal, true_ligands)

        # Method 1: GCIQA quantization
        t0 = time.time()
        quantized, dist_error = gciqa_quantize(true_dists, bits=4)
        gciqa_time = (time.time() - t0) * 1e6  # microseconds

        # For GCIQA, we don't reconstruct coordinates - just measure distance error
        results.append(Result(
            method="GCIQA (4-bit)",
            n_ligands=n_ligands,
            coord_rmsd=0.0,  # No coordinate reconstruction
            dist_error=dist_error,
            time_us=gciqa_time,
            success=dist_error < 0.15625,  # Within theoretical limit
        ))

        # Method 2: Distance geometry (reconstruct from true distances)
        t0 = time.time()
        reconstructed, dg_error = reconstruct_from_distances(true_dists, n_trials=50)
        dg_time = (time.time() - t0) * 1e6

        if reconstructed is not None:
            rmsd = coord_rmsd(true_ligands, reconstructed)
            results.append(Result(
                method="Dist. Geometry",
                n_ligands=n_ligands,
                coord_rmsd=rmsd,
                dist_error=dg_error,
                time_us=dg_time,
                success=rmsd < 0.5,
            ))

        # Method 3: Distance geometry from quantized distances
        t0 = time.time()
        reconstructed_q, dg_q_error = reconstruct_from_distances(quantized, n_trials=50)
        dg_q_time = (time.time() - t0) * 1e6

        if reconstructed_q is not None:
            rmsd_q = coord_rmsd(true_ligands, reconstructed_q)
            results.append(Result(
                method="DG (quantized)",
                n_ligands=n_ligands,
                coord_rmsd=rmsd_q,
                dist_error=dg_q_error,
                time_us=dg_q_time,
                success=rmsd_q < 0.5,
            ))

    return results


def print_results(results: list[Result]) -> None:
    """Print comparison results."""
    # Group by method and n_ligands
    by_key = {}
    for r in results:
        key = (r.method, r.n_ligands)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(r)

    print("\n" + "=" * 90)
    print("METAL SITE RECONSTRUCTION COMPARISON")
    print("=" * 90)

    # Print by ligand count
    for n_lig in sorted(set(r.n_ligands for r in results)):
        print(f"\n--- {n_lig} Ligands ---")
        print(f"{'Method':<20} {'Coord RMSD':>12} {'Dist Error':>12} {'Time (us)':>12} {'Success%':>10}")
        print("-" * 70)

        for method in ["GCIQA (4-bit)", "Dist. Geometry", "DG (quantized)"]:
            key = (method, n_lig)
            if key not in by_key:
                continue

            rs = by_key[key]
            n = len(rs)

            rmsds = [r.coord_rmsd for r in rs if r.coord_rmsd > 0]
            dist_errors = [r.dist_error for r in rs]
            times = [r.time_us for r in rs]
            successes = sum(1 for r in rs if r.success)

            mean_rmsd = np.mean(rmsds) if rmsds else 0
            mean_dist = np.mean(dist_errors)
            mean_time = np.mean(times)

            rmsd_str = f"{mean_rmsd:.4f}" if rmsds else "N/A"
            print(f"{method:<20} {rmsd_str:>12} {mean_dist:>12.4f} {mean_time:>12.1f} {100*successes/n:>9.1f}%")


def main():
    np.random.seed(42)

    print("GCIQA vs Distance Geometry: Metal Site Reconstruction")
    print("=" * 90)

    all_results = []

    for n_lig in [3, 4, 5, 6]:
        print(f"\nTesting {n_lig}-coordinate sites...")
        results = run_comparison(n_lig, n_trials=30)
        all_results.extend(results)
        print_results(all_results)

    # Final summary
    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # GCIQA stats
    gciqa = [r for r in all_results if r.method == "GCIQA (4-bit)"]
    dg = [r for r in all_results if r.method == "Dist. Geometry"]
    dg_q = [r for r in all_results if r.method == "DG (quantized)"]

    print("\nGCIQA (4-bit quantization):")
    print(f"  Distance error: {np.mean([r.dist_error for r in gciqa]):.4f} A")
    print(f"  Time: {np.mean([r.time_us for r in gciqa]):.1f} us")
    print(f"  Success rate: {100*sum(1 for r in gciqa if r.success)/len(gciqa):.1f}%")

    print("\nDistance Geometry (from true distances):")
    print(f"  Coord RMSD: {np.mean([r.coord_rmsd for r in dg]):.4f} A")
    print(f"  Time: {np.mean([r.time_us for r in dg]):.1f} us")
    print(f"  Success rate: {100*sum(1 for r in dg if r.success)/len(dg):.1f}%")

    print("\nDistance Geometry (from quantized distances):")
    print(f"  Coord RMSD: {np.mean([r.coord_rmsd for r in dg_q]):.4f} A")
    print(f"  Time: {np.mean([r.time_us for r in dg_q]):.1f} us")
    print(f"  Success rate: {100*sum(1 for r in dg_q if r.success)/len(dg_q):.1f}%")

    print(f"\nSpeedup: {np.mean([r.time_us for r in dg]) / np.mean([r.time_us for r in gciqa]):.0f}x")


if __name__ == "__main__":
    main()
