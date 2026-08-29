"""Direct computation interface: enumerate valid grid point combinations.

Same interface as search_gpu, but uses direct computation instead of brute-force.
Complexity: O(k^n) where k = valid grid points per distance, n = number of distances.
vs brute-force: O(2^(n*bits))

Usage:
    python direct_compute.py
"""

import os
import sys
import time
import math
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def search_direct(expected_dists, tolerance, bits, drange=(0.0, 5.0)):
    """Direct computation: enumerate valid grid point combinations.

    Same interface as search_gpu for comparison.

    Args:
        expected_dists: List of true distances.
        tolerance: ±tolerance in Å.
        bits: Bit depth for encoding.
        drange: Distance range (min, max) in Å.

    Returns:
        (best_err, n_valid, n_total, best_dists, time_s, all_confs)
        all_confs: list of (dists, error) for all valid combinations, sorted by error.
    """
    n = len(expected_dists)
    d_min, d_max = drange
    step = (d_max - d_min) / (2 ** bits)
    N = 2 ** (n * bits)  # Total search space (for compatibility)

    t0 = time.time()

    # Step 1: Find valid grid points for each distance
    valid_per_dist = []
    for d_true in expected_dists:
        valid = []
        for k in range(2 ** bits):
            d = d_min + (k + 0.5) * step
            if abs(d - d_true) <= tolerance:
                valid.append(d)
        valid_per_dist.append(valid)

    # If any distance has no valid grid points, no solution exists
    if any(len(v) == 0 for v in valid_per_dist):
        return float('inf'), 0, N, None, time.time() - t0, []

    # Step 2: Enumerate all combinations
    best_err = float('inf')
    best_dists = None
    n_valid = 0
    all_confs = []

    for combo in itertools.product(*valid_per_dist):
        err = sum(abs(d - e) for d, e in zip(combo, expected_dists)) / n
        n_valid += 1
        all_confs.append((list(combo), err))
        if err < best_err:
            best_err = err
            best_dists = list(combo)

    # Sort by error
    all_confs.sort(key=lambda x: x[1])

    elapsed = time.time() - t0
    return best_err, n_valid, N, best_dists, elapsed, all_confs


def main():
    """Compare direct computation vs brute-force search."""
    from constraint_density_gpu import parse_pdb_metal_sites, compute_pairwise, search_gpu

    print("=" * 60)
    print("Direct Computation vs Brute-Force Search")
    print("=" * 60)

    pdb_dirs = ['experiments/zn_metalloproteinase/data',
                'experiments/zn_metalloproteinase', 'data']
    pdb_files = []
    for d in pdb_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.pdb'):
                    pdb_files.append(os.path.join(d, f))

    all_sites = []
    for p in pdb_files:
        all_sites.extend(parse_pdb_metal_sites(p))

    print(f"Found {len(all_sites)} sites")

    bits = 4
    tol = 0.5

    results = []
    t0 = time.time()

    for i, site in enumerate(all_sites):
        coords = [site['metal_coord']] + [l['coord'] for l in site['ligands'][:3]]
        all_dists = compute_pairwise(coords)

        # Direct computation: 3 distances
        err_d3, nv_d3, nt_d3, bd_d3, t_d3, confs_d3 = search_direct(all_dists[:3], tol, bits)
        results.append({
            'pdb': site['pdb'], 'metal': site['metal'], 'config': '3dist_direct',
            'n_dists': 3, 'bits': bits,
            'error': err_d3, 'n_valid': nv_d3, 'n_total': nt_d3, 'time': t_d3,
            'confs': confs_d3, 'true_dists': all_dists[:3],
        })

        # Brute-force: 3 distances
        err_g3, nv_g3, nt_g3, bd_g3, t_g3 = search_gpu(all_dists[:3], tol, bits)
        results.append({
            'pdb': site['pdb'], 'metal': site['metal'], 'config': '3dist_gpu',
            'n_dists': 3, 'bits': bits,
            'error': err_g3, 'n_valid': nv_g3, 'n_total': nt_g3, 'time': t_g3,
        })

        # Direct computation: 6 distances
        err_d6, nv_d6, nt_d6, bd_d6, t_d6, confs_d6 = search_direct(all_dists[:6], tol, bits)
        results.append({
            'pdb': site['pdb'], 'metal': site['metal'], 'config': '6dist_direct',
            'n_dists': 6, 'bits': bits,
            'error': err_d6, 'n_valid': nv_d6, 'n_total': nt_d6, 'time': t_d6,
            'confs': confs_d6, 'true_dists': all_dists[:6],
        })

        # Brute-force: 6 distances
        err_g6, nv_g6, nt_g6, bd_g6, t_g6 = search_gpu(all_dists[:6], tol, bits)
        results.append({
            'pdb': site['pdb'], 'metal': site['metal'], 'config': '6dist_gpu',
            'n_dists': 6, 'bits': bits,
            'error': err_g6, 'n_valid': nv_g6, 'n_total': nt_g6, 'time': t_g6,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_sites)} done...")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS (bits={bits}, tol={tol})")
    print(f"{'='*60}")

    from collections import defaultdict
    by_config = defaultdict(lambda: {'err': [], 'nv': [], 't': []})
    for r in results:
        c = r['config']
        by_config[c]['err'].append(r['error'])
        by_config[c]['nv'].append(r['n_valid'])
        by_config[c]['t'].append(r['time'])

    print(f"\n{'Config':<15} {'N':>4} {'Mean Err':>10} {'Std':>8} {'Mean Valid':>11} {'Mean Time':>11}")
    print("-" * 60)
    for c in ['3dist_direct', '3dist_gpu', '6dist_direct', '6dist_gpu']:
        d = by_config[c]
        n = len(d['err'])
        m = sum(d['err']) / n
        s = math.sqrt(sum((e - m)**2 for e in d['err']) / n)
        v = sum(d['nv']) / n
        t = sum(d['t']) / n
        print(f"{c:<15} {n:>4} {m:>10.3f} {s:>8.3f} {v:>11.1f} {t:>11.6f}")

    # Verify: direct and GPU should give same results
    print(f"\nVerification (direct vs GPU should match):")
    mismatches = 0
    for i in range(0, len(results), 4):
        d3 = results[i]    # 3dist_direct
        g3 = results[i+1]  # 3dist_gpu
        d6 = results[i+2]  # 6dist_direct
        g6 = results[i+3]  # 6dist_gpu

        if d3['n_valid'] != g3['n_valid'] or d6['n_valid'] != g6['n_valid']:
            mismatches += 1
            print(f"  MISMATCH: {d3['pdb']} {d3['metal']}: "
                  f"3dist direct={d3['n_valid']} gpu={g3['n_valid']}, "
                  f"6dist direct={d6['n_valid']} gpu={g6['n_valid']}")

    if mismatches == 0:
        print(f"  All {len(results)//4} sites match!")
    else:
        print(f"  {mismatches} mismatches found")

    # Show 4FZP uranium site conformations
    print(f"\n{'='*60}")
    print("4FZP Uranium Site: All Valid Conformations (3-distance)")
    print(f"{'='*60}")

    for r in results:
        if r['pdb'] == '4FZP.pdb' and r['metal'] == 'U' and r['config'] == '3dist_direct':
            true_dists = r['true_dists']
            confs = r['confs']
            print(f"\nTrue distances: {[f'{d:.3f}' for d in true_dists]}")
            print(f"Valid conformations: {len(confs)}")
            print(f"\n{'#':>3} {'U-O1':>8} {'U-O2':>8} {'U-O3':>8} {'Error':>8}")
            print("-" * 40)
            for i, (dists, err) in enumerate(confs):
                print(f"{i+1:3d} {dists[0]:8.2f} {dists[1]:8.2f} {dists[2]:8.2f} {err:8.3f}")
            break


if __name__ == "__main__":
    main()
