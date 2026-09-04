"""Fast Error Modeling: Vectorized + Multiprocessing.

Optimizations:
1. NumPy vectorized bitstring decoding (no Python loop per state)
2. Multiprocessing across sites
3. Batch processing

Usage:
    python experiments/error_modeling_fast.py
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))



METAL_ELEMENTS = {'ZN', 'CU', 'FE', 'MN', 'NI', 'CO', 'CA', 'MG', 'MO', 'W',
                  'U', 'CD', 'HG', 'PB', 'V', 'CR'}
COORD_ELEMENTS = {'N', 'O', 'S', 'SE'}


def parse_pdb_metal_sites(pdb_path, max_coord_dist=3.0, min_coord_atoms=3):
    """Extract all metal binding sites from a PDB file."""
    atoms = []
    metals = []

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else ''

            record = {
                'atom_name': atom_name, 'res_name': res_name,
                'chain': chain, 'res_seq': res_seq,
                'x': x, 'y': y, 'z': z, 'element': element,
            }

            is_metal = element in METAL_ELEMENTS
            if not is_metal and line.startswith("HETATM"):
                if atom_name.strip() in METAL_ELEMENTS:
                    is_metal = True
            if is_metal:
                metals.append(record)
            elif element in COORD_ELEMENTS:
                atoms.append(record)

    sites = []
    for metal in metals:
        mx, my, mz = metal['x'], metal['y'], metal['z']
        ligands = []
        for atom in atoms:
            dx = atom['x'] - mx
            dy = atom['y'] - my
            dz = atom['z'] - mz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < max_coord_dist and dist > 0.1:
                ligands.append({
                    'atom_name': atom['atom_name'],
                    'res_name': atom['res_name'],
                    'coord': (atom['x'], atom['y'], atom['z']),
                    'distance': dist,
                    'element': atom['element'],
                })
        ligands.sort(key=lambda x: x['distance'])
        if len(ligands) >= min_coord_atoms:
            sites.append({
                'metal': metal['element'],
                'metal_coord': (mx, my, mz),
                'ligands': ligands,
                'pdb': os.path.basename(pdb_path),
            })
    return sites


def compute_pairwise_distances(coords):
    """Compute all pairwise distances between coordinates."""
    n = len(coords)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    return dists


def decode_all_bitstrings_vectorized(n_dists, bits, distance_range=(0.0, 5.0)):
    """Pre-compute all decoded distances for all possible bitstrings.

    Returns: numpy array of shape (2^(n_dists*bits), n_dists)
    """
    total_bits = n_dists * bits
    N = 2 ** total_bits
    step = (distance_range[1] - distance_range[0]) / (2 ** bits)

    # Generate all bitstrings as integers
    indices = np.arange(N, dtype=np.int64)

    # Extract each distance group
    dists = np.zeros((N, n_dists), dtype=np.float64)
    for d in range(n_dists):
        # Extract bits for this distance
        shift = d * bits
        mask = (1 << bits) - 1
        vals = (indices >> shift) & mask
        dists[:, d] = distance_range[0] + (vals + 0.5) * step

    return dists


def run_search_vectorized(expected_dists, tolerance, bits, distance_range=(0.0, 5.0)):
    """Vectorized search using NumPy.

    Returns: (best_error, n_valid, n_total, best_dists, time_s)
    """
    n_dists = len(expected_dists)
    t0 = time.time()

    # Decode all bitstrings at once
    all_dists = decode_all_bitstrings_vectorized(n_dists, bits, distance_range)
    N = all_dists.shape[0]

    # Check constraints vectorized
    expected = np.array(expected_dists)
    errors = np.abs(all_dists - expected).mean(axis=1)

    # Valid = all distances within tolerance
    within_tol = np.abs(all_dists - expected) <= tolerance
    valid_mask = within_tol.all(axis=1)

    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    elapsed = time.time() - t0

    if n_valid > 0:
        valid_errors = errors[valid_indices]
        best_idx = valid_indices[valid_errors.argmin()]
        return errors[best_idx], n_valid, N, all_dists[best_idx].tolist(), elapsed
    else:
        best_idx = errors.argmin()
        return errors[best_idx], 0, N, all_dists[best_idx].tolist(), elapsed


def process_site(args):
    """Process a single site with all parameter combinations."""
    site, bit_depths, tolerances = args
    metal = site['metal']
    pdb = site['pdb']
    n_ligands = len(site['ligands'])

    coords = [site['metal_coord']] + [l['coord'] for l in site['ligands'][:3]]
    expected_dists = compute_pairwise_distances(coords)[:3]

    results = []
    for bits in bit_depths:
        for tol in tolerances:
            best_err, n_valid, n_total, best_dists, t = run_search_vectorized(
                expected_dists, tolerance=tol, bits=bits)
            results.append({
                'pdb': pdb, 'metal': metal, 'n_coordinators': n_ligands,
                'bits': bits, 'tolerance': tol,
                'expected_dists': expected_dists,
                'best_dists': best_dists,
                'best_error': best_err,
                'n_valid': n_valid, 'n_total': n_total, 'time_s': t,
            })
    return results


def run_error_modeling_fast():
    """Main pipeline with multiprocessing."""
    print("=" * 70)
    print("GCIQA Error Modeling (Fast: NumPy Vectorized + Multiprocessing)")
    print("=" * 70)

    pdb_dirs = [
        'experiments/zn_metalloproteinase/data',
        'experiments/zn_metalloproteinase',
        'data',
    ]

    pdb_files = []
    for d in pdb_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.pdb'):
                pdb_files.append(os.path.join(d, f))

    print(f"\nFound {len(pdb_files)} PDB files")

    all_sites = []
    for pdb_path in pdb_files:
        sites = parse_pdb_metal_sites(pdb_path, min_coord_atoms=3)
        all_sites.extend(sites)

    print(f"Found {len(all_sites)} metal binding sites")

    metal_counts = defaultdict(int)
    for site in all_sites:
        metal_counts[site['metal']] += 1
    print(f"Metal types: {dict(metal_counts)}")

    bit_depths = [3, 4, 5]
    tolerances = [0.3, 0.5, 1.0]

    # Prepare tasks
    tasks = [(site, bit_depths, tolerances) for site in all_sites]

    # Run with multiprocessing
    n_workers = min(cpu_count(), 8)
    print(f"\nRunning with {n_workers} workers...")

    t0 = time.time()
    with Pool(n_workers) as pool:
        all_results = pool.map(process_site, tasks)
    elapsed = time.time() - t0

    # Flatten results
    results = []
    for site_results in all_results:
        results.extend(site_results)

    print(f"Completed {len(results)} tests in {elapsed:.1f}s")

    # Save
    output_path = 'experiments/error_modeling_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    print_summary(results)
    return results


def print_summary(results):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    by_config = defaultdict(list)
    for r in results:
        key = (r['bits'], r['tolerance'])
        by_config[key].append(r['best_error'])

    print(f"\n{'Bits':>4} {'Tol':>5} {'N':>4} {'Mean Err':>10} {'Std':>8} "
          f"{'Min':>8} {'Max':>8} {'Median':>8}")
    print("-" * 60)

    for (bits, tol), errors in sorted(by_config.items()):
        n = len(errors)
        mean = sum(errors) / n
        std = math.sqrt(sum((e - mean)**2 for e in errors) / n)
        mn = min(errors)
        mx = max(errors)
        med = sorted(errors)[n // 2]
        print(f"{bits:>4} {tol:>5.1f} {n:>4} {mean:>10.3f} {std:>8.3f} "
              f"{mn:>8.3f} {mx:>8.3f} {med:>8.3f}")

    by_metal = defaultdict(list)
    for r in results:
        if r['bits'] == 4 and r['tolerance'] == 0.5:
            by_metal[r['metal']].append(r['best_error'])

    if by_metal:
        print("\nBy metal type (4-bit, tol=0.5):")
        print(f"{'Metal':>6} {'N':>4} {'Mean Err':>10} {'Std':>8}")
        print("-" * 30)
        for metal, errors in sorted(by_metal.items()):
            n = len(errors)
            mean = sum(errors) / n
            std = math.sqrt(sum((e - mean)**2 for e in errors) / n) if n > 1 else 0
            print(f"{metal:>6} {n:>4} {mean:>10.3f} {std:>8.3f}")


if __name__ == "__main__":
    run_error_modeling_fast()
