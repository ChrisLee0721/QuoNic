"""GPU-Accelerated Error Modeling using PyTorch.

Uses PyTorch tensors for parallel bitstring decoding and constraint checking.
Falls back to CPU if CUDA not available.

Usage:
    F:/PyQQQ/.venv_gpu/Scripts/python.exe experiments/error_modeling_gpu.py
"""

import os
import sys
import time
import json
import math
import torch
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


METAL_ELEMENTS = {'ZN', 'CU', 'FE', 'MN', 'NI', 'CO', 'CA', 'MG', 'MO', 'W',
                  'U', 'CD', 'HG', 'PB', 'V', 'CR'}
COORD_ELEMENTS = {'N', 'O', 'S', 'SE'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def parse_pdb_metal_sites(pdb_path, max_coord_dist=3.0, min_coord_atoms=3):
    """Extract metal sites from PDB."""
    atoms = []
    metals = []

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else ''

            record = {'atom_name': atom_name, 'res_name': res_name,
                      'x': x, 'y': y, 'z': z, 'element': element}

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
    """All pairwise distances."""
    n = len(coords)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    return dists


def search_gpu(expected_dists, tolerance, bits, drange=(0.0, 5.0)):
    """GPU-accelerated search using PyTorch."""
    n_dists = len(expected_dists)
    total_bits = n_dists * bits
    N = 2 ** total_bits
    step = (drange[1] - drange[0]) / (2 ** bits)

    t0 = time.time()

    # Process in chunks to fit in VRAM
    max_chunk = 2 ** 22  # 4M states per chunk (4M × 6 × 4 bytes = 96 MB)

    expected = torch.tensor(expected_dists, dtype=torch.float32, device=DEVICE)
    best_err = float('inf')
    best_dists = None
    n_valid = 0

    for start in range(0, N, max_chunk):
        end = min(start + max_chunk, N)
        chunk_size = end - start

        # Generate indices on GPU
        indices = torch.arange(start, end, dtype=torch.int64, device=DEVICE)

        # Decode all distances in parallel
        dists = torch.zeros(chunk_size, n_dists, dtype=torch.float32, device=DEVICE)
        for d in range(n_dists):
            shift = d * bits
            mask = (1 << bits) - 1
            vals = (indices >> shift) & mask
            dists[:, d] = drange[0] + (vals.float() + 0.5) * step

        # Compute errors
        errs = torch.abs(dists - expected).mean(dim=1)

        # Check constraints
        within_tol = (torch.abs(dists - expected) <= tolerance).all(dim=1)
        valid_indices = torch.where(within_tol)[0]
        n_valid += len(valid_indices)

        if len(valid_indices) > 0:
            valid_errs = errs[valid_indices]
            chunk_best = valid_indices[valid_errs.argmin()]
            if errs[chunk_best] < best_err:
                best_err = errs[chunk_best].item()
                best_dists = dists[chunk_best].cpu().tolist()
        else:
            chunk_best = errs.argmin()
            if errs[chunk_best] < best_err:
                best_err = errs[chunk_best].item()
                best_dists = dists[chunk_best].cpu().tolist()

    elapsed = time.time() - t0
    return best_err, n_valid, N, best_dists, elapsed


def process_site(args):
    """Process one site with all parameter combinations."""
    site, bit_depths, tolerances = args
    metal = site['metal']
    pdb = site['pdb']
    n_ligands = len(site['ligands'])

    coords = [site['metal_coord']] + [l['coord'] for l in site['ligands'][:3]]
    expected_dists = compute_pairwise_distances(coords)[:3]

    results = []
    for bits in bit_depths:
        for tol in tolerances:
            err, nv, nt, bd, t = search_gpu(expected_dists, tol, bits)
            results.append({
                'pdb': pdb, 'metal': metal, 'n_coordinators': n_ligands,
                'bits': bits, 'tolerance': tol,
                'expected_dists': expected_dists,
                'best_dists': bd,
                'best_error': err,
                'n_valid': nv, 'n_total': nt, 'time_s': t,
            })
    return results


def main():
    print("=" * 70)
    print("GCIQA Error Modeling (GPU-Accelerated)")
    print("=" * 70)

    pdb_dirs = ['experiments/zn_metalloproteinase/data',
                'experiments/zn_metalloproteinase', 'data']

    pdb_files = []
    for d in pdb_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.pdb'):
                    pdb_files.append(os.path.join(d, f))

    print(f"Found {len(pdb_files)} PDB files")

    all_sites = []
    for p in pdb_files:
        all_sites.extend(parse_pdb_metal_sites(p))

    print(f"Found {len(all_sites)} metal sites")

    metal_counts = defaultdict(int)
    for s in all_sites:
        metal_counts[s['metal']] += 1
    print(f"Metal types: {dict(metal_counts)}")

    bit_depths = [3, 4, 5]
    tolerances = [0.3, 0.5, 1.0]

    # Run sequentially (GPU is already parallel)
    print(f"\nRunning on {DEVICE}...")
    t0 = time.time()

    all_results = []
    for i, site in enumerate(all_sites):
        results = process_site((site, bit_depths, tolerances))
        all_results.extend(results)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_sites)} sites done...")

    elapsed = time.time() - t0
    print(f"Completed {len(all_results)} tests in {elapsed:.1f}s")

    # Save
    with open('experiments/error_modeling_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    by_config = defaultdict(list)
    for r in all_results:
        by_config[(r['bits'], r['tolerance'])].append(r['best_error'])

    print(f"\n{'Bits':>4} {'Tol':>5} {'N':>4} {'Mean Err':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for (bits, tol), errs in sorted(by_config.items()):
        n = len(errs)
        m = sum(errs) / n
        s = math.sqrt(sum((e - m)**2 for e in errs) / n)
        print(f"{bits:>4} {tol:>5.1f} {n:>4} {m:>10.3f} {s:>8.3f} {min(errs):>8.3f} {max(errs):>8.3f}")

    by_metal = defaultdict(list)
    for r in all_results:
        if r['bits'] == 4 and r['tolerance'] == 0.5:
            by_metal[r['metal']].append(r['best_error'])

    if by_metal:
        print(f"\nBy metal (4-bit, tol=0.5):")
        print(f"{'Metal':>6} {'N':>4} {'Mean':>8} {'Std':>8}")
        print("-" * 28)
        for m, e in sorted(by_metal.items()):
            n = len(e)
            mean = sum(e) / n
            std = math.sqrt(sum((x - mean)**2 for x in e) / n) if n > 1 else 0
            print(f"{m:>6} {n:>4} {mean:>8.3f} {std:>8.3f}")


if __name__ == "__main__":
    main()
