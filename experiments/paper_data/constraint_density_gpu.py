"""Constraint Density Test on GPU: 3-dist vs 6-dist at 4-bit.

Uses PyTorch CUDA for parallel search.

Usage:
    F:/PyQQQ/.venv_gpu/Scripts/python.exe experiments/constraint_density_gpu.py
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

METAL_ELEMENTS = {'ZN', 'CU', 'FE', 'MN', 'NI', 'CO', 'CA', 'MG', 'MO', 'W',
                  'U', 'CD', 'HG', 'PB', 'V', 'CR'}
COORD_ELEMENTS = {'N', 'O', 'S', 'SE'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


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


def compute_pairwise(coords):
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
    """GPU search with chunking for large spaces."""
    n = len(expected_dists)
    N = 2 ** (n * bits)
    step = (drange[1] - drange[0]) / (2 ** bits)
    expected = torch.tensor(expected_dists, dtype=torch.float32, device=DEVICE)

    max_chunk = 2 ** 22  # 4M per chunk
    best_err = float('inf')
    best_dists = None
    n_valid = 0
    t0 = time.time()

    for start in range(0, N, max_chunk):
        end = min(start + max_chunk, N)
        indices = torch.arange(start, end, dtype=torch.int64, device=DEVICE)
        dists = torch.zeros(end - start, n, dtype=torch.float32, device=DEVICE)
        for d in range(n):
            shift = d * bits
            mask = (1 << bits) - 1
            vals = (indices >> shift) & mask
            dists[:, d] = drange[0] + (vals.float() + 0.5) * step

        errs = torch.abs(dists - expected).mean(dim=1)
        valid = (torch.abs(dists - expected) <= tolerance).all(dim=1)
        vi = torch.where(valid)[0]
        n_valid += len(vi)

        if len(vi) > 0:
            ve = errs[vi]
            bi = vi[ve.argmin()]
            if errs[bi] < best_err:
                best_err = errs[bi].item()
                best_dists = dists[bi].cpu().tolist()
        else:
            bi = errs.argmin()
            if errs[bi] < best_err:
                best_err = errs[bi].item()
                best_dists = dists[bi].cpu().tolist()

    return best_err, n_valid, N, best_dists, time.time() - t0


def main():
    print("=" * 60)
    print("Constraint Density Test (GPU): 3-dist vs 6-dist at 4-bit")
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

        # 3 distances
        err_3, nv_3, nt_3, bd_3, t_3 = search_gpu(all_dists[:3], tol, bits)
        results.append({
            'pdb': site['pdb'], 'metal': site['metal'], 'config': '3dist',
            'n_dists': 3, 'bits': bits,
            'error': err_3, 'n_valid': nv_3, 'n_total': nt_3, 'time': t_3,
        })

        # 6 distances
        err_6, nv_6, nt_6, bd_6, t_6 = search_gpu(all_dists[:6], tol, bits)
        results.append({
            'pdb': site['pdb'], 'metal': site['metal'], 'config': '6dist',
            'n_dists': 6, 'bits': bits,
            'error': err_6, 'n_valid': nv_6, 'n_total': nt_6, 'time': t_6,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_sites)} done...")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save
    with open('experiments/constraint_density_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS (4-bit, tol=0.5)")
    print(f"{'='*60}")

    by_config = defaultdict(lambda: {'err': [], 'nv': [], 't': []})
    for r in results:
        c = r['config']
        by_config[c]['err'].append(r['error'])
        by_config[c]['nv'].append(r['n_valid'])
        by_config[c]['t'].append(r['time'])

    print(f"\n{'Config':<8} {'N':>4} {'Mean Err':>10} {'Std':>8} {'Mean Valid':>11} {'Mean Time':>11}")
    print("-" * 55)
    for c in ['3dist', '6dist']:
        d = by_config[c]
        n = len(d['err'])
        m = sum(d['err']) / n
        s = math.sqrt(sum((e - m)**2 for e in d['err']) / n)
        v = sum(d['nv']) / n
        t = sum(d['t']) / n
        print(f"{c:<8} {n:>4} {m:>10.3f} {s:>8.3f} {v:>11.1f} {t:>11.3f}")

    # Per-site
    print("\nPer-site (first 15):")
    print(f"{'PDB':<12} {'Metal':<5} {'3dist_nv':>9} {'6dist_nv':>9} {'3dist_err':>10} {'6dist_err':>10}")
    print("-" * 58)

    site_map = defaultdict(dict)
    for r in results:
        site_map[(r['pdb'], r['metal'])][r['config']] = r

    for i, (k, v) in enumerate(sorted(site_map.items())):
        if i >= 15:
            break
        pdb, metal = k
        nv3 = v.get('3dist', {}).get('n_valid', 'N/A')
        nv6 = v.get('6dist', {}).get('n_valid', 'N/A')
        e3 = v.get('3dist', {}).get('error', 0)
        e6 = v.get('6dist', {}).get('error', 0)
        print(f"{pdb:<12} {metal:<5} {nv3:>9} {nv6:>9} {e3:>10.3f} {e6:>10.3f}")


if __name__ == "__main__":
    main()
