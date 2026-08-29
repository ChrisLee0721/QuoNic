"""Large-scale test for metal finding algorithm.

Tests the find_sites function with coordination constraints on multiple PDB files.
Collects statistics to identify failure patterns.
"""

import sys
import math
import time
sys.path.insert(0, 'src')

from gciqa import find_sites, GeometricConstraint, ConstraintSet
from gciqa.pdb import parse_pdb, find_metal_ions
from gciqa.ligand_detect import _COORD_ELEMENTS


def analyze_metal_site(pdb_file: str) -> dict:
    """Analyze a single PDB file for metal finding performance."""
    struct = parse_pdb(pdb_file)
    metals = find_metal_ions(struct)

    if not metals:
        return {'pdb': pdb_file, 'error': 'no metals found'}

    true_pos = metals[0].coord
    metal_elem = metals[0].element

    # Find coordinating atoms
    coord_atoms = []
    for i, (atom, coord) in enumerate(zip(struct.atoms, struct.coords)):
        if atom not in _COORD_ELEMENTS:
            continue
        d = math.sqrt(sum((a-b)**2 for a, b in zip(coord, true_pos)))
        if d <= 2.5:
            coord_atoms.append((i, atom, coord, d))

    if len(coord_atoms) < 4:
        return {'pdb': pdb_file, 'error': f'only {len(coord_atoms)} coordinating atoms'}

    # Distances between coordinating atoms
    inter_dists = []
    for i in range(len(coord_atoms)):
        for j in range(i+1, len(coord_atoms)):
            d = math.sqrt(sum((a-b)**2 for a, b in zip(coord_atoms[i][2], coord_atoms[j][2])))
            inter_dists.append(d)

    # Run find_sites
    constraints = ConstraintSet([
        GeometricConstraint.coordination(metal_elem, n_ligands=4, min_dist=1.8, max_dist=2.5),
    ])

    start = time.time()
    sites = find_sites(constraints, atoms=struct.atoms, coords=struct.coords, n_samples=20)
    elapsed = time.time() - start

    # Calculate RMSD for each site
    rmsds = []
    for s in sites:
        if metal_elem in s.positions:
            pred = s.positions[metal_elem]
            rmsd = math.sqrt(sum((a-b)**2 for a, b in zip(pred, true_pos)))
            rmsds.append(rmsd)

    return {
        'pdb': pdb_file.split('/')[-1],
        'metal': metal_elem,
        'true_pos': true_pos,
        'n_coordinating': len(coord_atoms),
        'max_inter_dist': max(inter_dists) if inter_dists else 0,
        'min_inter_dist': min(inter_dists) if inter_dists else 0,
        'n_sites': len(sites),
        'rmsds': rmsds,
        'best_rmsd': min(rmsds) if rmsds else float('inf'),
        'elapsed': elapsed,
    }


def main():
    pdb_files = [
        'experiments/zn_metalloproteinase/1CA2.pdb',
        'experiments/zn_metalloproteinase/1ZNB.pdb',
        'experiments/zn_metalloproteinase/1ZNF.pdb',
        'experiments/zn_metalloproteinase/2CBA.pdb',
        'experiments/zn_metalloproteinase/4MBN.pdb',
    ]

    print('=== Metal Finding Large-Scale Test ===')
    print(f'Testing {len(pdb_files)} PDB files')
    print()

    results = []
    for pdb_file in pdb_files:
        try:
            result = analyze_metal_site(pdb_file)
            results.append(result)

            if 'error' in result:
                print(f"{result['pdb']}: ERROR - {result['error']}")
            else:
                print(f"{result['pdb']}: {result['metal']} "
                      f"best={result['best_rmsd']:.2f}A, "
                      f"sites={result['n_sites']}, "
                      f"time={result['elapsed']:.1f}s")
        except Exception as e:
            print(f"{pdb_file}: EXCEPTION - {e}")

    # Summary statistics
    print()
    print('=== Summary ===')

    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print('No valid results')
        return

    all_rmsds = []
    for r in valid_results:
        all_rmsds.extend(r['rmsds'])

    if all_rmsds:
        all_rmsds.sort()
        print(f'Total sites: {len(all_rmsds)}')
        print(f'Best RMSD: {all_rmsds[0]:.2f}A')
        print(f'Median RMSD: {all_rmsds[len(all_rmsds)//2]:.2f}A')
        print(f'Worst RMSD: {all_rmsds[-1]:.2f}A')
        print(f'Sites < 2A: {sum(1 for r in all_rmsds if r < 2.0)} ({100*sum(1 for r in all_rmsds if r < 2.0)/len(all_rmsds):.1f}%)')
        print(f'Sites < 5A: {sum(1 for r in all_rmsds if r < 5.0)} ({100*sum(1 for r in all_rmsds if r < 5.0)/len(all_rmsds):.1f}%)')
        print(f'Sites < 10A: {sum(1 for r in all_rmsds if r < 10.0)} ({100*sum(1 for r in all_rmsds if r < 10.0)/len(all_rmsds):.1f}%)')

    # Per-PDB analysis
    print()
    print('=== Per-PDB Analysis ===')
    for r in valid_results:
        if r['rmsds']:
            best = min(r['rmsds'])
            print(f"{r['pdb']}: best={best:.2f}A, "
                  f"coordinating_atoms={r['n_coordinating']}, "
                  f"inter_dist={r['min_inter_dist']:.1f}-{r['max_inter_dist']:.1f}A")


if __name__ == '__main__':
    main()
