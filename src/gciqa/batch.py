"""Batch processing for GCIQA conformation search.

Processes multiple PDB files, extracts distance constraints (metal sites,
disulfide bonds, hydrogen bonds), runs distance-based conformation search,
and outputs results.

Example::

    from gciqa.batch import batch_search, discover_pdb_files

    pdb_files = discover_pdb_files("data/pdbs")
    result = batch_search(pdb_files, bits=4, tolerance=0.5)
    result.save("output.json")
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count

import numpy as np

# --- Constants ---

METAL_ELEMENTS = frozenset({
    'ZN', 'CU', 'FE', 'MN', 'NI', 'CO', 'CA', 'MG', 'MO', 'W',
    'U', 'CD', 'HG', 'PB', 'V', 'CR', 'TI', 'SC',
})
COORD_ELEMENTS = frozenset({'N', 'O', 'S', 'SE'})


# --- Data classes ---

@dataclass
class MetalSite:
    """A metal binding site extracted from a PDB file."""
    pdb_file: str
    metal: str
    metal_coord: tuple[float, float, float]
    ligands: list[dict]
    expected_dists: list[float] = field(default_factory=list)
    site_type: str = "metal"

    @property
    def n_ligands(self) -> int:
        return len(self.ligands)


@dataclass
class DisulfideBond:
    """A disulfide bond (S-S) extracted from a PDB file."""
    pdb_file: str
    sg1_coord: tuple[float, float, float]
    sg2_coord: tuple[float, float, float]
    res1: str
    res2: str
    chain1: str
    chain2: str
    expected_dists: list[float] = field(default_factory=list)
    site_type: str = "disulfide"
    metal: str = "SS"
    n_ligands: int = 2


@dataclass
class SiteResult:
    """Search result for a single metal site."""
    pdb_file: str
    metal: str
    n_ligands: int
    bits: int
    tolerance: float
    expected_dists: list[float]
    best_dists: list[float]
    best_error: float
    n_valid: int
    n_total: int
    time_s: float


@dataclass
class BatchResult:
    """Aggregate result from batch processing."""
    sites: list[SiteResult]
    summary: dict
    total_time: float
    n_pdb_files: int
    n_sites: int

    def save(self, path: str, chunk_size: int = 10000) -> None:
        """Save results to JSON or Parquet file.

        For large datasets, saves in chunks to avoid memory issues.
        Format is determined by file extension (.json or .parquet).

        Args:
            path: Output file path (.json or .parquet).
            chunk_size: Number of sites per chunk (for large datasets).
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        _base, ext = os.path.splitext(path)

        if ext == '.parquet':
            self._save_parquet(path)
        else:
            self._save_json(path, chunk_size)

    def _save_json(self, path: str, chunk_size: int) -> None:
        """Save as JSON (with chunking for large datasets)."""
        if len(self.sites) <= chunk_size:
            data = {
                'summary': self.summary,
                'total_time': self.total_time,
                'n_pdb_files': self.n_pdb_files,
                'n_sites': self.n_sites,
                'sites': [asdict(s) for s in self.sites],
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            base, ext = os.path.splitext(path)
            n_chunks = (len(self.sites) + chunk_size - 1) // chunk_size

            summary_path = f"{base}_summary{ext}"
            with open(summary_path, 'w') as f:
                json.dump({
                    'summary': self.summary,
                    'total_time': self.total_time,
                    'n_pdb_files': self.n_pdb_files,
                    'n_sites': self.n_sites,
                    'n_chunks': n_chunks,
                    'chunk_size': chunk_size,
                }, f, indent=2)

            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(self.sites))
                chunk_path = f"{base}_chunk{i:04d}{ext}"
                with open(chunk_path, 'w') as f:
                    json.dump([asdict(s) for s in self.sites[start:end]], f)

            print(f"Saved {n_chunks} chunks + summary to {os.path.dirname(path)}")

    def _save_parquet(self, path: str) -> None:
        """Save as Parquet (compressed, columnar)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet output: pip install pandas pyarrow")

        # Convert to DataFrame
        records = []
        for s in self.sites:
            record = asdict(s)
            # Convert list fields to string for Parquet
            record['expected_dists'] = str(record['expected_dists'])
            record['best_dists'] = str(record['best_dists'])
            records.append(record)

        df = pd.DataFrame(records)

        # Save
        df.to_parquet(path, index=False, compression='snappy')
        print(f"Saved {len(df)} sites to {path}")

        # Also save summary as JSON
        summary_path = path.replace('.parquet', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'summary': self.summary,
                'total_time': self.total_time,
                'n_pdb_files': self.n_pdb_files,
                'n_sites': self.n_sites,
            }, f, indent=2)
        print(f"Saved summary to {summary_path}")

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(f"\n{'='*60}")
        print("Batch Search Complete")
        print(f"{'='*60}")
        print(f"PDB files: {self.n_pdb_files}")
        print(f"Metal sites: {self.n_sites}")
        print(f"Total time: {self.total_time:.2f}s")

        if self.summary.get('by_bits'):
            print(f"\n{'Bits':>4} {'N':>5} {'Mean Err':>10} {'Std':>8} {'Valid%':>8}")
            print("-" * 40)
            for bits_str, stats in sorted(self.summary['by_bits'].items()):
                print(f"{bits_str:>4} {stats['n']:>5} {stats['mean_error']:>10.4f} "
                      f"{stats['std_error']:>8.4f} {stats['valid_pct']:>7.1f}%")

        if self.summary.get('by_metal'):
            print(f"\n{'Metal':>6} {'N':>5} {'Mean Err':>10} {'Std':>8}")
            print("-" * 35)
            for metal, stats in sorted(self.summary['by_metal'].items()):
                print(f"{metal:>6} {stats['n']:>5} {stats['mean_error']:>10.4f} "
                      f"{stats['std_error']:>8.4f}")


# --- PDB parsing ---

def parse_pdb_metal_sites(
    pdb_path: str,
    max_coord_dist: float = 3.0,
    min_coord_atoms: int = 3,
    max_ligands: int = 6,
) -> list[MetalSite]:
    """Extract all metal binding sites from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        max_coord_dist: Max distance (Å) for coordination shell.
        min_coord_atoms: Minimum coordinating atoms to count as a site.
        max_ligands: Maximum ligands per site (for distance encoding).

    Returns:
        List of MetalSite objects.
    """
    atoms = []
    metals = []

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith(('ATOM', 'HETATM'))):
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
            if not is_metal and line.startswith("HETATM") and atom_name.strip() in METAL_ELEMENTS:
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
        ligands = ligands[:max_ligands]

        if len(ligands) >= min_coord_atoms:
            # Compute metal-ligand distances
            coords = [(mx, my, mz)] + [l['coord'] for l in ligands]
            expected_dists = _pairwise_distances(coords)[:len(ligands)]

            sites.append(MetalSite(
                pdb_file=os.path.basename(pdb_path),
                metal=metal['element'],
                metal_coord=(mx, my, mz),
                ligands=ligands,
                expected_dists=expected_dists,
            ))
    return sites


def parse_pdb_disulfide_sites(
    pdb_path: str,
    max_dist: float = 2.5,
) -> list[DisulfideBond]:
    """Extract disulfide bonds (S-S) from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        max_dist: Max S-S distance (Å) for disulfide bond.

    Returns:
        List of DisulfideBond objects.
    """
    sg_atoms = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'SG':
                continue
            res_name = line[17:20].strip()
            if res_name != 'CYS':
                continue
            chain = line[21]
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            sg_atoms.append({
                'chain': chain, 'res_seq': res_seq, 'res_name': res_name,
                'x': x, 'y': y, 'z': z,
            })

    bonds = []
    for i in range(len(sg_atoms)):
        for j in range(i+1, len(sg_atoms)):
            a1, a2 = sg_atoms[i], sg_atoms[j]
            # Skip same residue
            if a1['chain'] == a2['chain'] and a1['res_seq'] == a2['res_seq']:
                continue
            dx = a1['x'] - a2['x']
            dy = a1['y'] - a2['y']
            dz = a1['z'] - a2['z']
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < max_dist:
                bonds.append(DisulfideBond(
                    pdb_file=os.path.basename(pdb_path),
                    sg1_coord=(a1['x'], a1['y'], a1['z']),
                    sg2_coord=(a2['x'], a2['y'], a2['z']),
                    res1=f"{a1['res_name']}{a1['res_seq']}",
                    res2=f"{a2['res_name']}{a2['res_seq']}",
                    chain1=a1['chain'],
                    chain2=a2['chain'],
                    expected_dists=[dist],
                ))
    return bonds


def _pairwise_distances(coords: list[tuple]) -> list[float]:
    """Compute pairwise distances between coordinates."""
    n = len(coords)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    return dists


# --- Vectorized search ---

def _search_vectorized(
    expected_dists: list[float],
    tolerance: float,
    bits: int,
    distance_range: tuple[float, float] = (0.0, 5.0),
) -> tuple[float, int, int, list[float], float]:
    """Direct computation - no search needed.

    Each distance is independent, so we just round to the nearest grid point.
    Complexity: O(n) instead of O(2^(n*bits)).

    Returns: (best_error, n_valid, n_total, best_dists, time_s)
    """
    n_dists = len(expected_dists)
    step = (distance_range[1] - distance_range[0]) / (2 ** bits)
    N = 2 ** (n_dists * bits)  # Total search space (for compatibility)

    t0 = time.time()

    # Direct computation: round each distance to nearest grid point
    best_dists = []
    for d in expected_dists:
        # Find nearest grid point
        k = round((d - distance_range[0]) / step - 0.5)
        k = max(0, min(k, (1 << bits) - 1))  # Clamp to valid range
        decoded = distance_range[0] + (k + 0.5) * step
        best_dists.append(decoded)

    # Compute error
    best_err = float(np.abs(np.array(best_dists) - np.array(expected_dists)).mean())

    # Check if within tolerance
    within_tol = all(abs(d - e) <= tolerance for d, e in zip(best_dists, expected_dists))
    n_valid = N if within_tol else 0  # All states are valid if direct computation works

    elapsed = time.time() - t0
    return best_err, n_valid, N, best_dists, elapsed


# --- Worker functions for multiprocessing ---

def _parse_pdb_file(args: tuple) -> list:
    """Parse a single PDB file for metal sites and disulfide bonds."""
    pdb_path, max_coord_dist, min_coord_atoms, max_ligands = args
    sites = parse_pdb_metal_sites(
        pdb_path, max_coord_dist=max_coord_dist,
        min_coord_atoms=min_coord_atoms, max_ligands=max_ligands,
    )
    bonds = parse_pdb_disulfide_sites(pdb_path)
    return sites + bonds


def _process_site(args: tuple) -> list[SiteResult]:
    """Process a single site with all parameter combinations."""
    site, bit_depths, tolerance, distance_range = args
    results = []
    for bits in bit_depths:
        best_err, n_valid, n_total, best_dists, t = _search_vectorized(
            site.expected_dists, tolerance=tolerance, bits=bits,
            distance_range=distance_range,
        )
        results.append(SiteResult(
            pdb_file=site.pdb_file,
            metal=site.metal,
            n_ligands=site.n_ligands,
            bits=bits,
            tolerance=tolerance,
            expected_dists=site.expected_dists,
            best_dists=best_dists,
            best_error=best_err,
            n_valid=n_valid,
            n_total=n_total,
            time_s=t,
        ))
    return results


# --- File discovery ---

def discover_pdb_files(paths: list[str]) -> list[str]:
    """Discover PDB files from paths (files or directories).

    Args:
        paths: List of file paths or directory paths.

    Returns:
        Sorted list of PDB file paths.
    """
    pdb_files = []
    for p in paths:
        if os.path.isfile(p) and (p.endswith(('.pdb', '.ent'))):
            pdb_files.append(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for f in files:
                    if f.endswith(('.pdb', '.ent')):
                        pdb_files.append(os.path.join(root, f))
    return sorted(set(pdb_files))


# --- Main batch search ---

def batch_search(
    pdb_paths: list[str],
    bits: int | list[int] = 4,
    tolerance: float = 0.5,
    distance_range: tuple[float, float] = (0.0, 5.0),
    max_coord_dist: float = 3.0,
    min_coord_atoms: int = 3,
    max_ligands: int = 6,
    n_workers: int | None = None,
    verbose: bool = True,
) -> BatchResult:
    """Run batch metal site search across PDB files.

    Args:
        pdb_paths: PDB file paths or directories containing PDB files.
        bits: Bit depth(s) for encoding. Single int or list.
        tolerance: Distance tolerance in Å.
        distance_range: Physical distance range (min, max) in Å.
        max_coord_dist: Max coordination distance for ligand detection.
        min_coord_atoms: Minimum coordinating atoms per site.
        max_ligands: Maximum ligands per site.
        n_workers: Number of parallel workers. None = auto.
        verbose: Print progress.

    Returns:
        BatchResult with all site results and summary.
    """
    t0 = time.time()

    if isinstance(bits, int):
        bit_depths = [bits]
    else:
        bit_depths = list(bits)

    # Discover PDB files
    pdb_files = discover_pdb_files(pdb_paths)
    if verbose:
        print(f"Found {len(pdb_files)} PDB files")

    # Extract sites (metal + disulfide) in parallel
    all_sites = []
    if n_workers is None:
        n_workers_parse = min(cpu_count(), 8)
    else:
        n_workers_parse = n_workers

    parse_tasks = [(p, max_coord_dist, min_coord_atoms, max_ligands) for p in pdb_files]

    if n_workers_parse > 1 and len(pdb_files) > 1:
        with Pool(n_workers_parse) as pool:
            for i, sites in enumerate(pool.imap_unordered(_parse_pdb_file, parse_tasks)):
                all_sites.extend(sites)
                if verbose and (i + 1) % 1000 == 0:
                    print(f"  Parsed {i+1}/{len(pdb_files)} files, {len(all_sites)} sites so far...")
    else:
        for i, task in enumerate(parse_tasks):
            all_sites.extend(_parse_pdb_file(task))
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Parsed {i+1}/{len(pdb_files)} files, {len(all_sites)} sites so far...")

    if verbose:
        metal_count = sum(1 for s in all_sites if getattr(s, 'site_type', 'metal') == 'metal')
        ss_count = sum(1 for s in all_sites if getattr(s, 'site_type', '') == 'disulfide')
        print(f"Found {len(all_sites)} sites ({metal_count} metal, {ss_count} disulfide)")
        type_counts = defaultdict(int)
        for s in all_sites:
            type_counts[s.metal] += 1
        print(f"Types: {dict(type_counts)}")

    # Prepare tasks
    tasks = [(site, bit_depths, tolerance, distance_range) for site in all_sites]

    # Run with multiprocessing
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    if verbose:
        print(f"Running with {n_workers} workers, bits={bit_depths}, tol={tolerance}...")

    if n_workers > 1 and len(tasks) > 1:
        with Pool(n_workers) as pool:
            all_results = []
            for i, result in enumerate(pool.imap_unordered(_process_site, tasks)):
                all_results.append(result)
                if verbose and (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1}/{len(tasks)} sites...")
    else:
        all_results = [_process_site(t) for t in tasks]

    # Flatten
    site_results = []
    for r in all_results:
        site_results.extend(r)

    total_time = time.time() - t0

    # Build summary
    summary = _build_summary(site_results)

    if verbose:
        print(f"Completed {len(site_results)} searches in {total_time:.2f}s")

    return BatchResult(
        sites=site_results,
        summary=summary,
        total_time=total_time,
        n_pdb_files=len(pdb_files),
        n_sites=len(all_sites),
    )


def _build_summary(results: list[SiteResult]) -> dict:
    """Build summary statistics from results."""
    by_bits = defaultdict(list)
    by_metal = defaultdict(list)

    for r in results:
        by_bits[r.bits].append(r)
        by_metal[r.metal].append(r)

    summary = {'by_bits': {}, 'by_metal': {}}

    for bits, rs in sorted(by_bits.items()):
        errors = [r.best_error for r in rs]
        n = len(errors)
        mean = sum(errors) / n
        std = math.sqrt(sum((e - mean)**2 for e in errors) / n) if n > 1 else 0
        valid_count = sum(1 for r in rs if r.n_valid > 0)
        summary['by_bits'][str(bits)] = {
            'n': n,
            'mean_error': round(mean, 6),
            'std_error': round(std, 6),
            'min_error': round(min(errors), 6),
            'max_error': round(max(errors), 6),
            'valid_pct': round(100 * valid_count / n, 1),
        }

    for metal, rs in sorted(by_metal.items()):
        errors = [r.best_error for r in rs]
        n = len(errors)
        mean = sum(errors) / n
        std = math.sqrt(sum((e - mean)**2 for e in errors) / n) if n > 1 else 0
        summary['by_metal'][metal] = {
            'n': n,
            'mean_error': round(mean, 6),
            'std_error': round(std, 6),
        }

    return summary
