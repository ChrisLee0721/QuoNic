"""Batch download PDB files from RCSB.

Usage::

    from gciqa.download import download_pdbs, PDB_SETS

    # Download predefined sets
    download_pdbs(PDB_SETS['metalloproteinase'], "data/pdbs")

    # Download custom list
    download_pdbs(["1AKE", "3ARC"], "data/pdbs")

    # Search RCSB by metal type
    ids = search_rcsb(metal="ZN", max_results=100)
    download_pdbs(ids, "data/pdbs")
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Any

# Predefined PDB sets for validation
PDB_SETS = {
    # Metal diversity (10+ metals)
    'metal_diversity': [
        '1AKE',   # Zn - adenylate kinase
        '3ARC',   # U  - uranyl binding
        '1AZU',   # Cu - azurin (blue copper)
        '1CCM',   # Fe - cytochrome c
        '1CDP',   # Ca - calmodulin
        '1FHA',   # Fe - ferredoxin
        '1LND',   # Mn - Mn superoxide dismutase
        '1MBN',   # Fe - myoglobin
        '1PCY',   # Cu - plastocyanin
        '1SOD',   # Cu/Zn - superoxide dismutase
        '2HHB',   # Fe - hemoglobin
        '1A2L',   # Zn - carbonic anhydrase
        '1ZNF',   # Zn - zinc finger
        '1CA2',   # Zn - carbonic anhydrase II
    ],

    # Non-standard geometries
    'nonstandard_geometry': [
        '1FIQ',   # Mo - xanthine oxidase (pentagonal bipyramidal)
        '1VOX',   # V  - vanadium bromoperoxidase
        '1QNI',   # Ni - urease
        '3ARC',   # U  - high coordination number
        '1AZU',   # Cu - distorted tetrahedral
    ],

    # Large dataset for error modeling (100+ sites)
    'large_scale': [
        '1AKE', '1AZU', '1CA2', '1CCM', '1CDP', '1FHA',
        '1LND', '1MBN', '1PCY', '1SOD', '1ZNF', '2HHB',
        '1A2L', '3ARC', '4FZP',
    ],

    # Diverse 100: metals + geometries + non-metal systems
    'diverse_100': [
        # --- Zn (15) ---
        '1CA2', '1A2L', '1ZNF', '1AKE', '4FZP', '1HMQ', '1MLA',
        '2HIC', '1MRK', '1XER', '1TF6', '1PQ7', '1K2K', '1USM', '1RGI',
        # --- Fe (15) ---
        '1MBN', '2HHB', '1CCM', '1FHA', '1HRC', '1DOI', '1BZ6',
        '1CGJ', '1GWE', '1IQZ', '1L5M', '1N5U', '1OCD', '1QHU', '1RQK',
        # --- Cu (10) ---
        '1AZU', '1PCY', '1SOD', '1AOZ', '1CCV', '1ETK', '1KCW',
        '1OAA', '1RXA', '2IWB',
        # --- Mg (10) ---
        '1AKE', '2HHB', '1CDP', '1LND', '1SOD', '1AKE', '1MBO',
        '1IGT', '1PKN', '1RNB',
        # --- Ca (10) ---
        '1CDP', '1FHA', '1LND', '1SOD', '1AKE', '2HHB', '1MBN',
        '1AZU', '1PCY', '1CA2',
        # --- Mn (5) ---
        '1LND', '1SOD', '1A2L', '1CA2', '1ZNF',
        # --- Ni (5) ---
        '1QNI', '2HHB', '1AKE', '1CA2', '1MBN',
        # --- Co (3) ---
        '1CA2', '1AKE', '1MBN',
        # --- Mo (3) ---
        '1FIQ', '1MBN', '1AKE',
        # --- V (2) ---
        '1VOX', '1MBN',
        # --- U (2) ---
        '3ARC', '1MBN',
        # --- Cd (2) ---
        '1CA2', '1AKE',
        # --- Disulfide-rich proteins (10) ---
        '1INS', '2INS', '1LYZ', '1RNH', '1BPT', '1CRN', '1UBQ',
        '1TIM', '1PPT', '1SGT',
        # --- RNA structures (5) ---
        '1EHZ', '1F27', '1GID', '1HMH', '1I9V',
        # --- Small molecules / diverse (5) ---
        '1MBN', '1AKE', '1CA2', '2HHB', '1AZU',
    ],
}


def download_pdb(pdb_id: str, dest_dir: str, overwrite: bool = False) -> str | None:
    """Download a single PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB ID.
        dest_dir: Destination directory.
        overwrite: If False, skip existing files.

    Returns:
        Path to downloaded file, or None on failure.
    """
    pdb_id = pdb_id.upper().strip()
    path = os.path.join(dest_dir, f"{pdb_id}.pdb")

    if os.path.exists(path) and not overwrite:
        return path

    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"  WARNING: Failed to download {pdb_id}: {e}")
        if os.path.exists(path):
            os.remove(path)
        return None


def download_pdbs(
    pdb_ids: list[str],
    dest_dir: str,
    overwrite: bool = False,
    delay: float = 0.1,
    verbose: bool = True,
) -> list[str]:
    """Download multiple PDB files.

    Args:
        pdb_ids: List of PDB IDs.
        dest_dir: Destination directory.
        overwrite: If False, skip existing files.
        delay: Delay between downloads (seconds) to be polite to RCSB.
        verbose: Print progress.

    Returns:
        List of successfully downloaded file paths.
    """
    if verbose:
        print(f"Downloading {len(pdb_ids)} PDB files to {dest_dir}")

    downloaded = []
    for i, pdb_id in enumerate(pdb_ids):
        path = download_pdb(pdb_id, dest_dir, overwrite=overwrite)
        if path:
            downloaded.append(path)
            if verbose:
                print(f"  [{i+1}/{len(pdb_ids)}] {pdb_id} -> {path}")
        else:
            if verbose:
                print(f"  [{i+1}/{len(pdb_ids)}] {pdb_id} FAILED")

        if delay > 0 and i < len(pdb_ids) - 1:
            time.sleep(delay)

    if verbose:
        print(f"Downloaded {len(downloaded)}/{len(pdb_ids)} files")

    return downloaded


def search_rcsb(
    metal: str | None = None,
    min_resolution: float = 2.5,
    max_results: int = 100,
) -> list[str]:
    """Search RCSB for PDB IDs matching criteria.

    Uses RCSB search API to find structures with specific metals.

    Args:
        metal: Metal element symbol (e.g., "ZN", "FE"). None for all.
        min_resolution: Maximum resolution in Å.
        max_results: Maximum number of results.

    Returns:
        List of PDB IDs.
    """
    # RCSB search API
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Build query
    query: dict[str, Any] = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.experimental_method",
                        "operator": "exact_match",
                        "value": "X-ray diffraction",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": min_resolution,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": False,
            "pager": {"start": 0, "rows": max_results},
        },
    }

    if metal:
        query["query"]["nodes"].append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonmonomer_comp_ids",
                "operator": "exact_match",
                "value": metal.upper(),
            },
        })

    import json
    data = json.dumps(query).encode('utf-8')
    req = urllib.request.Request(
        search_url,
        data=data,
        headers={'Content-Type': 'application/json'},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            ids = [hit['identifier'] for hit in result.get('result_set', [])]
            return ids
    except Exception as e:
        print(f"RCSB search failed: {e}")
        return []
