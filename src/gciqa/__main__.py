"""CLI entry point for GCIQA batch processing.

Usage::

    # Process all PDB files in a directory
    python -m gciqa data/pdbs/ -o results.json

    # Process specific files
    python -m gciqa file1.pdb file2.pdb -o results.json

    # Custom parameters
    python -m gciqa data/pdbs/ --bits 3 4 5 --tolerance 0.3 -o results.json

    # Download PDB files first
    python -m gciqa --download data/pdbs/ --set metal_diversity
    python -m gciqa --download data/pdbs/ --ids 1AKE 3ARC 1AZU
    python -m gciqa --download data/pdbs/ --metal ZN --max 100

    # Download then process in one step
    python -m gciqa --download data/pdbs/ --set metal_diversity --run -o results.json
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="python -m gciqa",
        description="GCIQA batch metal site search. Processes PDB files, "
                    "extracts metal binding sites, runs distance-based "
                    "conformation search, outputs results to JSON.",
    )

    # Download options
    parser.add_argument(
        "--download",
        metavar="DIR",
        help="Download PDB files to this directory",
    )
    parser.add_argument(
        "--set",
        choices=['metal_diversity', 'nonstandard_geometry', 'large_scale', 'diverse_100'],
        help="Predefined PDB set to download",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific PDB IDs to download",
    )
    parser.add_argument(
        "--metal",
        help="Search RCSB for structures containing this metal (e.g., ZN, FE)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=100,
        help="Max results from RCSB search (default: 100)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run batch search after download",
    )

    # Search options
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDB files or directories containing PDB files",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (.json or .parquet, default: print to stdout)",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[4],
        help="Bit depth(s) for encoding (default: 4)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Distance tolerance in Å (default: 0.5)",
    )
    parser.add_argument(
        "--dist-range",
        nargs=2,
        type=float,
        default=[0.0, 5.0],
        metavar=("MIN", "MAX"),
        help="Physical distance range in Å (default: 0.0 5.0)",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=3.0,
        help="Max coordination distance for ligand detection (default: 3.0)",
    )
    parser.add_argument(
        "--min-ligands",
        type=int,
        default=3,
        help="Minimum coordinating atoms per site (default: 3)",
    )
    parser.add_argument(
        "--max-ligands",
        type=int,
        default=6,
        help="Maximum ligands per site (default: 6)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Download mode
    if args.download:
        from .download import PDB_SETS, download_pdbs, search_rcsb

        dest = args.download
        pdb_ids = []

        if args.set:
            pdb_ids = PDB_SETS[args.set]
            print(f"Using predefined set '{args.set}': {len(pdb_ids)} PDB IDs")
        elif args.ids:
            pdb_ids = args.ids
        elif args.metal:
            print(f"Searching RCSB for {args.metal} structures...")
            pdb_ids = search_rcsb(metal=args.metal, max_results=args.max)
            print(f"Found {len(pdb_ids)} structures")
        else:
            print("Error: specify --set, --ids, or --metal for download", file=sys.stderr)
            sys.exit(1)

        downloaded = download_pdbs(pdb_ids, dest, verbose=not args.quiet)

        if not args.run:
            print(f"\nDone. {len(downloaded)} files in {dest}")
            print(f"To process: python -m gciqa {dest} -o results.json")
            return

        # If --run, use download dir as input
        args.paths = [dest]

    # Search mode
    if not args.paths:
        parser.print_help()
        sys.exit(1)

    # Validate paths
    for p in args.paths:
        if not os.path.exists(p):
            print(f"Error: path not found: {p}", file=sys.stderr)
            sys.exit(1)

    from .batch import batch_search

    result = batch_search(
        pdb_paths=args.paths,
        bits=args.bits if len(args.bits) > 1 else args.bits[0],
        tolerance=args.tolerance,
        distance_range=tuple(args.dist_range),
        max_coord_dist=args.max_dist,
        min_coord_atoms=args.min_ligands,
        max_ligands=args.max_ligands,
        n_workers=args.workers,
        verbose=not args.quiet,
    )

    if args.output:
        result.save(args.output)
        if not args.quiet:
            print(f"Results saved to {args.output}")
    else:
        result.print_summary()


if __name__ == "__main__":
    main()
