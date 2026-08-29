"""Download all PDB files from RCSB using rsync.

Usage:
    python scripts/download_all_pdb.py --dest /data/pdbs --workers 4

RCSB provides rsync access to all PDB files:
    rsync -rlpt -v -z --delete --port=33444 \
        rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
        /data/pdbs/
"""

import argparse
import os
import subprocess
import sys
import time


RCSB_RSYNC = "rsync.rcsb.org::ftp_data/structures/divided/pdb/"


def download_all(dest: str, workers: int = 4, dry_run: bool = False):
    """Download all PDB files using rsync.

    Args:
        dest: Destination directory.
        workers: Number of parallel transfers.
        dry_run: If True, only show what would be downloaded.
    """
    os.makedirs(dest, exist_ok=True)

    cmd = [
        "rsync",
        "-rlpt",  # recursive, links, perms, times
        "-v",     # verbose
        "-z",     # compress
        "--delete",
        f"--port=33444",
        f"--bwlimit=0",  # no bandwidth limit
        f"--progress",
    ]

    if workers > 1:
        cmd.extend([f"--parallel={workers}"])

    if dry_run:
        cmd.append("--dry-run")

    cmd.extend([RCSB_RSYNC, dest])

    print(f"Running: {' '.join(cmd)}")
    print(f"Destination: {dest}")
    print(f"Workers: {workers}")
    print(f"Dry run: {dry_run}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\nDownload completed in {elapsed:.0f}s")
        # Count files
        n_files = sum(1 for f in os.listdir(dest) if f.endswith('.pdb.gz'))
        print(f"Downloaded {n_files} files")
    else:
        print(f"\nDownload failed with return code {result.returncode}")
        sys.exit(1)


def download_selected(dest: str, pdb_ids: list[str], delay: float = 0.1):
    """Download specific PDB files.

    Args:
        dest: Destination directory.
        pdb_ids: List of PDB IDs.
        delay: Delay between downloads.
    """
    import urllib.request

    os.makedirs(dest, exist_ok=True)

    for i, pdb_id in enumerate(pdb_ids):
        pdb_id = pdb_id.upper().strip()
        path = os.path.join(dest, f"{pdb_id}.pdb")

        if os.path.exists(path):
            print(f"  [{i+1}/{len(pdb_ids)}] {pdb_id} exists, skipping")
            continue

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  [{i+1}/{len(pdb_ids)}] {pdb_id} downloaded")
        except Exception as e:
            print(f"  [{i+1}/{len(pdb_ids)}] {pdb_id} FAILED: {e}")

        if delay > 0:
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Download PDB files from RCSB")
    parser.add_argument("--dest", default="data/pdbs_all",
                        help="Destination directory")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for rsync")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded")
    parser.add_argument("--method", choices=["rsync", "http"], default="rsync",
                        help="Download method")
    parser.add_argument("--ids", nargs="+",
                        help="Specific PDB IDs to download (http method)")

    args = parser.parse_args()

    if args.method == "rsync":
        download_all(args.dest, args.workers, args.dry_run)
    else:
        if not args.ids:
            print("Error: --ids required for http method", file=sys.stderr)
            sys.exit(1)
        download_selected(args.dest, args.ids)


if __name__ == "__main__":
    main()
