"""Run full PDB batch search on AWS instance."""
import sys
import time
import traceback

sys.path.insert(0, '/data/gciqa/PyQQQ/src')

from gciqa.batch import batch_search


def main():
    print("Starting full PDB batch search...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        result = batch_search(
            ['/data/gciqa/pdbs'],
            bits=4,
            n_workers=8,
            tolerance=0.5,
            verbose=True,
        )

        result.save('/data/gciqa/results/full_pdb_4bit.parquet')
        result.print_summary()

        print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Results saved to /data/gciqa/results/full_pdb_4bit.parquet")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
