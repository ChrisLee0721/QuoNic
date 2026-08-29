# GCIQA Full PDB Processing Scripts

## Quick Start (Local)

```bash
# Download all PDB files
python scripts/download_all_pdb.py --dest data/pdbs_all --method rsync

# Run batch search
PYTHONPATH=src python -m gciqa data/pdbs_all --bits 4 --workers 8 -o results/full_pdb_4bit.json
```

## AWS Deployment

### Option 1: Automated (Recommended)

```bash
# Create instance and run
python scripts/aws_deploy.py --bucket gciqa-results --instance-type r6i.2xlarge

# Wait for completion
python scripts/aws_deploy.py --bucket gciqa-results --wait --instance-id i-xxxxx

# Download results
python scripts/aws_deploy.py --bucket gciqa-results --download results/
```

### Option 2: Manual

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type r6i.2xlarge \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'

# 2. SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# 3. Run the script
bash scripts/aws_run_full.sh
```

## Resource Estimates

| Resource | Estimate |
|----------|----------|
| PDB files | 246,905 |
| Download size | ~25 GB |
| Download time | ~2 hours |
| Metal sites | ~5,000,000 |
| Compute time | ~1 hour (8 cores) |
| Results size | ~50-100 GB |
| AWS cost | ~$5-10 |

## Output Format

For large datasets (>10K sites), results are saved in chunks:

```
results/
├── full_pdb_4bit_summary.json    # Summary statistics
├── full_pdb_4bit_chunk0000.json  # Sites 0-9999
├── full_pdb_4bit_chunk0001.json  # Sites 10000-19999
└── ...
```

## Analysis

```python
import json
import glob

# Load summary
with open('results/full_pdb_4bit_summary.json') as f:
    summary = json.load(f)

print(f"Total sites: {summary['n_sites']}")
print(f"By bits: {summary['summary']['by_bits']}")

# Load all chunks
all_sites = []
for chunk_file in sorted(glob.glob('results/full_pdb_4bit_chunk*.json')):
    with open(chunk_file) as f:
        all_sites.extend(json.load(f))

# Check theoretical limit
theoretical_limit = 5.0 / (2 ** (4 + 1))  # 0.15625 Å
exceeded = [s for s in all_sites if s['best_error'] > theoretical_limit]
print(f"Sites exceeding limit: {len(exceeded)}")
```
