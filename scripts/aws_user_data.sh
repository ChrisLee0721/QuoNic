#!/bin/bash
set -e
exec > /var/log/gciqa-setup.log 2>&1

echo "=========================================="
echo "GCIQA Full PDB Processing"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Install dependencies
echo "[1/6] Installing dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv rsync awscli git

# Setup directories
echo "[2/6] Setting up directories..."
mkdir -p /data/gciqa/{pdbs,results}
cd /data/gciqa

# Create Python environment
echo "[3/6] Creating Python environment..."
python3 -m venv /data/gciqa/venv
source /data/gciqa/venv/bin/activate
pip install numpy pandas pyarrow --quiet

# Clone repo
echo "[4/6] Getting GCIQA code..."
git clone https://github.com/anthropics/PyQQQ.git /data/gciqa/PyQQQ 2>/dev/null || \
git clone https://github.com/user/PyQQQ.git /data/gciqa/PyQQQ 2>/dev/null || \
echo "Git clone failed, will try S3..."

export PYTHONPATH=/data/gciqa/PyQQQ/src

# Download PDB files from S3
echo "[5/6] Downloading PDB files from S3..."
aws configure set default.s3.max_concurrent_requests 128
aws configure set default.s3.max_queue_size 10000
aws s3 sync s3://pdbsnapshots/20260101/pub/pdb/data/structures/divided/pdb/ /data/gciqa/pdbs/ --request-payer requester --region us-east-1

echo "Download complete: $(find /data/gciqa/pdbs -name '*.ent.gz' | wc -l) files"

# Decompress
echo "Decompressing PDB files..."
cd /data/gciqa/pdbs
find . -name '*.ent.gz' -exec gunzip -k {} \;
echo "Decompressed: $(find . -name '*.ent' | wc -l) files"

# Run GCIQA
echo "[6/6] Running GCIQA batch search..."
cd /data/gciqa
python3 -m gciqa \
    /data/gciqa/pdbs \
    --bits 4 \
    --workers $(nproc) \
    --tolerance 0.5 \
    -o /data/gciqa/results/full_pdb_4bit.parquet \
    2>&1 | tee /data/gciqa/results/run.log

# Upload results to S3
echo "Uploading results to S3..."
aws s3 cp /data/gciqa/results/ s3://gciqa-results/results/ --recursive
aws s3 cp /var/log/gciqa-setup.log s3://gciqa-results/logs/setup.log

echo ""
echo "=========================================="
echo "Processing complete!"
echo "End time: $(date)"
echo "Results: s3://gciqa-results/results/"
echo "=========================================="

# Shutdown
echo "Shutting down in 5 minutes..."
sleep 300
shutdown -h now
