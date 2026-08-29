#!/bin/bash
# AWS Full PDB Processing - Optimized Version
#
# Usage:
#   aws ec2 run-instances --instance-type r6i.2xlarge ...
#   ssh -i key.pem ubuntu@<ip>
#   bash aws_full_run.sh

set -e

# Configuration
S3_BUCKET="s3://gciqa-results"
WORK_DIR="/data/gciqa"
PDB_DIR="$WORK_DIR/pdbs"
RESULTS_DIR="$WORK_DIR/results"
PYTHON_ENV="$WORK_DIR/venv"
PYTHONPATH="$WORK_DIR/PyQQQ/src"

echo "=========================================="
echo "GCIQA Full PDB Processing (Optimized)"
echo "=========================================="
echo "Work dir: $WORK_DIR"
echo "PDB dir: $PDB_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# Setup
mkdir -p $WORK_DIR $PDB_DIR $RESULTS_DIR
cd $WORK_DIR

# Install dependencies
echo "[1/6] Setting up Python environment..."
if [ ! -d "$PYTHON_ENV" ]; then
    python3 -m venv $PYTHON_ENV
fi
source $PYTHON_ENV/bin/activate
pip install numpy pandas pyarrow --quiet

# Clone repo
echo "[2/6] Getting GCIQA code..."
if [ ! -d "$WORK_DIR/PyQQQ" ]; then
    # Copy from S3 or git clone
    aws s3 cp s3://gciqa-code/PyQQQ.zip /tmp/PyQQQ.zip --no-sign-request 2>/dev/null || \
    git clone https://github.com/your-repo/PyQQQ.git $WORK_DIR/PyQQQ
fi
export PYTHONPATH

# Download PDB files from RCSB
echo "[3/6] Downloading PDB files from RCSB..."
if [ ! -f "$PDB_DIR/.download_complete" ]; then
    # Use rsync for efficient download
    rsync -rlpt -z --delete --port=33444 \
        rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
        $PDB_DIR/ \
        --parallel=4 \
        --progress

    touch $PDB_DIR/.download_complete
    echo "Download complete: $(find $PDB_DIR -name '*.pdb.gz' | wc -l) files"
else
    echo "Already downloaded, skipping..."
fi

# Decompress
echo "[4/6] Decompressing PDB files..."
if [ ! -f "$PDB_DIR/.decompress_complete" ]; then
    cd $PDB_DIR
    find . -name '*.pdb.gz' -exec gunzip -k {} \;
    touch .decompress_complete
    echo "Decompressed: $(find . -name '*.pdb' | wc -l) files"
fi

# Run batch search
echo "[5/6] Running GCIQA batch search (optimized)..."
cd $WORK_DIR
python3 -m gciqa \
    $PDB_DIR \
    --bits 4 \
    --workers $(nproc) \
    --tolerance 0.5 \
    -o $RESULTS_DIR/full_pdb_4bit.parquet \
    2>&1 | tee $RESULTS_DIR/run.log

# Upload results to S3
echo "[6/6] Uploading results to S3..."
aws s3 cp $RESULTS_DIR/ $S3_BUCKET/results/ --recursive

echo ""
echo "=========================================="
echo "Processing complete!"
echo "Results: $S3_BUCKET/results/"
echo "=========================================="

# Shutdown
echo "Shutting down in 5 minutes..."
sleep 300
shutdown -h now
