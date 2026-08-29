#!/bin/bash
# Quick AWS launch script for GCIQA full PDB processing
#
# Usage: bash scripts/run_aws.sh

set -e

PROFILE="gciqa"
REGION="us-east-1"
INSTANCE_TYPE="r6i.2xlarge"

echo "=========================================="
echo "GCIQA Full PDB Processing - AWS Launch"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE"
echo ""

# Create S3 bucket for results
echo "[1/5] Creating S3 bucket..."
aws s3 mb s3://gciqa-results --profile $PROFILE --region $REGION 2>/dev/null || true

# Create launch template
echo "[2/5] Creating launch template..."
cat > /tmp/gciqa_user_data.sh << 'USERDATA'
#!/bin/bash
set -e
exec > /var/log/gciqa-setup.log 2>&1

apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv rsync awscli git

mkdir -p /data/gciqa
cd /data/gciqa

# Clone repo
git clone https://github.com/your-repo/PyQQQ.git /data/gciqa/PyQQQ 2>/dev/null || true

# Setup Python
python3 -m venv /data/gciqa/venv
source /data/gciqa/venv/bin/activate
pip install numpy pandas pyarrow

# Download PDB
echo "Downloading PDB files..."
rsync -rlpt -z --delete --port=33444 \
    rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
    /data/gciqa/pdbs/ --parallel=4

# Decompress
cd /data/gciqa/pdbs
find . -name '*.pdb.gz' -exec gunzip -k {} \;

# Run GCIQA
cd /data/gciqa
export PYTHONPATH=/data/gciqa/PyQQQ/src
python3 -m gciqa /data/gciqa/pdbs --bits 4 --workers $(nproc) --tolerance 0.5 -o /data/gciqa/results/full_pdb_4bit.parquet

# Upload results
aws s3 cp /data/gciqa/results/ s3://gciqa-results/results/ --recursive
aws s3 cp /var/log/gciqa-setup.log s3://gciqa-results/logs/setup.log

shutdown -h now
USERDATA

# Launch instance
echo "[3/5] Launching Spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type $INSTANCE_TYPE \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":2000,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data file:///tmp/gciqa_user_data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gciqa-full-pdb},{Key=Project,Value=gciqa}]' \
    --profile $PROFILE \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for instance
echo "[4/5] Waiting for instance to be running..."
while true; do
    STATE=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --profile $PROFILE \
        --region $REGION \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text)
    
    if [ "$STATE" = "running" ]; then
        break
    fi
    
    echo "  State: $STATE, waiting 10s..."
    sleep 10
done

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --profile $PROFILE \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "[5/5] Instance ready!"
echo ""
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "=========================================="
echo ""
echo "To monitor:"
echo "  ssh -i key.pem ubuntu@$PUBLIC_IP"
echo "  tail -f /var/log/gciqa-setup.log"
echo ""
echo "To download results:"
echo "  aws s3 cp s3://gciqa-results/results/ ./results/ --recursive"
echo ""
echo "To terminate:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
