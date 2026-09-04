#!/bin/bash
# One-click AWS EC2 deployment for QuoNic experiments.
#
# Usage:
#   1. Make sure AWS CLI is configured: aws configure
#   2. Run: bash experiments/paper/run_on_aws.sh
#
# What it does:
#   - Launches a c6i.4xlarge spot instance
#   - Installs Python + quonic dependencies
#   - Runs exp15, exp14, exp13 in order
#   - Downloads results to experiments/paper/results/
#   - Terminates the instance
#
# Estimated time: ~15-20 minutes total
# Estimated cost: ~$0.30-0.40 (spot pricing, m6i.9xlarge ~$1.72/hr on-demand, ~$0.52/hr spot)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INSTANCE_TYPE="m6i.9xlarge"  # 36 vCPU, 144 GB RAM
AMI_ID="ami-0c02fb55956877378"  # Amazon Linux 2023 (us-east-1)
KEY_NAME="quonic-exp-key"
SECURITY_GROUP_NAME="quonic-exp-sg"
REGION="us-east-1"
RESULTS_DIR="$(dirname "$0")/results"

echo "=========================================="
echo "QuoNic Experiment Runner (AWS EC2)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Create key pair if not exists
# ---------------------------------------------------------------------------
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
        --query 'KeyMaterial' --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "Key saved to ${KEY_NAME}.pem"
else
    echo "Key pair $KEY_NAME already exists"
fi

# ---------------------------------------------------------------------------
# Create security group if not exists
# ---------------------------------------------------------------------------
if ! aws ec2 describe-security-groups --group-names "$SECURITY_GROUP_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating security group: $SECURITY_GROUP_NAME"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "QuoNic experiment SG" \
        --region "$REGION" \
        --query 'GroupId' --output text)
    echo "Security group: $SG_ID"
else
    SG_ID=$(aws ec2 describe-security-groups \
        --group-names "$SECURITY_GROUP_NAME" \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' --output text)
    echo "Security group exists: $SG_ID"
fi

# ---------------------------------------------------------------------------
# Launch spot instance
# ---------------------------------------------------------------------------
echo "Launching $INSTANCE_TYPE spot instance..."

USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e

# Install Python + dependencies
yum update -y
yum install -y python3 python3-pip git
pip3 install --upgrade pip

# Clone repo (or upload code)
cd /home/ec2-user
git clone https://github.com/ChrisLee0721/QuoNic.git || true
cd QuoNic
pip3 install -e ".[all]" 2>/dev/null || pip3 install -e .

# Run experiments
mkdir -p experiments/paper/results

echo "=== Running exp15 (real workloads, ~1 min) ==="
python3 experiments/paper/exp15_real_workloads.py 2>&1 | tee experiments/paper/results/exp15_log.txt

echo "=== Running exp14 (groverize scaled, ~10 min) ==="
python3 experiments/paper/exp14_groverize_scaled.py 2>&1 | tee experiments/paper/results/exp14_log.txt

echo "=== Running exp13 (benchmark suite, ~1-2 hours) ==="
python3 experiments/paper/exp13_benchmark_suite.py 2>&1 | tee experiments/paper/results/exp13_log.txt

# Signal completion
touch /home/ec2-user/QuoNic/experiments/paper/results/DONE
USERDATA
)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP_NAME" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --user-data "$USER_DATA" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance running at: $PUBLIC_IP"
echo ""
echo "To monitor progress:"
echo "  ssh -i ${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo "  tail -f /home/ec2-user/QuoNic/experiments/paper/results/exp13_log.txt"
echo ""
echo "To download results when done:"
echo "  scp -i ${KEY_NAME}.pem ec2-user@${PUBLIC_IP}:/home/ec2-user/QuoNic/experiments/paper/results/*.json $RESULTS_DIR/"
echo ""
echo "To terminate instance:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"

# ---------------------------------------------------------------------------
# Wait for completion and download results
# ---------------------------------------------------------------------------
echo ""
echo "Waiting for experiments to complete (checking every 60s)..."
while true; do
    if ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        ec2-user@"$PUBLIC_IP" "test -f /home/ec2-user/QuoNic/experiments/paper/results/DONE" 2>/dev/null; then
        echo "Experiments complete!"
        break
    fi
    echo "  Still running... ($(date +%H:%M:%S))"
    sleep 60
done

# Download results
echo "Downloading results..."
mkdir -p "$RESULTS_DIR"
scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
    ec2-user@"$PUBLIC_IP":/home/ec2-user/QuoNic/experiments/paper/results/exp1[345]_*.json \
    "$RESULTS_DIR/"

echo "Results downloaded to $RESULTS_DIR"

# Terminate instance
echo "Terminating instance..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Instance $INSTANCE_ID terminated"
echo ""
echo "Done! Results in $RESULTS_DIR"
