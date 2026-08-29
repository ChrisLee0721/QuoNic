"""AWS deployment script for full PDB processing.

Usage:
    python scripts/aws_deploy.py --bucket gciqa-results --instance-type r6i.2xlarge

This script:
1. Creates an EC2 instance
2. Downloads all PDB files from RCSB
3. Runs GCIQA batch search
4. Uploads results to S3
5. Terminates the instance
"""

import argparse
import json
import os
import subprocess
import sys
import time


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        sys.exit(1)
    return result


def create_instance(instance_type: str, bucket: str) -> str:
    """Create EC2 instance for processing."""
    print("Creating EC2 instance...")

    # User data script
    user_data = f"""#!/bin/bash
set -e
exec > /var/log/gciqa-setup.log 2>&1

# Install dependencies
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv rsync awscli

# Setup
mkdir -p /data/gciqa
cd /data/gciqa

# Clone repo
git clone https://github.com/your-repo/PyQQQ.git /data/gciqa/PyQQQ
export PYTHONPATH=/data/gciqa/PyQQQ/src

# Create venv
python3 -m venv /data/gciqa/venv
source /data/gciqa/venv/bin/activate
pip install numpy

# Download PDB files
echo "Downloading PDB files..."
rsync -rlpt -z --delete --port=33444 \
    rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
    /data/gciqa/pdbs/ \
    --parallel=4

# Decompress
cd /data/gciqa/pdbs
find . -name '*.pdb.gz' -exec gunzip -k {{}} \\;

# Run GCIQA
cd /data/gciqa
python3 -m gciqa \
    /data/gciqa/pdbs \
    --bits 4 \
    --workers 8 \
    --tolerance 0.5 \
    -o /data/gciqa/results/full_pdb_4bit.json

# Upload results
aws s3 cp /data/gciqa/results/ s3://{bucket}/results/ --recursive

# Signal completion
aws s3 cp /var/log/gciqa-setup.log s3://{bucket}/logs/setup.log

# Shutdown
shutdown -h now
"""

    # Write user data to file
    with open('/tmp/gciqa_user_data.sh', 'w') as f:
        f.write(user_data)

    # Create instance
    cmd = f"""aws ec2 run-instances \
        --image-id ami-0c55b159cbfafe1f0 \
        --instance-type {instance_type} \
        --instance-market-options 'MarketType=spot' \
        --block-device-mappings '[{{"DeviceName":"/dev/sda1","Ebs":{{"VolumeSize":100,"VolumeType":"gp3"}}}}]' \
        --user-data file:///tmp/gciqa_user_data.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{{Key=Name,Value=gciqa-full-pdb}}]' \
        --query 'Instances[0].InstanceId' \
        --output text"""

    result = run_cmd(cmd)
    instance_id = result.stdout.strip()
    print(f"Instance created: {instance_id}")

    return instance_id


def wait_for_completion(instance_id: str, bucket: str, check_interval: int = 60):
    """Wait for instance to complete processing."""
    print(f"Waiting for {instance_id} to complete...")

    while True:
        # Check instance state
        cmd = f"aws ec2 describe-instances --instance-ids {instance_id} --query 'Reservations[0].Instances[0].State.Name' --output text"
        result = run_cmd(cmd, check=False)
        state = result.stdout.strip()

        if state == 'terminated':
            print("Instance terminated.")
            break
        elif state == 'stopped':
            print("Instance stopped.")
            break

        # Check if results exist in S3
        cmd = f"aws s3 ls s3://{bucket}/results/full_pdb_4bit.json"
        result = run_cmd(cmd, check=False)
        if result.returncode == 0:
            print("Results found in S3!")
            break

        print(f"  State: {state}, waiting {check_interval}s...")
        time.sleep(check_interval)


def download_results(bucket: str, local_dir: str):
    """Download results from S3."""
    print(f"Downloading results to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)

    cmd = f"aws s3 cp s3://{bucket}/results/ {local_dir}/ --recursive"
    run_cmd(cmd)

    print(f"Results downloaded to {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="AWS deployment for full PDB processing")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--instance-type", default="r6i.2xlarge",
                        help="EC2 instance type (default: r6i.2xlarge)")
    parser.add_argument("--download", help="Download results to local directory")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for completion")
    parser.add_argument("--instance-id", help="Existing instance ID to wait for")

    args = parser.parse_args()

    if args.download:
        download_results(args.bucket, args.download)
        return

    if args.wait:
        if not args.instance_id:
            print("Error: --instance-id required with --wait", file=sys.stderr)
            sys.exit(1)
        wait_for_completion(args.instance_id, args.bucket)
        return

    # Create instance and run
    instance_id = create_instance(args.instance_type, args.bucket)
    print(f"\nInstance {instance_id} created.")
    print(f"Monitor: aws ec2 describe-instances --instance-ids {instance_id}")
    print(f"Logs: aws s3 ls s3://{args.bucket}/logs/")
    print(f"\nTo wait: python scripts/aws_deploy.py --bucket {args.bucket} --wait --instance-id {instance_id}")
    print(f"To download: python scripts/aws_deploy.py --bucket {args.bucket} --download results/")


if __name__ == "__main__":
    main()
