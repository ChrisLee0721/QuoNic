"""Launch AWS EC2 instance for full PDB processing.

Usage:
    python scripts/launch_aws_full.py
"""

import base64
import subprocess
import sys
import time


PROFILE = "gciqa"
REGION = "us-east-1"
INSTANCE_TYPE = "r6i.2xlarge"
BUCKET_NAME = "gciqa-results"


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        sys.exit(1)
    return result


def main():
    print("=" * 50)
    print("GCIQA Full PDB Processing - AWS Launch")
    print("=" * 50)
    print(f"Profile: {PROFILE}")
    print(f"Region: {REGION}")
    print(f"Instance: {INSTANCE_TYPE}")
    print()

    # Create S3 bucket
    print("[1/5] Creating S3 bucket...")
    run_cmd(f"aws s3 mb s3://{BUCKET_NAME} --profile {PROFILE} --region {REGION}", check=False)

    # User data script
    user_data = """#!/bin/bash
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
rsync -rlpt -z --delete --port=33444 \\
    rsync.rcsb.org::ftp_data/structures/divided/pdb/ \\
    /data/gciqa/pdbs/ --parallel=4

# Decompress
cd /data/gciqa/pdbs
find . -name '*.pdb.gz' -exec gunzip -k {} \\;

# Run GCIQA
cd /data/gciqa
export PYTHONPATH=/data/gciqa/PyQQQ/src
python3 -m gciqa /data/gciqa/pdbs --bits 4 --workers $(nproc) --tolerance 0.5 -o /data/gciqa/results/full_pdb_4bit.parquet

# Upload results
aws s3 cp /data/gciqa/results/ s3://gciqa-results/results/ --recursive
aws s3 cp /var/log/gciqa-setup.log s3://gciqa-results/logs/setup.log

shutdown -h now
"""

    # Encode user data
    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    # Create launch template
    print("[2/5] Creating launch template...")
    launch_template_data = f'''{{
        "ImageId": "ami-0c55b159cbfafe1f0",
        "InstanceType": "{INSTANCE_TYPE}",
        "BlockDeviceMappings": [
            {{
                "DeviceName": "/dev/sda1",
                "Ebs": {{
                    "VolumeSize": 2000,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": true
                }}
            }}
        ],
        "UserData": "{user_data_b64}",
        "TagSpecifications": [
            {{
                "ResourceType": "instance",
                "Tags": [
                    {{"Key": "Name", "Value": "gciqa-full-pdb"}},
                    {{"Key": "Project", "Value": "gciqa"}}
                ]
            }}
        ]
    }}'''

    # Write launch template data to file
    with open("launch_template.json", "w") as f:
        f.write(launch_template_data)

    result = run_cmd(f'aws ec2 create-launch-template --launch-template-name gciqa-full-pdb --launch-template-data file://launch_template.json --profile {PROFILE} --region {REGION} --query "LaunchTemplate.LaunchTemplateId" --output text')
    template_id = result.stdout.strip()
    print(f"Launch template: {template_id}")

    # Launch Spot instance
    print("[3/5] Launching Spot instance...")
    market_options = '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
    result = run_cmd(f'aws ec2 run-instances --launch-template LaunchTemplateId={template_id} --instance-market-options {market_options} --profile {PROFILE} --region {REGION} --query "Instances[0].InstanceId" --output text')
    instance_id = result.stdout.strip()
    print(f"Instance ID: {instance_id}")

    # Wait for instance
    print("[4/5] Waiting for instance to be running...")
    while True:
        result = run_cmd(f'aws ec2 describe-instances --instance-ids {instance_id} --profile {PROFILE} --region {REGION} --query "Reservations[0].Instances[0].State.Name" --output text', check=False)
        state = result.stdout.strip()

        if state == "running":
            print("Instance is running!")
            break
        elif state in ["terminated", "shutting-down"]:
            print(f"Instance {state}!")
            sys.exit(1)

        print(f"  State: {state}, waiting 10s...")
        time.sleep(10)

    # Get public IP
    result = run_cmd(f'aws ec2 describe-instances --instance-ids {instance_id} --profile {PROFILE} --region {REGION} --query "Reservations[0].Instances[0].PublicIpAddress" --output text')
    public_ip = result.stdout.strip()

    print("[5/5] Instance ready!")
    print()
    print("=" * 50)
    print(f"Instance ID: {instance_id}")
    print(f"Public IP: {public_ip}")
    print("=" * 50)
    print()
    print("To monitor:")
    print(f"  ssh -i key.pem ubuntu@{public_ip}")
    print(f"  tail -f /var/log/gciqa-setup.log")
    print()
    print("To download results:")
    print(f"  aws s3 cp s3://{BUCKET_NAME}/results/ ./results/ --recursive")
    print()
    print("To terminate:")
    print(f"  aws ec2 terminate-instances --instance-ids {instance_id}")


if __name__ == "__main__":
    main()
