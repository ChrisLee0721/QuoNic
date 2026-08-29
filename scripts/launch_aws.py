"""Launch AWS EC2 instance for full PDB processing.

Usage:
    python scripts/launch_aws.py --profile gciqa --region us-east-1
"""

import argparse
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


def create_launch_template(profile: str, region: str) -> str:
    """Create EC2 launch template."""
    print("Creating launch template...")

    # User data script
    user_data = """#!/bin/bash
set -e
exec > /var/log/gciqa-setup.log 2>&1

# Install dependencies
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv rsync awscli git

# Setup
mkdir -p /data/gciqa
cd /data/gciqa

# Clone repo (replace with your repo URL)
git clone https://github.com/your-repo/PyQQQ.git /data/gciqa/PyQQQ 2>/dev/null || \
aws s3 cp s3://gciqa-code/PyQQQ.zip /tmp/PyQQQ.zip --no-sign-request 2>/dev/null

# Create venv
python3 -m venv /data/gciqa/venv
source /data/gciqa/venv/bin/activate
pip install numpy pandas pyarrow

# Download PDB files
echo "Downloading PDB files..."
rsync -rlpt -z --delete --port=33444 \
    rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
    /data/gciqa/pdbs/ \
    --parallel=4

# Decompress
cd /data/gciqa/pdbs
find . -name '*.pdb.gz' -exec gunzip -k {} \\;

# Run GCIQA
cd /data/gciqa
export PYTHONPATH=/data/gciqa/PyQQQ/src
python3 -m gciqa \
    /data/gciqa/pdbs \
    --bits 4 \
    --workers $(nproc) \
    --tolerance 0.5 \
    -o /data/gciqa/results/full_pdb_4bit.parquet

# Upload results
aws s3 cp /data/gciqa/results/ s3://gciqa-results/results/ --recursive

# Signal completion
aws s3 cp /var/log/gciqa-setup.log s3://gciqa-results/logs/setup.log

# Shutdown
shutdown -h now
"""

    # Write user data to file
    with open('/tmp/gciqa_user_data.sh', 'w') as f:
        f.write(user_data)

    # Create launch template
    cmd = f"""aws ec2 create-launch-template \
        --launch-template-name gciqa-full-pdb \
        --launch-template-data '{{
            "ImageId": "ami-0c55b159cbfafe1f0",
            "InstanceType": "r6i.2xlarge",
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
            "UserData": "$(base64 -w0 /tmp/gciqa_user_data.sh)",
            "TagSpecifications": [
                {{
                    "ResourceType": "instance",
                    "Tags": [
                        {{"Key": "Name", "Value": "gciqa-full-pdb"}},
                        {{"Key": "Project", "Value": "gciqa"}}
                    ]
                }}
            ]
        }}' \
        --profile {profile} \
        --region {region} \
        --query 'LaunchTemplate.LaunchTemplateId' \
        --output text"""

    result = run_cmd(cmd)
    template_id = result.stdout.strip()
    print(f"Launch template created: {template_id}")
    return template_id


def launch_spot_instance(template_id: str, profile: str, region: str) -> str:
    """Launch Spot instance."""
    print("Launching Spot instance...")

    cmd = f"""aws ec2 run-instances \
        --launch-template LaunchTemplateId={template_id} \
        --instance-market-options 'MarketType=spot,SpotOptions={{SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}}' \
        --profile {profile} \
        --region {region} \
        --query 'Instances[0].InstanceId' \
        --output text"""

    result = run_cmd(cmd)
    instance_id = result.stdout.strip()
    print(f"Spot instance launched: {instance_id}")
    return instance_id


def wait_for_instance(instance_id: str, profile: str, region: str) -> str:
    """Wait for instance to be running."""
    print(f"Waiting for {instance_id} to be running...")

    while True:
        cmd = f"""aws ec2 describe-instances \
            --instance-ids {instance_id} \
            --profile {profile} \
            --region {region} \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text"""

        result = run_cmd(cmd, check=False)
        state = result.stdout.strip()

        if state == 'running':
            print("Instance is running!")
            break
        elif state in ['terminated', 'shutting-down']:
            print(f"Instance {state}!")
            sys.exit(1)

        print(f"  State: {state}, waiting 10s...")
        time.sleep(10)

    # Get public IP
    cmd = f"""aws ec2 describe-instances \
        --instance-ids {instance_id} \
        --profile {profile} \
        --region {region} \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text"""

    result = run_cmd(cmd)
    public_ip = result.stdout.strip()
    print(f"Public IP: {public_ip}")
    return public_ip


def main():
    parser = argparse.ArgumentParser(description="Launch AWS instance for full PDB processing")
    parser.add_argument("--profile", default="gciqa", help="AWS CLI profile")
    parser.add_argument("--region", default="us-east-1", help="AWS region")

    args = parser.parse_args()

    # Create launch template
    template_id = create_launch_template(args.profile, args.region)

    # Launch Spot instance
    instance_id = launch_spot_instance(template_id, args.profile, args.region)

    # Wait for instance
    public_ip = wait_for_instance(instance_id, args.profile, args.region)

    print(f"\n==========================================")
    print(f"Instance ready!")
    print(f"==========================================")
    print(f"Instance ID: {instance_id}")
    print(f"Public IP: {public_ip}")
    print(f"\nTo monitor:")
    print(f"  ssh -i key.pem ubuntu@{public_ip}")
    print(f"  tail -f /var/log/gciqa-setup.log")
    print(f"\nTo download results:")
    print(f"  aws s3 cp s3://gciqa-results/results/ ./results/ --recursive")
    print(f"\nTo terminate:")
    print(f"  aws ec2 terminate-instances --instance-ids {instance_id}")


if __name__ == "__main__":
    main()
