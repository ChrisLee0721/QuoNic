"""Check AWS Braket available devices across all regions."""

from braket.aws import AwsDevice
import boto3

print("=== AWS Braket QPU Devices ===\n")

regions = ['us-east-1', 'us-west-1', 'us-west-2', 'eu-north-1', 'eu-west-2', 'ap-southeast-1']

for region in regions:
    try:
        client = boto3.client('braket', region_name=region)
        resp = client.search_devices(filters=[{'name': 'deviceType', 'values': ['QPU']}])
        devices = resp.get('devices', [])
        if devices:
            print(f"Region: {region}")
            for d in devices:
                name = d['deviceName']
                status = d['deviceStatus']
                provider = d['providerName']
                arn = d['deviceArn']
                print(f"  {provider:15s} {name:25s} status={status}")
                print(f"    {arn}")
    except Exception as e:
        pass

print("\n=== Simulators ===")
for region in ['us-east-1', 'us-west-1', 'us-west-2']:
    try:
        sim_arn = f"arn:aws:braket:{region}::device/quantum-simulator/amazon/sv1"
        device = AwsDevice(sim_arn)
        print(f"SV1 ({region}): status={device.status}")
    except Exception as e:
        print(f"SV1 ({region}): {str(e)[:80]}")
