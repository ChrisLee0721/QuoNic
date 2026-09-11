"""Search AWS Braket for ALL devices (QPUs + simulators)."""

import boto3

print("=== AWS Braket Device Search ===\n")

regions = ['us-east-1', 'us-west-1', 'us-west-2', 'eu-north-1', 'eu-west-2']

for region in regions:
    client = boto3.client('braket', region_name=region)
    
    # Search all devices
    try:
        resp = client.search_devices(filters=[])
        devices = resp.get('devices', [])
        if devices:
            print(f"Region: {region} ({len(devices)} devices)")
            for d in devices:
                name = d.get('deviceName', '?')
                status = d.get('deviceStatus', '?')
                provider = d.get('providerName', '?')
                dtype = d.get('deviceType', '?')
                arn = d.get('deviceArn', '?')
                marker = ">>>" if status == "ONLINE" else "   "
                print(f"  {marker} {provider:15s} {name:30s} type={dtype:10s} status={status}")
            print()
    except Exception as e:
        print(f"  {region}: error - {str(e)[:80]}")
