"""Quick check AWS Braket access."""
import boto3

client = boto3.client('braket', region_name='us-west-1')
resp = client.get_device(deviceArn='arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q')
print(f"Device: {resp['deviceName']}")
print(f"Status: {resp['deviceStatus']}")
print(f"Qubits: {resp['deviceCapabilities']['paradigm']['qubitCount']}")
print("\nAWS Braket access OK!")
