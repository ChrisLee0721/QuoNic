"""Check AWS Braket pricing for each device."""

from braket.aws import AwsDevice

devices = {
    'Rigetti Cepheus-1-108Q': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q',
    'IQM Emerald': 'arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald',
    'IQM Garnet': 'arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet',
    'IonQ Forte Enterprise': 'arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1',
}

for name, arn in devices.items():
    try:
        device = AwsDevice(arn)
        pricing = device.properties.service.deviceCost
        print(f"{name}:")
        print(f"  type: {type(pricing)}")
        print(f"  attrs: {[a for a in dir(pricing) if not a.startswith('_')]}")
        # Try to access pricing fields
        for attr in dir(pricing):
            if not attr.startswith('_') and not callable(getattr(pricing, attr)):
                val = getattr(pricing, attr)
                print(f"  {attr}: {val}")
        print(f"  Status: {device.status}")
        print()
    except Exception as e:
        print(f"{name}: error - {str(e)[:120]}")
        print()
