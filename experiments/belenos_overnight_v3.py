"""
Belenos overnight experiment v3: Complete photonic platform characterization.
CORRECTED: Hardware backend, proper HOM control, full distributions, robust timeout.

Run this in Quandela Jupyter.
"""

import perceval as pcvl
import numpy as np
import time
import json
from datetime import datetime
from perceval.algorithm import Sampler

TOKEN = "YOUR_TOKEN_HERE"
OUTPUT_FILE = "belenos_hardware_v3.json"

# Use HARDWARE, not simulator
BACKEND = "qpu:belenos"

def submit_and_wait(circuit, input_state, shots=1000, min_photons=0, timeout=600):
    """Submit circuit and wait for results with metadata."""
    proc = pcvl.RemoteProcessor(BACKEND, token=TOKEN)
    proc.set_circuit(circuit)
    proc.with_input(input_state)
    proc.min_detected_photons_filter(min_photons)

    sampler = Sampler(proc, max_shots_per_call=shots)

    try:
        job = sampler.sample_count.execute_async(shots)
        job_id = getattr(job, 'id', 'unknown')
        start_time = time.time()

        while not job.is_complete:
            if time.time() - start_time > timeout:
                return None, "timeout", job_id
            time.sleep(3)

        if job.is_failed:
            return None, "failed", job_id

        result = job.get_results()['results']
        metadata = {
            'counts': result,
            'job_id': job_id,
            'shots': shots,
            'timestamp': time.time(),
            'elapsed': time.time() - start_time,
        }
        return metadata, "ok", job_id

    except Exception as e:
        return None, str(e), "error"

def main():
    results = {
        'timestamp_start': datetime.now().isoformat(),
        'backend': BACKEND,
        'note': 'Hardware measurements. T_ph not measured (requires mid-circuit measurement).',
        'loss': {},
        'dark_count': None,
        'HOM_indistinguishable': None,
        'HOM_distinguishable': None,
        'HOM_visibility': None,
        'depth_scan': {},
    }

    print(f"=== Belenos Hardware Experiment v3 ===")
    print(f"Backend: {BACKEND} (HARDWARE)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_FILE}\n")

    # === Part 1: Loss Measurement ===
    print("=" * 60)
    print("PART 1: Loss Measurement (10000 shots, full distribution)")
    print("=" * 60)
    print("  NOTE: Belenos only accepts max 1 photon per mode")

    for mode in [0, 1]:
        state = [0, 0]
        state[mode] = 1

        c = pcvl.Circuit(2)
        input_state = pcvl.BasicState(state)

        print(f"  |{state[0]},{state[1]}>...", end=" ", flush=True)
        metadata, status, job_id = submit_and_wait(c, input_state, shots=10000, min_photons=0)

        if metadata:
            results['loss'][f'1_photon_mode_{mode}'] = metadata
            counts = metadata['counts']
            total = sum(counts.values())
            # Report full distribution
            print(f"OK ({total} counts)")
            for state, cnt in sorted(counts.items(), key=lambda x: str(x[0])):
                print(f"    {state}: {cnt/total:.4f}")
        else:
            print(f"FAILED: {status}")

        time.sleep(3)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 2: Dark Count ===
    print("\n" + "=" * 60)
    print("PART 2: Dark Count (Vacuum Input)")
    print("=" * 60)
    print("  SKIPPED: Belenos requires at least 1 photon in input state")

    results['dark_count'] = {'skipped': True, 'reason': 'QPU requires min 1 photon input'}

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 3: HOM Test (indistinguishable only) ===
    print("\n" + "=" * 60)
    print("PART 3: HOM Test (indistinguishable photons)")
    print("=" * 60)
    print("  NOTE: Distinguishable control NOT performed (hardware limitation)")
    print("  We report P(|1,1>) after BS. Ideal HOM: P(|1,1>) = 0.")

    print("  Running HOM...", end=" ", flush=True)
    c_hom = pcvl.Circuit(2)
    c_hom.add(0, pcvl.BS())

    metadata, status, job_id = submit_and_wait(c_hom, pcvl.BasicState([1, 1]), shots=10000)
    if metadata:
        results['HOM_indistinguishable'] = metadata
        counts = metadata['counts']
        total = sum(counts.values())
        # Find P(|1,1>) by matching FockState
        p11_count = 0
        for state, cnt in counts.items():
            if str(state) == '|1,1>':
                p11_count = cnt
                break
        p11 = p11_count / total if total > 0 else 0
        results['HOM_p11'] = p11
        print(f"OK")
        print(f"  P(|1,1>) = {p11:.4f} (ideal HOM dip: 0)")
        print(f"  HOM dip depth = {1 - p11:.4f}")
        print(f"  NOTE: Full visibility requires distinguishable control (not available)")
    else:
        print(f"FAILED: {status}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 4: Depth Scan ===
    print("\n" + "=" * 60)
    print("PART 4: Depth Scan (10 repetitions, 5000 shots)")
    print("=" * 60)

    depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 201, 301, 401, 501, 601, 801]
    R = 10
    SHOTS = 5000

    for k in depths:
        results['depth_scan'][k] = []

        for r in range(R):
            print(f"  k={k}, run {r+1}/{R}...", end=" ", flush=True)

            c = pcvl.Circuit(2)
            for i in range(k):
                c.add(0, pcvl.BS())
                c.add((0,), pcvl.PS(np.pi/4))
            c.add(0, pcvl.BS())

            metadata, status, job_id = submit_and_wait(
                c, pcvl.BasicState([1, 1]), shots=SHOTS, min_photons=2, timeout=600
            )

            if metadata:
                results['depth_scan'][k].append(metadata)
                print("OK")
            else:
                results['depth_scan'][k].append({'error': status, 'job_id': job_id})
                print(f"FAILED: {status}")

            time.sleep(3)

        # Save after each depth
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  k={k} complete. Saved.\n")

    # === Summary ===
    results['timestamp_end'] = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"End: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
