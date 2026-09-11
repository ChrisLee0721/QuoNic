"""
Belenos overnight experiment v2: Complete photonic platform characterization.
CORRECTED: Uses qpu:belenos (hardware), proper controls, robust error handling.

Run this in Quandela Jupyter.
"""

import perceval as pcvl
import numpy as np
import time
import json
from datetime import datetime
from perceval.algorithm import Sampler

TOKEN = "YOUR_TOKEN_HERE"
OUTPUT_FILE = "belenos_hardware_results.json"

# Use HARDWARE, not simulator
BACKEND = "qpu:belenos"

def submit_and_wait(circuit, input_state, shots=1000, min_photons=0, timeout=180):
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
            time.sleep(3)  # Longer sleep to avoid rate limits

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
        'note': 'Hardware measurements on Belenos. T_ph not measured (requires mid-circuit measurement).',
        'loss': {},
        'dark_count': None,
        'HOM_indistinguishable': None,
        'HOM_distinguishable': None,
        'HOM_visibility': None,
        'depth_scan': {},
    }

    print(f"=== Belenos Hardware Experiment ===")
    print(f"Backend: {BACKEND} (HARDWARE)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_FILE}\n")

    # === Part 1: Loss Measurement ===
    print("=" * 60)
    print("PART 1: Loss Measurement (10000 shots)")
    print("=" * 60)

    for photons in [1, 2, 3]:
        for mode in [0, 1]:
            state = [0, 0]
            state[mode] = photons

            c = pcvl.Circuit(2)
            input_state = pcvl.BasicState(state)

            print(f"  |{photons},{1-photons if mode==0 else 0}> (mode {mode})...", end=" ", flush=True)
            metadata, status, job_id = submit_and_wait(c, input_state, shots=10000, min_photons=0)

            if metadata:
                results['loss'][f'{photons}_photons_mode_{mode}'] = metadata
                counts = metadata['counts']
                total = sum(counts.values())
                survival = sum(v for k, v in counts.items() if not all(x == '0' for x in k.strip('|').split(',')))
                print(f"OK (survival rate: {survival/total:.4f})")
            else:
                print(f"FAILED: {status}")

            time.sleep(3)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 2: Dark Count ===
    print("\n" + "=" * 60)
    print("PART 2: Dark Count (Vacuum Input, no photon filter)")
    print("=" * 60)

    c = pcvl.Circuit(2)
    input_state = pcvl.BasicState([0, 0])

    print("  Vacuum input (min_photons=0)...", end=" ", flush=True)
    metadata, status, job_id = submit_and_wait(c, input_state, shots=10000, min_photons=0)

    if metadata:
        results['dark_count'] = metadata
        counts = metadata['counts']
        total_dark = sum(v for k, v in counts.items() if k not in ['|0,0>', '|0>', '0'])
        print(f"OK (dark counts: {total_dark}/{sum(counts.values())})")
    else:
        print(f"FAILED: {status}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 3: HOM Visibility (WITH control) ===
    print("\n" + "=" * 60)
    print("PART 3: HOM Visibility (WITH distinguishable control)")
    print("=" * 60)

    # 3a: Indistinguishable photons (normal HOM)
    print("  3a: Indistinguishable...", end=" ", flush=True)
    c_indist = pcvl.Circuit(2)
    c_indist.add(0, pcvl.BS())

    metadata, status, job_id = submit_and_wait(c_indist, pcvl.BasicState([1, 1]), shots=10000)
    if metadata:
        results['HOM_indistinguishable'] = metadata
        counts = metadata['counts']
        total = sum(counts.values())
        p11_indist = counts.get('|1,1>', 0) / total if total > 0 else 0
        print(f"OK (P(|1,1>) = {p11_indist:.4f})")
    else:
        print(f"FAILED: {status}")
        p11_indist = None

    time.sleep(3)

    # 3b: Distinguishable photons (add phase to make them distinguishable)
    print("  3b: Distinguishable (with phase)...", end=" ", flush=True)
    c_dist = pcvl.Circuit(2)
    c_dist.add(0, pcvl.BS())
    c_dist.add((0,), pcvl.PS(np.pi/2))  # Phase to break symmetry

    metadata, status, job_id = submit_and_wait(c_dist, pcvl.BasicState([1, 1]), shots=10000)
    if metadata:
        results['HOM_distinguishable'] = metadata
        counts = metadata['counts']
        total = sum(counts.values())
        p11_dist = counts.get('|1,1>', 0) / total if total > 0 else 0
        print(f"OK (P(|1,1>) = {p11_dist:.4f})")
    else:
        print(f"FAILED: {status}")
        p11_dist = None

    # Calculate visibility
    if p11_indist is not None and p11_dist is not None and p11_dist > 0:
        visibility = 1 - (p11_indist / p11_dist)
        results['HOM_visibility'] = visibility
        print(f"\n  HOM Visibility = {visibility:.4f}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 4: Depth Scan ===
    print("\n" + "=" * 60)
    print("PART 4: Depth Scan (5 repetitions per depth)")
    print("=" * 60)

    depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 201, 301, 501, 601, 801]
    R = 10

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
                c, pcvl.BasicState([1, 1]), shots=5000, min_photons=2
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
    print("\nNOTE: This experiment measures Readout, Loss, and Control parameters.")
    print("Full T_ph measurement requires mid-circuit measurement (future work).")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
