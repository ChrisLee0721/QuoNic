"""
Belenos overnight experiment: Complete photonic platform characterization.
Run this in Quandela Jupyter.
"""

import perceval as pcvl
import numpy as np
import time
import json
from datetime import datetime
from perceval.algorithm import Sampler

TOKEN = "YOUR_TOKEN_HERE"
OUTPUT_FILE = "belenos_full_results.json"

def submit_and_wait(circuit, input_state, shots=1000, timeout=120):
    """Submit circuit and wait for results."""
    proc = pcvl.RemoteProcessor("sim:belenos", token=TOKEN)
    proc.set_circuit(circuit)
    proc.with_input(input_state)
    proc.min_detected_photons_filter(0)

    sampler = Sampler(proc, max_shots_per_call=shots)

    try:
        job = sampler.sample_count.execute_async(shots)
        start_time = time.time()

        while not job.is_complete:
            if time.time() - start_time > timeout:
                return None, "timeout"
            time.sleep(2)

        if job.is_failed:
            return None, "failed"

        return job.get_results()['results'], "ok"
    except Exception as e:
        return None, str(e)

def main():
    results = {
        'timestamp': datetime.now().isoformat(),
        'loss': {},
        'dark_count': None,
        'HOM': {},
        'depth_scan': {},
    }

    print(f"=== Belenos Overnight Experiment ===")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_FILE}\n")

    # === Part 1: Loss Measurement ===
    print("=" * 50)
    print("PART 1: Loss Measurement")
    print("=" * 50)

    for photons in [1, 2, 3]:
        for mode in [0, 1]:
            state = [0, 0]
            state[mode] = photons

            c = pcvl.Circuit(2)
            input_state = pcvl.BasicState(state)

            print(f"  {photons} photons, mode {mode}...", end=" ", flush=True)
            result, status = submit_and_wait(c, input_state)

            if result:
                results['loss'][f'{photons}_photons_mode_{mode}'] = result
                print(f"OK: {result}")
            else:
                print(f"FAILED: {status}")

            time.sleep(2)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 2: Dark Count ===
    print("\n" + "=" * 50)
    print("PART 2: Dark Count (Vacuum Input)")
    print("=" * 50)

    c = pcvl.Circuit(2)
    input_state = pcvl.BasicState([0, 0])

    print("  Vacuum input...", end=" ", flush=True)
    result, status = submit_and_wait(c, input_state)

    if result:
        results['dark_count'] = result
        print(f"OK: {result}")
    else:
        print(f"FAILED: {status}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 3: HOM Visibility ===
    print("\n" + "=" * 50)
    print("PART 3: Hong-Ou-Mandel Visibility")
    print("=" * 50)

    c = pcvl.Circuit(2)
    c.add(0, pcvl.BS())  # 50:50 beam splitter

    print("  HOM experiment...", end=" ", flush=True)
    result, status = submit_and_wait(c, pcvl.BasicState([1, 1]))

    if result:
        results['HOM'] = result
        print(f"OK: {result}")

        # Calculate visibility
        total = sum(result.values())
        p11 = result.get('|1,1>', 0) / total if total > 0 else 0
        visibility = 1 - p11 / 0.5  # Ideal P(|1,1>) for distinguishable = 0.5
        results['HOM_visibility'] = visibility
        print(f"  HOM Visibility: {visibility:.4f}")
    else:
        print(f"FAILED: {status}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Part 4: Depth Scan ===
    print("\n" + "=" * 50)
    print("PART 4: Depth Scan")
    print("=" * 50)

    depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 201, 301, 401, 501, 601, 801]
    R = 5

    for k in depths:
        results['depth_scan'][k] = []

        for r in range(R):
            print(f"  k={k}, run {r+1}/{R}...", end=" ", flush=True)

            c = pcvl.Circuit(2)
            for i in range(k):
                c.add(0, pcvl.BS())
                c.add((0,), pcvl.PS(np.pi/4))
            c.add(0, pcvl.BS())

            result, status = submit_and_wait(c, pcvl.BasicState([1, 1]))

            if result:
                results['depth_scan'][k].append(result)
                print("OK")
            else:
                results['depth_scan'][k].append(None)
                print(f"FAILED: {status}")

            time.sleep(1)

        # Save after each depth
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  k={k} complete. Progress saved.\n")

    # === Summary ===
    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"End: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
