"""
Belenos deep depths experiment: k=50 to k=801.
Re-run the depths that failed due to hardware maintenance.
"""

import perceval as pcvl
import numpy as np
import time
import json
from datetime import datetime
from perceval.algorithm import Sampler

TOKEN = "YOUR_TOKEN_HERE"
OUTPUT_FILE = "belenos_deep_depths.json"

BACKEND = "qpu:belenos"

# Deep depths that failed previously
DEPTHS = [50, 75, 100, 150, 201, 301, 401, 501, 601, 801]
REPS = 10
SHOTS = 5000

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
        'depths': DEPTHS,
        'reps': REPS,
        'shots': SHOTS,
        'note': 'Deep depths re-run after hardware maintenance',
        'depth_scan': {},
    }

    print(f"=== Belenos Deep Depths Experiment ===")
    print(f"Backend: {BACKEND} (HARDWARE)")
    print(f"Depths: {DEPTHS}")
    print(f"Reps: {REPS}, Shots: {SHOTS}")
    print(f"Start: {datetime.now().isoformat()}\n")

    for k in DEPTHS:
        results['depth_scan'][k] = []

        for r in range(REPS):
            print(f"  k={k}, run {r+1}/{REPS}...", end=" ", flush=True)

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

    # Summary
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
