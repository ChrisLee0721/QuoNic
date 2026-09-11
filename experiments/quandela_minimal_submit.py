"""Minimal HOM circuit — local + remote (sim:slos) on Quandela Cloud."""

import perceval as pcvl
from perceval import BS, BasicState, Processor, RemoteProcessor
from perceval.algorithm import Sampler

TOKEN = "YOUR_TOKEN_HERE"

# Build minimal circuit: Hong-Ou-Mandel effect
input_state = BasicState("|1,1>")
circuit = BS()

print("=== Circuit ===")
print(f"  Modes: {circuit.m}")
print(f"  Input: {input_state}")

# --- Local simulation ---
print("\n=== Local SLOS Simulation ===")
processor = Processor("SLOS", circuit)
processor.with_input(input_state)
processor.min_detected_photons_filter(1)

sampler = Sampler(processor)
samples = sampler.sample_count(10000)['results']
probs = sampler.probs()['results']
print(f"Samples (10k): {samples}")
print(f"Probabilities: {probs}")

# --- Remote simulation on Quandela Cloud ---
print("\n=== Remote sim:slos on Quandela Cloud ===")
try:
    remote = RemoteProcessor("sim:slos", TOKEN, noise=None)
    remote.add(0, circuit)
    remote.with_input(input_state)
    remote.min_detected_photons_filter(1)

    sampler_r = Sampler(remote, max_shots_per_call=10_000)
    remote_samples = sampler_r.sample_count(100)['results']
    remote_probs = sampler_r.probs()['results']
    print(f"Remote Samples: {remote_samples}")
    print(f"Remote Probabilities: {remote_probs}")
except Exception as e:
    print(f"Remote connection failed: {e}")
    print("Tip: verify your token is active at https://cloud.quandela.com")
