# Paper Data: GCIQA Error Modeling

## Essential Files

| File | Description |
|------|-------------|
| `error_modeling_results.json` | 195 metal sites, 3/4/5-bit error statistics |
| `constraint_density_results.json` | 3-dist vs 6-dist comparison |
| `angle_oracle_hardware_fixed_20260828.json` | Hardware validation (WK_C180) |
| `failure_log.md` | Bug history (NAA artifact, oracle fixes) |

## Scripts

| File | Description |
|------|-------------|
| `error_modeling_fast.py` | NumPy vectorized + multiprocessing |
| `error_modeling_gpu.py` | PyTorch CUDA version |
| `constraint_density_gpu.py` | Constraint density test (GPU) |
| `benchmark_classical_advantage.py` | GCIQA vs force fields |

## Key Results

- 4-bit encoding: 0.070 ± 0.026 Å error (195 sites, 10 metals)
- 3 distances sufficient (6 distances don't improve accuracy)
- Tolerance doesn't affect error (only encoding resolution matters)
- GPU 9x speedup for large search spaces (16.7M states)
