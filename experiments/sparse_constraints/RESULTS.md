# Sparse Constraint Benchmark Results

## Summary

GCIQA's classical mode can solve sparse constraint problems for small systems (3-4 points),
but fails for larger systems (5+ points) due to exponential search space.

## Test Setup

- **System**: Chain of N points with bond length ~2.0Å
- **Constraints**: Chain connectivity (N-1 constraints) + random extra constraints
- **Noise**: ±1.0Å on all distance constraints
- **Resolution**: 0.667-0.645Å per coordinate step (4-5 bits per coord)
- **Search**: 10000 shots, 15 iterations

## Results

| Points | Chain Only | + 1 Extra | + 2 Extra | + 3 Extra | + 4 Extra |
|--------|-----------|-----------|-----------|-----------|-----------|
| 3 | 1.33Å OK | 1.58Å OK | 1.47Å OK | - | - |
| 4 | 1.57Å OK | - | 1.46Å OK | - | 1.36Å OK |
| 5 | 1.39Å OK | - | - | FAIL | - |

## Key Findings

1. **GCIQA works for 3-4 points** with sparse constraints (RMSD 1.3-1.6Å)
2. **Fails at 5+ points** with extra constraints — search space too large
3. **Coordinate range narrowing** helps (5 points chain-only: 2.11Å → 1.39Å)
4. **Incremental placement** helps for chain constraints but not for non-sequential constraints
5. **Discrete grid resolution** (0.667Å) limits precision

## Why It Fails at 5+ Points

With 5 points and 4 bits per coordinate:
- Search space: 2^(5×3×4) = 2^60 ≈ 10^18 states
- With 10000 shots: sample 10^4/10^18 = 10^-14 of space
- Probability of finding valid conformation: essentially zero

## Implications

- **Classical mode**: Random sampling in exponential space. Works for small systems only.
- **Quantum mode**: Grover's algorithm provides quadratic speedup. Would help for larger systems.
- **Real value**: GCIQA's advantage is quantum search, not classical random sampling.

## Recommendations

1. For small systems (3-4 points): GCIQA classical mode works
2. For larger systems: Need quantum mode or smarter search strategies
3. For the Zn²⁺ benchmark: GCIQA works because metal coordination has predefined geometry
4. For general sparse constraints: GCIQA's classical mode is not competitive

## Conclusion

The sparse constraint benchmark reveals GCIQA's fundamental limitation in classical mode:
it's random sampling in an exponential search space. The quantum advantage (Grover's algorithm)
is the real value proposition, but it can't be demonstrated without a quantum computer.
