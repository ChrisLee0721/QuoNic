"""GPU acceleration / GPU 加速

GPU acceleration / GPU 加速

## Application / 应用场景
- Quantum computing (量子计算)
- Algorithm demonstration (算法演示)
- Educational (教学)

## Output / 输出
See code comments for output explanation.
参见代码注释了解输出说明。"""

import math
import time

from quonic import creg, cwhile, qgate, reset
from quonic.backends import get_backend
from quonic.gates import CCX, CX, H, Ry
from quonic.scheduler import circuit_features, recommend_backend_gpu
from quonic.stack import current_circuit


def _take():
    c = current_circuit()
    reset()
    return c


def demo_direct_gpu():
    """Direct GPU execution via method="gpu"."""
    print("=" * 60)
    print("1. Direct GPU execution")
    print("=" * 60)

    # Build a Bell circuit
    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    circuit = _take()

    # Run on different backends with method="gpu"
    for backend_name in ["qulacs", "cupy"]:
        try:
            be = get_backend(backend_name)
            t0 = time.time()
            result = be.run(circuit, shots=1024, method="gpu")
            elapsed = time.time() - t0
            print(f"  {backend_name:12s} GPU: {result.counts}  ({elapsed:.3f}s)")
        except NotImplementedError:
            print(f"  {backend_name:12s} GPU: not supported")
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"  {backend_name:12s} GPU: error: {e}")

    # Compare with CPU
    be = get_backend("qulacs")
    t0 = time.time()
    result = be.run(circuit, shots=1024)
    elapsed = time.time() - t0
    print(f"  {'qulacs':12s} CPU: {result.counts}  ({elapsed:.3f}s)")
    print()


def demo_smart_scheduling():
    """Smart scheduling via recommend_backend_gpu()."""
    print("=" * 60)
    print("2. Smart scheduling")
    print("=" * 60)

    # Case 1: Small entangled circuit
    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    qgate(CCX, 0, 1, 2)
    circuit = _take()
    feats = circuit_features(circuit)
    rec = recommend_backend_gpu(feats)
    print(f"  GHZ-3 (n={feats['n']}, entanglement={feats['entanglement']}):")
    print(f"    → {rec.backend} ({rec.method})")

    # Case 2: Large low-entanglement circuit
    reset()
    for i in range(25):
        qgate(Ry(0.1 * math.pi), i)
    circuit = _take()
    feats = circuit_features(circuit)
    rec = recommend_backend_gpu(feats)
    print(f"  Rotation-25 (n={feats['n']}, entanglement={feats['entanglement']}):")
    print(f"    → {rec.backend} ({rec.method})")

    # Case 3: Circuit with classical control flow
    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(H, 0)
        flag.measure(0)
    static = loop.groverize()
    feats = circuit_features(static)
    rec = recommend_backend_gpu(feats)
    print(f"  Grover (n={feats['n']}, has_ctrl={feats['has_ctrl']}):")
    print(f"    → {rec.backend} ({rec.method})")
    print()


def demo_cupy_fallback():
    """CuPy fallback when native GPU is unavailable."""
    print("=" * 60)
    print("3. CuPy fallback")
    print("=" * 60)

    # Build a GHZ-5 circuit
    reset()
    qgate(H, 0)
    for i in range(4):
        qgate(CX, i, i + 1)
    circuit = _take()

    # Run on qulacs (will fallback to CuPy if no GPU)
    be = get_backend("qulacs")
    try:
        t0 = time.time()
        result = be.run(circuit, shots=1024, method="gpu")
        elapsed = time.time() - t0
        print(f"  qulacs GPU: {result.counts}  ({elapsed:.3f}s)")
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"  qulacs GPU: error: {e}")

    # Run on CuPy directly
    be = get_backend("cupy")
    try:
        t0 = time.time()
        result = be.run(circuit, shots=1024, method="gpu")
        elapsed = time.time() - t0
        print(f"  cupy GPU:   {result.counts}  ({elapsed:.3f}s)")
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"  cupy GPU:   error: {e}")

    # Compare with CPU
    be = get_backend("native")
    t0 = time.time()
    result = be.run(circuit, shots=1024)
    elapsed = time.time() - t0
    print(f"  native CPU: {result.counts}  ({elapsed:.3f}s)")
    print()


def demo_error_handling():
    """Error handling for unsupported GPU backends."""
    print("=" * 60)
    print("4. Error handling")
    print("=" * 60)

    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    circuit = _take()

    # Try GPU on backends that don't support it
    for backend_name in ["cirq", "native"]:
        try:
            be = get_backend(backend_name)
            be.run(circuit, shots=1024, method="gpu")
            print(f"  {backend_name}: unexpected success")
        except NotImplementedError as e:
            print(f"  {backend_name}: correctly rejected — {e}")
    print()


def main():
    print("QuoNic GPU Acceleration Demo")
    print()

    demo_direct_gpu()
    demo_smart_scheduling()
    demo_cupy_fallback()
    demo_error_handling()

    print("=" * 60)
    print("Done! GPU backends provide hardware-accelerated simulation.")
    print("Use method='gpu' or recommend_backend_gpu() for automatic selection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
