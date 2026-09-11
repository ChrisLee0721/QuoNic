# QuoNic Decision Module Design

## Overview

Implement the paper's decision boundary framework as a QuoNic module for automatic compilation strategy selection.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    QuoNic Compiler                       │
├─────────────────────────────────────────────────────────┤
│  Decision Module                                        │
│  ├── Platform Database                                  │
│  ├── Circuit Analyzer                                   │
│  ├── Decision Boundary Calculator                       │
│  └── Strategy Selector                                  │
├─────────────────────────────────────────────────────────┤
│  Compilation Backends                                   │
│  ├── Unitary Compiler                                   │
│  └── Dynamic Compiler                                   │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Platform Database

```python
# quonic/decision/platforms.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class PlatformParams:
    """Platform-specific parameters for decision boundary."""
    name: str
    F_gate: float          # Gate fidelity
    F_meas: float          # Measurement fidelity
    T_fb: float            # Feedback latency (μs)
    T2: float              # Decoherence time (μs)
    F_eff: Optional[float] = None  # Effective path fidelity
    
    @property
    def T(self) -> float:
        """Measurement tax: T = F_meas * exp(-T_fb/T2)."""
        return self.F_meas * np.exp(-self.T_fb / self.T2)
    
    @property
    def threshold(self) -> float:
        """Decision boundary threshold: |ln T| / |ln F_eff|."""
        F_eff = self.F_eff or self.F_gate
        return abs(np.log(self.T)) / abs(np.log(F_eff))

# Platform database
PLATFORMS = {
    "ibm_heron": PlatformParams(
        name="IBM Heron",
        F_gate=0.995,
        F_meas=0.99,
        T_fb=34,
        T2=200,
    ),
    "ibm_marrakesh": PlatformParams(
        name="IBM ibm_marrakesh",
        F_gate=0.995,
        F_meas=0.99,
        T_fb=34,
        T2=200,
    ),
    "iqm_garnet": PlatformParams(
        name="IQM Garnet",
        F_gate=0.9989,
        F_meas=0.98,  # Estimated
        T_fb=50,      # Estimated
        T2=100,       # Estimated
    ),
    "iqm_emerald": PlatformParams(
        name="IQM Emerald",
        F_gate=0.9995,
        F_meas=0.98,
        T_fb=50,
        T2=100,
    ),
    "qi_tuna17": PlatformParams(
        name="Quantum Inspire Tuna-17",
        F_gate=0.99763,
        F_meas=0.95,
        T_fb=100,
        T2=50,
    ),
    "quandela_belenos": PlatformParams(
        name="Quandela Belenos",
        F_gate=None,  # Photonic - different model
        F_meas=None,
        T_fb=None,
        T2=None,
    ),
}
```

### 2. Circuit Analyzer

```python
# quonic/decision/analyzer.py

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class CircuitAnalysis:
    """Analysis results for a quantum circuit."""
    n_qubits: int
    depth: int                    # Transpiled depth
    n_conditional: int            # Number of conditional operations
    c: float                      # Compilation overhead factor
    d: int                        # Bare unitary depth
    has_midcircuit_measurement: bool
    control_structures: List[str]  # for, if, switch, while
    
def analyze_circuit(circuit, backend=None) -> CircuitAnalysis:
    """Analyze circuit for decision boundary calculation."""
    
    # Detect control structures
    control_structures = detect_control_structures(circuit)
    
    # Calculate compilation overhead
    c = estimate_compilation_overhead(circuit, backend)
    
    # Get bare unitary depth
    d = get_bare_depth(circuit)
    
    # Check for mid-circuit measurement
    has_mcm = has_midcircuit_measurement(circuit)
    
    # Count conditional operations
    n_conditional = count_conditional_operations(circuit)
    
    return CircuitAnalysis(
        n_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        n_conditional=n_conditional,
        c=c,
        d=d,
        has_midcircuit_measurement=has_mcm,
        control_structures=control_structures,
    )

def detect_control_structures(circuit) -> List[str]:
    """Detect classical control structures in circuit."""
    structures = []
    
    for instruction in circuit.data:
        # Check for mid-circuit measurement + conditional
        if is_conditional_gate(instruction):
            structures.append("if")
        # Check for loops (heuristic)
        if is_loop_pattern(instruction):
            structures.append("while")
    
    return list(set(structures))

def estimate_compilation_overhead(circuit, backend) -> float:
    """Estimate compilation overhead factor c."""
    # Native CNOT: c = 1
    # With SWAP routing: c > 1
    # Depends on connectivity and qubit mapping
    
    if backend is None:
        return 1.0
    
    # Estimate based on circuit depth and connectivity
    physical_depth = transpile_depth(circuit, backend)
    logical_depth = logical_circuit_depth(circuit)
    
    return physical_depth / logical_depth if logical_depth > 0 else 1.0
```

### 3. Decision Boundary Calculator

```python
# quonic/decision/boundary.py

from dataclasses import dataclass
from enum import Enum
import numpy as np

class Strategy(Enum):
    """Compilation strategy."""
    UNITARY = "unitary"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"

@dataclass
class DecisionResult:
    """Decision boundary calculation result."""
    strategy: Strategy
    threshold: float
    (c_minus_1)d: float
    advantage: float  # Expected fidelity advantage
    confidence: str   # "high", "medium", "low"
    explanation: str

def calculate_decision(
    analysis: CircuitAnalysis,
    platform: PlatformParams,
) -> DecisionResult:
    """Calculate optimal compilation strategy."""
    
    # Get platform threshold
    threshold = platform.threshold
    
    # Calculate (c-1)*d
    cd = (analysis.c - 1) * analysis.d
    
    # Decision boundary
    if cd < threshold:
        strategy = Strategy.UNITARY
        advantage = calculate_advantage(analysis, platform)
        confidence = "high" if cd < threshold * 0.5 else "medium"
        explanation = (
            f"Unitary compilation recommended. "
            f"(c-1)d = {cd:.1f} < threshold = {threshold:.1f}. "
            f"Expected advantage: {advantage:.1f}x"
        )
    else:
        strategy = Strategy.DYNAMIC
        advantage = 1.0
        confidence = "high" if cd > threshold * 2 else "medium"
        explanation = (
            f"Dynamic execution recommended. "
            f"(c-1)d = {cd:.1f} > threshold = {threshold:.1f}."
        )
    
    return DecisionResult(
        strategy=strategy,
        threshold=threshold,
        (c_minus_1)d=cd,
        advantage=advantage,
        confidence=confidence,
        explanation=explanation,
    )

def calculate_advantage(
    analysis: CircuitAnalysis,
    platform: PlatformParams,
) -> float:
    """Calculate expected fidelity advantage."""
    
    N = analysis.n_conditional
    T = platform.T
    F_eff = platform.F_eff or platform.F_gate
    
    # R_total = (F_eff^((c-1)*d) / T)^N
    advantage = (F_eff ** ((analysis.c - 1) * analysis.d) / T) ** N
    
    return advantage
```

### 4. Strategy Selector

```python
# quonic/decision/selector.py

def select_strategy(
    circuit,
    platform_name: str,
    backend=None,
    user_override=None,
) -> Tuple[Strategy, DecisionResult]:
    """Select optimal compilation strategy."""
    
    # Get platform parameters
    platform = PLATFORMS.get(platform_name)
    if platform is None:
        raise ValueError(f"Unknown platform: {platform_name}")
    
    # Analyze circuit
    analysis = analyze_circuit(circuit, backend)
    
    # Calculate decision
    decision = calculate_decision(analysis, platform)
    
    # Apply user override if specified
    if user_override is not None:
        decision.strategy = user_override
        decision.explanation += f" (User override: {user_override})"
    
    return decision.strategy, decision
```

## qshow() Enhancement

```python
# quonic/visualization/decision.py

def qshow_decision(
    circuit,
    platform_name: str,
    backend=None,
):
    """Enhanced qshow with decision boundary analysis."""
    
    # Original qshow functionality
    qshow(circuit)
    
    # Decision boundary analysis
    strategy, decision = select_strategy(circuit, platform_name, backend)
    platform = PLATFORMS[platform_name]
    
    print("\n" + "="*60)
    print("Decision Boundary Analysis")
    print("="*60)
    print(f"Platform: {platform.name}")
    print(f"Measurement tax T: {platform.T:.4f}")
    print(f"Threshold: {decision.threshold:.1f}")
    print(f"(c-1)*d: {decision.(c_minus_1)d:.1f}")
    print(f"Strategy: {decision.strategy.value.upper()}")
    print(f"Expected advantage: {decision.advantage:.1f}x")
    print(f"Confidence: {decision.confidence}")
    print(f"\n{decision.explanation}")
    
    # Visualize decision boundary
    plot_decision_boundary(platform, decision)
```

## Platform Benchmarking

```python
# quonic/benchmark/platform_comparison.py

def benchmark_platforms(
    circuit,
    platform_names: List[str],
) -> pd.DataFrame:
    """Compare compilation strategies across platforms."""
    
    results = []
    
    for name in platform_names:
        strategy, decision = select_strategy(circuit, name)
        platform = PLATFORMS[name]
        
        results.append({
            "platform": platform.name,
            "T": platform.T,
            "threshold": platform.threshold,
            "strategy": strategy.value,
            "advantage": decision.advantage,
            "confidence": decision.confidence,
        })
    
    return pd.DataFrame(results)

def plot_platform_comparison(df: pd.DataFrame):
    """Visualize platform comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Measurement tax
    axes[0].bar(df["platform"], df["T"])
    axes[0].set_ylabel("Measurement Tax (T)")
    axes[0].set_title("Platform Measurement Tax")
    
    # Plot 2: Threshold
    axes[1].bar(df["platform"], df["threshold"])
    axes[1].set_ylabel("Decision Threshold")
    axes[1].set_title("Platform Decision Threshold")
    
    # Plot 3: Expected advantage
    axes[2].bar(df["platform"], df["advantage"])
    axes[2].set_ylabel("Expected Advantage (x)")
    axes[2].set_title("Unitary Advantage")
    
    plt.tight_layout()
    return fig
```

## Usage Examples

### Example 1: Basic Decision

```python
from quonic.decision import select_strategy, PLATFORMS

# Create circuit
circuit = create_grover_circuit(n_qubits=3, n_solutions=1)

# Select strategy
strategy, decision = select_strategy(circuit, "ibm_heron")

print(f"Recommended: {strategy.value}")
print(f"Advantage: {decision.advantage:.1f}x")
```

### Example 2: Platform Comparison

```python
from quonic.benchmark import benchmark_platforms

# Compare across platforms
platforms = ["ibm_heron", "iqm_garnet", "qi_tuna17"]
df = benchmark_platforms(circuit, platforms)

print(df)
plot_platform_comparison(df)
```

### Example 3: Enhanced qshow

```python
from quonic.visualization import qshow_decision

# Show circuit with decision analysis
qshow_decision(circuit, "ibm_heron")
```

## Testing

```python
# tests/test_decision.py

def test_decision_boundary():
    """Test decision boundary calculation."""
    
    # Unitary case: (c-1)*d < threshold
    analysis = CircuitAnalysis(
        n_qubits=3,
        depth=100,
        n_conditional=10,
        c=1.0,
        d=1,
        has_midcircuit_measurement=False,
        control_structures=["if"],
    )
    
    platform = PLATFORMS["ibm_heron"]
    decision = calculate_decision(analysis, platform)
    
    assert decision.strategy == Strategy.UNITARY
    assert decision.advantage > 1.0

def test_dynamic_case():
    """Test dynamic case: (c-1)*d > threshold."""
    
    analysis = CircuitAnalysis(
        n_qubits=3,
        depth=1000,
        n_conditional=100,
        c=10.0,
        d=100,
        has_midcircuit_measurement=True,
        control_structures=["while"],
    )
    
    platform = PLATFORMS["ibm_heron"]
    decision = calculate_decision(analysis, platform)
    
    assert decision.strategy == Strategy.DYNAMIC
```

## Integration with QuoNic

### Step 1: Add decision module

```
quonic/
├── decision/
│   ├── __init__.py
│   ├── platforms.py
│   ├── analyzer.py
│   ├── boundary.py
│   └── selector.py
├── benchmark/
│   ├── __init__.py
│   └── platform_comparison.py
└── visualization/
    ├── __init__.py
    └── decision.py
```

### Step 2: Update compile() function

```python
# quonic/compiler.py

def compile(circuit, platform=None, strategy="auto"):
    """Compile circuit with optional strategy selection."""
    
    if strategy == "auto" and platform:
        # Auto-select strategy
        selected_strategy, decision = select_strategy(circuit, platform)
        
        if selected_strategy == Strategy.UNITARY:
            return compile_unitary(circuit)
        else:
            return compile_dynamic(circuit)
    else:
        # User-specified strategy
        if strategy == "unitary":
            return compile_unitary(circuit)
        else:
            return compile_dynamic(circuit)
```

### Step 3: Update qshow()

```python
# quonic/visualization/qshow.py

def qshow(circuit, platform=None, show_decision=False):
    """Show circuit with optional decision analysis."""
    
    # Original functionality
    _show_circuit(circuit)
    
    # Optional decision analysis
    if show_decision and platform:
        qshow_decision(circuit, platform)
```

## Future Extensions

1. **Dynamic calibration**: Measure T and F_eff on platform
2. **Machine learning**: Predict optimal strategy from circuit features
3. **Multi-objective optimization**: Balance fidelity, depth, and time
4. **Quantum error correction**: Extend to measurement-free QEC

## References

- Paper: "Static Unitary Compilation Outperforms Dynamic Feedback on NISQ Hardware"
- Decision boundary: (c-1)d < |ln T| / |ln F_eff|
- Measurement tax: T = F_meas * exp(-T_fb/T2)
