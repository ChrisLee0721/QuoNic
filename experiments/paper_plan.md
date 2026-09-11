# Unitary-First Compilation — Nature 论文计划

## 标题
"Pure Unitary Circuits Outperform Dynamic Quantum Programs on Noisy Hardware"

## Abstract (~150 words)
- Quantum computing relies on classical control layers (mid-circuit measurement + feedback)
- We show this is unnecessary: all structured control flow compiles to pure unitary circuits
- On IBM's 156-qubit Heron processor, pure unitary achieves 9.49x higher fidelity at 17 conditional operations
- The advantage grows monotonically with circuit depth
- Feedback latency must improve 4x to break even; even then, pure unitary wins at scale
- This challenges the "quantum data, classical control" paradigm since Deutsch 1985

## Main Text (~4000 words)

### Introduction (800 words)
- Quantum computing's dirty secret: every quantum program has a classical brain
- Deutsch QTM (1985): classical state drives unitary selection
- IBM dynamic circuits, OpenQASM 3: industry investing heavily in classical control
- The cost: mid-circuit measurement destroys coherence, adds noise
- Our insight: for programs whose semantics are final measurement, classical control is eliminable
- We prove: for/if/switch/while all compile to pure unitary
- We demonstrate: on real hardware, pure unitary wins by up to 9.49x

### Results (1500 words)

#### The Compilation Framework
- for → static unrolling (already unitary)
- if → controlled gate (CNOT, Toffoli)
- switch → multiplexer unitary
- while → amplitude amplification (groverize)
- Key insight: mid-circuit measurement is not a quantum operation — it's a classical bridge that can be replaced

#### Experimental Setup
- IBM ibm_marrakesh (156-qubit Heron)
- Program: N conditional operations (if q0==1: flip q[i+1])
- Two implementations: dynamic (measure+branch) vs pure unitary (CX)
- Fixed qubit layout, DD off, Twirling off, 4000 shots
- Same hardware, same calibration, same physical qubits

#### Scaling Results
- N=1..17: monotonic advantage growth
- N=1: 96.9% vs 92.4% (+4.5%)
- N=9: 70.4% vs 34.6% (+35.8%)
- N=17: 38.9% vs 4.1% (+34.8%, 9.49x)
- N=18..23: dynamic drops below 1%, pure unitary still 17-33%

#### Why Pure Unitary Wins
- Dynamic: each step loses ~17% fidelity (measurement + feedback + gate)
- Pure unitary: only gate noise, no measurement penalty
- Feedback latency analysis: current ~34μs, needs <8μs to break even at N=17
- Even at zero latency, dynamic has inherent F_gate × F_measure ≈ 0.985 per step

### Discussion (1200 words)

#### Implications for Quantum Computing
- The "quantum data, classical control" paradigm is a historical accident, not a necessity
- Industry is investing in the wrong direction (mid-circuit measurement hardware)
- Pure unitary is simpler, cheaper, more reliable

#### Optimization Space
- Current: naive CX sequence, no circuit optimization
- Qubit routing: centering control qubit reduces depth ~15%
- Native multi-qubit gates, pulse-level compilation: further improvements
- We test the worst case; optimization only improves pure unitary

#### Limitations
- Tested on one hardware platform (IBM Heron)
- Simple program structure (conditional X gates)
- Does not address error correction (separate concern)
- Groverize depth is O(N²) with naive routing; O(N) with optimized routing

#### Broader Impact
- Quantum programming languages should prioritize unitary-first design
- Hardware vendors should focus on gate fidelity, not mid-circuit measurement speed
- This is a paradigm shift: from "quantum + classical" to "quantum only"

### Methods (500 words)
- Hardware specifications
- Circuit construction details
- Statistical methods (confidence intervals, TVD)
- Data availability

## Figures

### Figure 1: The Compilation Framework
- Side-by-side: dynamic circuit vs pure unitary
- Show how if/while/switch compile to controlled gates

### Figure 2: Scaling Curve (MAIN FIGURE)
- X: N (number of conditional operations)
- Y: Probability of correct outcome (%)
- Two curves: Dynamic (red, decaying) vs Groverize (blue, slowly decaying)
- Shaded region: groverize advantage
- Inset: relative advantage (x multiplier)

### Figure 3: Fidelity Analysis
- Per-step fidelity comparison
- Break-even feedback latency vs N

### Figure 4: Dynamic Death
- N=17..23 extension showing dynamic → 0%
- Groverize still 17-33%

## Supplementary Information
- Full N=1..23 data table
- Qubit layout details
- Statistical analysis
- Additional hardware runs

## Key Messages for Nature
1. "Classical control in quantum computing is eliminable" — big claim
2. "9.49x advantage on real hardware" — concrete evidence
3. "Industry is investing in the wrong direction" — provocative
4. "This challenges 40 years of quantum computing theory" — historical significance

## Timeline
- [ ] Write Introduction + Results (today)
- [ ] Write Discussion + Methods
- [ ] Create figures
- [ ] Internal review
- [ ] Submit to Nature
