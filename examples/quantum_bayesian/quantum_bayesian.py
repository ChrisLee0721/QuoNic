"""Quantum Bayesian Inference / 量子贝叶斯推断

Quantum circuit for Bayesian posterior estimation using amplitude encoding
and controlled rotations. The posterior is computed by the quantum circuit,
not by classical pre-computation.

量子电路实现贝叶斯后验估计，使用振幅编码和受控旋转。后验由量子电路计算，
而非经典预计算。

## Application / 应用场景
- Hypothesis testing (假设检验)
- Decision making (决策)
- Medical diagnosis (医学诊断)

## Output / 输出
Posterior probabilities sampled from quantum circuit.
量子电路采样的后验概率。"""

from quonic.algorithms import quantum_bayesian

result = quantum_bayesian(prior_h0=0.5, likelihood_h0=0.8, likelihood_h1=0.3)
print(f"Quantum posterior P(H0|data): {result.value:.4f}")
print(f"Classical posterior:          {result.metadata['classical_posterior']:.4f}")
print(f"Counts: {result.metadata['counts']}")
