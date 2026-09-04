"""PDE Solver / PDE 求解器

Quantum algorithm for partial differential equations.
偏微分方程的量子算法。

## Application / 应用场景
- Fluid dynamics (流体力学)
- Heat transfer (热传导)
- Electromagnetics (电磁学)

## Output / 输出
Solution field.
解场。"""

from quonic.algorithms import quantum_pde

result = quantum_pde(shots=1024)
print(result.counts)
