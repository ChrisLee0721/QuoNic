"""ODE Solver / ODE 求解器

Quantum algorithm for ordinary differential equations.
常微分方程的量子算法。

## Application / 应用场景
- Physics simulation (物理模拟)
- Engineering (工程)
- Dynamics (动力学)

## Output / 输出
Solution trajectory.
解轨迹。"""

from quonic.algorithms import quantum_ode

result = quantum_ode(shots=1024)
print(result.counts)
