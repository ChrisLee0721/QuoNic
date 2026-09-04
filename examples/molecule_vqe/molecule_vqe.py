"""Molecular ground state / 分子基态

Compute ground state energy of molecules.
计算分子的基态能量。

## Application / 应用场景
- Drug discovery (药物发现)
- Material design (材料设计)
- Chemical reactions (化学反应)

## Output / 输出
Ground state energy of molecule.
分子的基态能量。"""

from quonic.algorithms import molecule_vqe

result = molecule_vqe(maxiter=200)
print(f"Ground state energy: {result.value}")
