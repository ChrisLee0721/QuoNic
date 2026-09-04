"""Quantum Reinforcement Learning / 量子强化学习

Quantum agent learning in classical environment.
经典环境中的量子智能体学习。

## Application / 应用场景
- Game playing (游戏)
- Robotics (机器人)
- Optimization (优化)

## Output / 输出
Learned policy.
学习到的策略。"""

from quonic.algorithms import qrl

result = qrl(n_episodes=10)
print(result.counts)
