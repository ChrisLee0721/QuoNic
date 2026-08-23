"""Update ALL examples with bilingual documentation."""

import glob
import os

# Bilingual docs for each example category
DOCS = {
    # Quantum algorithms
    "amplitude_amplification": {
        "problem": "Amplify probability of target state / 放大目标态概率",
        "desc": "Like Grover but with custom state preparation. Boosts success probability.\n类似 Grover 但支持自定义态制备。提升成功概率。",
        "app": "- Quantum algorithms (量子算法)\n- State preparation (态制备)\n- Error mitigation (错误缓解)",
        "output": "Target state probability amplified from p to ~1.\n目标态概率从 p 放大到 ~1。",
    },
    "amplitude_estimation": {
        "problem": "Estimate success probability / 估计成功概率",
        "desc": "Quantum algorithm to estimate the amplitude of a marked state.\n量子算法估计标记态的振幅。",
        "app": "- Monte Carlo integration (蒙特卡洛积分)\n- Risk analysis (风险分析)\n- Option pricing (期权定价)",
        "output": "Estimated amplitude with quadratic speedup over classical.\n估计振幅，相比经典有二次加速。",
    },
    "bb84": {
        "problem": "Quantum key distribution / 量子密钥分发",
        "desc": "BB84 protocol for secure key exchange using quantum mechanics.\nBB84 协议利用量子力学实现安全密钥交换。",
        "app": "- Secure communication (安全通信)\n- Quantum cryptography (量子密码学)\n- Key distribution (密钥分发)",
        "output": "Shared secret key between Alice and Bob.\nAlice 和 Bob 共享的密钥。",
    },
    "bernstein_vazirani": {
        "problem": "Find hidden bitstring / 找到隐藏比特串",
        "desc": "Find secret s in f(x) = s·x mod 2. One query suffices.\n在 f(x) = s·x mod 2 中找到秘密 s。一次查询即可。",
        "app": "- Oracle problems (预言机问题)\n- Cryptography (密码学)\n- Learning parity (学习奇偶性)",
        "output": "All shots give the hidden string s.\n所有测量结果给出隐藏串 s。",
    },
    "bit_flip_code": {
        "problem": "Correct bit-flip errors / 纠正比特翻转错误",
        "desc": "3-qubit code corrects single bit-flip errors.\n3 比特码纠正单个比特翻转错误。",
        "app": "- Quantum error correction (量子纠错)\n- Fault-tolerant computing (容错计算)\n- NISQ algorithms (NISQ 算法)",
        "output": "Corrected logical state despite physical errors.\n尽管有物理错误，纠正后的逻辑态。",
    },
    "color_code": {
        "problem": "Color code error correction / 颜色码纠错",
        "desc": "Topological error correction code with transversal gates.\n具有横向门的拓扑纠错码。",
        "app": "- Fault-tolerant quantum computing (容错量子计算)\n- Topological codes (拓扑码)\n- Quantum memory (量子存储)",
        "output": "Encoded logical qubit with error protection.\n具有错误保护的编码逻辑比特。",
    },
    "deutsch_jozsa": {
        "problem": "Constant or balanced function? / 常数还是平衡函数？",
        "desc": "Determine if f is constant or balanced in one query.\n一次查询确定 f 是常数还是平衡函数。",
        "app": "- Oracle complexity (预言机复杂度)\n- Quantum advantage (量子优势)\n- Function analysis (函数分析)",
        "output": "All zeros = constant, anything else = balanced.\n全零 = 常数，其他 = 平衡。",
    },
    "discrete_log": {
        "problem": "Discrete logarithm / 离散对数",
        "desc": "Find x such that a^x = b mod p.\n找到 x 使得 a^x = b mod p。",
        "app": "- Cryptography (密码学)\n- RSA breaking (RSA 破解)\n- Key exchange (密钥交换)",
        "output": "The discrete logarithm x.\n离散对数 x。",
    },
    "dqaoa": {
        "problem": "Dynamic QAOA / 动态 QAOA",
        "desc": "Adaptive layer QAOA that adds layers until convergence.\n自适应层 QAOA，添加层直到收敛。",
        "app": "- Combinatorial optimization (组合优化)\n- MaxCut (最大割)\n- Scheduling (调度)",
        "output": "Approximate optimal solution.\n近似最优解。",
    },
    "dynamics_simulation": {
        "problem": "Quantum dynamics simulation / 量子动力学模拟",
        "desc": "Simulate time evolution of quantum systems.\n模拟量子系统的时间演化。",
        "app": "- Quantum chemistry (量子化学)\n- Material science (材料科学)\n- Condensed matter (凝聚态)",
        "output": "Evolved state after time t.\n时间 t 后的演化态。",
    },
    "e91": {
        "problem": "E91 key distribution / E91 密钥分发",
        "desc": "E91 protocol using entangled pairs and Bell inequality.\nE91 协议使用纠缠对和 Bell 不等式。",
        "app": "- Quantum key distribution (量子密钥分发)\n- Entanglement verification (纠缠验证)\n- Device-independent QKD (设备无关 QKD)",
        "output": "Shared secret key with security verification.\n带有安全验证的共享密钥。",
    },
    "elliptic_curve": {
        "problem": "Elliptic curve quantum algorithm / 椭圆曲线量子算法",
        "desc": "Quantum approach to elliptic curve discrete log.\n椭圆曲线离散对数的量子方法。",
        "app": "- Post-quantum cryptography (后量子密码学)\n- Blockchain security (区块链安全)\n- Digital signatures (数字签名)",
        "output": "Approximate solution to ECDLP.\nECDLP 的近似解。",
    },
    "ft_gate": {
        "problem": "Fault-tolerant gates / 容错门",
        "desc": "Gates implemented with error detection/correction.\n带有错误检测/纠正的门实现。",
        "app": "- Fault-tolerant computing (容错计算)\n- Quantum error correction (量子纠错)\n- Logical gates (逻辑门)",
        "output": "Logically encoded state with error protection.\n具有错误保护的逻辑编码态。",
    },
    "hadamard_test": {
        "problem": "Estimate Re(<ψ|U|ψ>) / 估计 Re(<ψ|U|ψ>)",
        "desc": "Primitive for inner product estimation.\n内积估计的基本操作。",
        "app": "- Quantum algorithms (量子算法)\n- State overlap (态重叠)\n- Expectation values (期望值)",
        "output": "Probability of |0⟩ encodes the real part.\n|0⟩ 的概率编码实部。",
    },
    "hamiltonian_simulation": {
        "problem": "Hamiltonian simulation / 哈密顿量模拟",
        "desc": "Simulate e^{-iHt} for given Hamiltonian.\n模拟给定哈密顿量的 e^{-iHt}。",
        "app": "- Quantum chemistry (量子化学)\n- Material science (材料科学)\n- Quantum simulation (量子模拟)",
        "output": "Evolved state under Hamiltonian evolution.\n哈密顿量演化下的演化态。",
    },
    "hhl": {
        "problem": "Linear system solver / 线性方程组求解器",
        "desc": "Quantum algorithm for Ax = b, exponential speedup.\n量子算法求解 Ax = b，指数加速。",
        "app": "- Machine learning (机器学习)\n- Optimization (优化)\n- Differential equations (微分方程)",
        "output": "Quantum state proportional to x = A^{-1}b.\n与 x = A^{-1}b 成正比的量子态。",
    },
    "hsp": {
        "problem": "Hidden Subgroup Problem / 隐藏子群问题",
        "desc": "General framework for Simon, Shor, and other HSP algorithms.\nSimon、Shor 和其他 HSP 算法的通用框架。",
        "app": "- Factoring (因式分解)\n- Discrete log (离散对数)\n- Graph isomorphism (图同构)",
        "output": "Subgroup generators.\n子群生成元。",
    },
    "jordan_wigner": {
        "problem": "Jordan-Wigner transform / Jordan-Wigner 变换",
        "desc": "Map fermionic Hamiltonian to qubit Hamiltonian.\n将费米子哈密顿量映射到量子比特哈密顿量。",
        "app": "- Quantum chemistry (量子化学)\n- Fermionic systems (费米子系统)\n- Hubbard model (Hubbard 模型)",
        "output": "Qubit Hamiltonian for simulation.\n用于模拟的量子比特哈密顿量。",
    },
    "lattice_svp": {
        "problem": "Shortest Vector Problem / 最短向量问题",
        "desc": "Quantum approach to lattice-based cryptography.\n格密码的量子方法。",
        "app": "- Post-quantum cryptography (后量子密码学)\n- Lattice-based crypto (格密码)\n- Security analysis (安全分析)",
        "output": "Approximate shortest vector.\n近似最短向量。",
    },
    "molecule_vqe": {
        "problem": "Molecular ground state / 分子基态",
        "desc": "Compute ground state energy of molecules.\n计算分子的基态能量。",
        "app": "- Drug discovery (药物发现)\n- Material design (材料设计)\n- Chemical reactions (化学反应)",
        "output": "Ground state energy of molecule.\n分子的基态能量。",
    },
    "phase_flip_code": {
        "problem": "Correct phase-flip errors / 纠正相位翻转错误",
        "desc": "3-qubit code corrects single phase-flip errors.\n3 比特码纠正单个相位翻转错误。",
        "app": "- Quantum error correction (量子纠错)\n- Phase protection (相位保护)\n- NISQ algorithms (NISQ 算法)",
        "output": "Corrected logical state despite phase errors.\n尽管有相位错误，纠正后的逻辑态。",
    },
    "qaoa_knapsack": {
        "problem": "Knapsack problem / 背包问题",
        "desc": "QAOA for knapsack: maximize value within weight limit.\nQAOA 求解背包问题：在重量限制内最大化价值。",
        "app": "- Combinatorial optimization (组合优化)\n- Resource allocation (资源分配)\n- Logistics (物流)",
        "output": "Optimal subset of items.\n最优物品子集。",
    },
    "qaoa_maxcut": {
        "problem": "MaxCut problem / 最大割问题",
        "desc": "QAOA for MaxCut: partition graph to maximize edges between sets.\nQAOA 求解最大割：划分图以最大化集合间边数。",
        "app": "- Graph partitioning (图划分)\n- Network design (网络设计)\n- Clustering (聚类)",
        "output": "Approximate max cut value.\n近似最大割值。",
    },
    "qaoa_mis": {
        "problem": "Maximum Independent Set / 最大独立集",
        "desc": "QAOA for MIS: find largest set of non-adjacent vertices.\nQAOA 求解最大独立集：找到最大的非相邻顶点集。",
        "app": "- Graph theory (图论)\n- Scheduling (调度)\n- Resource allocation (资源分配)",
        "output": "Approximate MIS size.\n近似最大独立集大小。",
    },
    "qaoa_tsp": {
        "problem": "Traveling Salesman Problem / 旅行商问题",
        "desc": "QAOA for TSP: find shortest route visiting all cities.\nQAOA 求解 TSP：找到访问所有城市的最短路线。",
        "app": "- Logistics (物流)\n- Route planning (路线规划)\n- Circuit design (电路设计)",
        "output": "Approximate tour cost.\n近似旅行成本。",
    },
    "qbm": {
        "problem": "Quantum Boltzmann Machine / 量子玻尔兹曼机",
        "desc": "Quantum version of Boltzmann machine for generative modeling.\n量子版玻尔兹曼机用于生成建模。",
        "app": "- Generative models (生成模型)\n- Sampling (采样)\n- Machine learning (机器学习)",
        "output": "Learned probability distribution.\n学习到的概率分布。",
    },
    "qcnn": {
        "problem": "Quantum Convolutional Neural Network / 量子卷积神经网络",
        "desc": "Quantum CNN for classification tasks.\n量子 CNN 用于分类任务。",
        "app": "- Image classification (图像分类)\n- Pattern recognition (模式识别)\n- Quantum ML (量子机器学习)",
        "output": "Classification accuracy.\n分类准确率。",
    },
    "qgan": {
        "problem": "Quantum GAN / 量子 GAN",
        "desc": "Quantum generator + classical discriminator.\n量子生成器 + 经典判别器。",
        "app": "- Data generation (数据生成)\n- Image synthesis (图像合成)\n- Quantum ML (量子机器学习)",
        "output": "Generated data distribution.\n生成的数据分布。",
    },
    "qgnn": {
        "problem": "Quantum Graph Neural Network / 量子图神经网络",
        "desc": "Quantum GNN for graph-structured data.\n量子 GNN 用于图结构数据。",
        "app": "- Graph classification (图分类)\n- Molecular property prediction (分子性质预测)\n- Social networks (社交网络)",
        "output": "Graph/node embeddings.\n图/节点嵌入。",
    },
    "qng": {
        "problem": "Quantum Natural Gradient / 量子自然梯度",
        "desc": "Uses quantum Fisher information for better optimization.\n使用量子 Fisher 信息进行更好的优化。",
        "app": "- Variational algorithms (变分算法)\n- VQE optimization (VQE 优化)\n- Quantum ML (量子机器学习)",
        "output": "Optimized parameters with faster convergence.\n更快收敛的优化参数。",
    },
    "qnn": {
        "problem": "Quantum Neural Network / 量子神经网络",
        "desc": "Variational quantum circuit as neural network.\n变分量子电路作为神经网络。",
        "app": "- Classification (分类)\n- Regression (回归)\n- Function approximation (函数逼近)",
        "output": "Trained model predictions.\n训练模型预测。",
    },
    "qpca": {
        "problem": "Quantum PCA / 量子 PCA",
        "desc": "Exponentially faster PCA for density matrices.\n密度矩阵的指数加速 PCA。",
        "app": "- Dimensionality reduction (降维)\n- Data analysis (数据分析)\n- Feature extraction (特征提取)",
        "output": "Principal eigenvalues.\n主特征值。",
    },
    "qrl": {
        "problem": "Quantum Reinforcement Learning / 量子强化学习",
        "desc": "Quantum agent learning in classical environment.\n经典环境中的量子智能体学习。",
        "app": "- Game playing (游戏)\n- Robotics (机器人)\n- Optimization (优化)",
        "output": "Learned policy.\n学习到的策略。",
    },
    "qsp": {
        "problem": "Quantum Signal Processing / 量子信号处理",
        "desc": "Core subroutine for quantum singular value transformation.\n量子奇异值变换的核心子程序。",
        "app": "- Quantum algorithms (量子算法)\n- Hamiltonian simulation (哈密顿量模拟)\n- Eigenvalue problems (特征值问题)",
        "output": "Transformed signal.\n变换后的信号。",
    },
    "qsvm": {
        "problem": "Quantum Support Vector Machine / 量子支持向量机",
        "desc": "SVM with quantum kernel for classification.\n使用量子核的 SVM 进行分类。",
        "app": "- Classification (分类)\n- Pattern recognition (模式识别)\n- Quantum ML (量子机器学习)",
        "output": "Classification accuracy.\n分类准确率。",
    },
    "qtda": {
        "problem": "Quantum Topological Data Analysis / 量子拓扑数据分析",
        "desc": "Quantum algorithm for persistent homology.\n持续同调的量子算法。",
        "app": "- Data analysis (数据分析)\n- Shape recognition (形状识别)\n- Topology (拓扑学)",
        "output": "Topological features.\n拓扑特征。",
    },
    "qtransformer": {
        "problem": "Quantum Transformer / 量子 Transformer",
        "desc": "Quantum attention mechanism for sequence modeling.\n用于序列建模的量子注意力机制。",
        "app": "- NLP (自然语言处理)\n- Sequence modeling (序列建模)\n- Quantum ML (量子机器学习)",
        "output": "Attention weights.\n注意力权重。",
    },
    "quantum_annealing": {
        "problem": "Quantum Annealing / 量子退火",
        "desc": "Hybrid classical-quantum annealing for optimization.\n用于优化的混合经典-量子退火。",
        "app": "- Optimization (优化)\n- Combinatorial problems (组合问题)\n- Sampling (采样)",
        "output": "Approximate ground state.\n近似基态。",
    },
    "quantum_bayesian": {
        "problem": "Quantum Bayesian Inference / 量子贝叶斯推断",
        "desc": "Quantum version of Bayesian updating.\n量子版贝叶斯更新。",
        "app": "- Inference (推断)\n- Decision making (决策)\n- Statistics (统计)",
        "output": "Posterior probabilities.\n后验概率。",
    },
    "quantum_clustering": {
        "problem": "Quantum Clustering / 量子聚类",
        "desc": "Quantum algorithm for unsupervised clustering.\n无监督聚类的量子算法。",
        "app": "- Data analysis (数据分析)\n- Customer segmentation (客户细分)\n- Anomaly detection (异常检测)",
        "output": "Cluster assignments.\n聚类分配。",
    },
    "quantum_eigenvalue": {
        "problem": "Eigenvalue Estimation / 特征值估计",
        "desc": "Estimate eigenvalues of unitary operators.\n估计酉算子的特征值。",
        "app": "- Quantum chemistry (量子化学)\n- Physics (物理学)\n- Linear algebra (线性代数)",
        "output": "Eigenvalue estimates.\n特征值估计。",
    },
    "quantum_fitting": {
        "problem": "Quantum Curve Fitting / 量子曲线拟合",
        "desc": "Quantum version of regression/curve fitting.\n量子版回归/曲线拟合。",
        "app": "- Data fitting (数据拟合)\n- Prediction (预测)\n- Machine learning (机器学习)",
        "output": "Fitted parameters.\n拟合参数。",
    },
    "quantum_kernel": {
        "problem": "Quantum Kernel Estimation / 量子核估计",
        "desc": "Compute quantum kernel matrix for ML.\n计算用于机器学习的量子核矩阵。",
        "app": "- Kernel methods (核方法)\n- SVM (支持向量机)\n- Quantum ML (量子机器学习)",
        "output": "Kernel matrix entries.\n核矩阵元素。",
    },
    "quantum_matrix_inversion": {
        "problem": "Matrix Inversion / 矩阵求逆",
        "desc": "HHL-based matrix inversion for linear systems.\n基于 HHL 的线性系统矩阵求逆。",
        "app": "- Linear systems (线性系统)\n- Machine learning (机器学习)\n- Optimization (优化)",
        "output": "Solution vector.\n解向量。",
    },
    "quantum_monte_carlo": {
        "problem": "Quantum Monte Carlo / 量子蒙特卡洛",
        "desc": "Quantum speedup for Monte Carlo methods.\n蒙特卡洛方法的量子加速。",
        "app": "- Integration (积分)\n- Risk analysis (风险分析)\n- Finance (金融)",
        "output": "Estimated integral value.\n估计积分值。",
    },
    "quantum_ode": {
        "problem": "ODE Solver / ODE 求解器",
        "desc": "Quantum algorithm for ordinary differential equations.\n常微分方程的量子算法。",
        "app": "- Physics simulation (物理模拟)\n- Engineering (工程)\n- Dynamics (动力学)",
        "output": "Solution trajectory.\n解轨迹。",
    },
    "quantum_pde": {
        "problem": "PDE Solver / PDE 求解器",
        "desc": "Quantum algorithm for partial differential equations.\n偏微分方程的量子算法。",
        "app": "- Fluid dynamics (流体力学)\n- Heat transfer (热传导)\n- Electromagnetics (电磁学)",
        "output": "Solution field.\n解场。",
    },
    "quantum_walk": {
        "problem": "Quantum Walk / 量子行走",
        "desc": "Quantum analogue of random walk, spreads quadratically faster.\n随机行走的量子类比，二次方更快扩展。",
        "app": "- Search algorithms (搜索算法)\n- Graph algorithms (图算法)\n- Transport phenomena (输运现象)",
        "output": "Position distribution after n steps.\nn 步后的位置分布。",
    },
    "rejection_sampling": {
        "problem": "Rejection Sampling / 拒绝采样",
        "desc": "Quantum-enhanced rejection sampling.\n量子增强的拒绝采样。",
        "app": "- Sampling (采样)\n- Distribution generation (分布生成)\n- Monte Carlo (蒙特卡洛)",
        "output": "Samples from target distribution.\n目标分布的样本。",
    },
    "shor_code": {
        "problem": "Shor's 9-qubit Code / Shor 9 比特码",
        "desc": "First quantum error correction code, corrects arbitrary errors.\n第一个量子纠错码，纠正任意错误。",
        "app": "- Quantum error correction (量子纠错)\n- Fault tolerance (容错)\n- Quantum memory (量子存储)",
        "output": "Corrected logical qubit.\n纠正后的逻辑比特。",
    },
    "simon": {
        "problem": "Simon's Algorithm / Simon 算法",
        "desc": "Find hidden period of 2-to-1 function. Precursor to Shor.\n找到 2-to-1 函数的隐藏周期。Shor 的前身。",
        "app": "- Cryptography (密码学)\n- Period finding (周期查找)\n- Quantum advantage (量子优势)",
        "output": "Hidden period string.\n隐藏周期串。",
    },
    "stabilizer": {
        "problem": "Stabilizer Formalism / 稳定子形式",
        "desc": "Clifford group simulation via stabilizer tableau.\n通过稳定子表模拟 Clifford 群。",
        "app": "- Error correction (纠错)\n- Clifford simulation (Clifford 模拟)\n- Quantum circuits (量子电路)",
        "output": "Stabilizer state measurements.\n稳定子态测量。",
    },
    "steane_code": {
        "problem": "Steane Code / Steane 码",
        "desc": "[[7,1,3]] CSS code, corrects arbitrary single-qubit errors.\n[[7,1,3]] CSS 码，纠正任意单比特错误。",
        "app": "- Quantum error correction (量子纠错)\n- Fault tolerance (容错)\n- Logical gates (逻辑门)",
        "output": "Corrected logical qubit.\n纠正后的逻辑比特。",
    },
    "superdense_coding": {
        "problem": "Superdense Coding / 超密编码",
        "desc": "Send 2 classical bits using 1 qubit.\n用 1 个量子比特发送 2 个经典比特。",
        "app": "- Quantum communication (量子通信)\n- Bandwidth doubling (带宽翻倍)\n- Teleportation (隐形传态)",
        "output": "Decoded 2-bit message.\n解码的 2 比特消息。",
    },
    "surface_code": {
        "problem": "Surface Code / 表面码",
        "desc": "Leading candidate for fault-tolerant quantum computing.\n容错量子计算的主要候选方案。",
        "app": "- Fault tolerance (容错)\n- Quantum memory (量子存储)\n- Logical qubits (逻辑比特)",
        "output": "Logical qubit with error protection.\n具有错误保护的逻辑比特。",
    },
    "swap_test": {
        "problem": "SWAP Test / SWAP 测试",
        "desc": "Estimate overlap between two quantum states.\n估计两个量子态之间的重叠。",
        "app": "- State comparison (态比较)\n- Kernel estimation (核估计)\n- Fidelity measurement (保真度测量)",
        "output": "P(|0⟩) = (1 + |⟨a|b⟩|²) / 2.",
    },
    "syndrome": {
        "problem": "Syndrome Measurement / Syndrome 测量",
        "desc": "Extract error syndromes without disturbing encoded state.\n提取错误 syndrome 而不扰动态。",
        "app": "- Error detection (错误检测)\n- QEC decoding (QEC 解码)\n- Fault tolerance (容错)",
        "output": "Syndrome bits indicating error location.\n指示错误位置的 syndrome 比特。",
    },
    "vqr": {
        "problem": "Variational Quantum Regressor / 变分量子回归器",
        "desc": "Quantum model for regression tasks.\n用于回归任务的量子模型。",
        "app": "- Regression (回归)\n- Prediction (预测)\n- Function fitting (函数拟合)",
        "output": "Predicted values.\n预测值。",
    },
    # Additional examples
    "basic_gates": "Basic quantum gates demonstration / 基本量子门演示",
    "compare": "Compare backends / 比较后端",
    "controlled": "Controlled gates / 受控门",
    "coupling_map": "Coupling map / 耦合图",
    "creg_multi": "Multiple classical registers / 多经典寄存器",
    "cwhile": "Classical while loop / 经典 while 循环",
    "decompose": "Gate decomposition / 门分解",
    "diffusion": "Diffusion operator / 扩散算子",
    "error_correction": "Error correction / 纠错",
    "error_mitigation": "Error mitigation / 错误缓解",
    "foundational": "Foundational circuits / 基础电路",
    "from_qiskit_nature": "Convert from Qiskit Nature / 从 Qiskit Nature 转换",
    "ghz": "GHZ state / GHZ 态",
    "gpu_demo": "GPU acceleration / GPU 加速",
    "groverize": "Groverize cwhile / Grover 化 cwhile",
    # hardware_compile is defined below with full bilingual docs
    "mark_state": "Mark state / 标记态",
    "noise": "Noise simulation / 噪声模拟",
    "noise_model": "Noise model / 噪声模型",
    "oracle": "Oracle construction / 预言机构造",
    "qif": "Quantum if / 量子 if",
    "qint": "Quantum integer / 量子整数",
    "qpe": "Quantum Phase Estimation / 量子相位估计",
    "quantum_counting": "Quantum Counting / 量子计数",
    "schedule": "Scheduling / 调度",
    "shor": "Shor's algorithm / Shor 算法",
    "trotter": "Trotterization / Trotter 分解",
    "vqc": "Variational Quantum Classifier / 变分量子分类器",
    # Advanced workflow examples
    "chemistry_workflow": {
        "problem": "Quantum Chemistry Workflow / 量子化学工作流",
        "desc": "Complete workflow for molecular ground state energy calculation.\n分子基态能量计算的完整工作流。",
        "app": "- Drug discovery (药物发现)\n- Material design (材料设计)\n- Chemical reactions (化学反应)",
        "output": "Ground state energy with error mitigation.\n带有错误缓解的基态能量。",
    },
    "optimization_workflow": {
        "problem": "Quantum Optimization Workflow / 量子优化工作流",
        "desc": "Complete workflow for combinatorial optimization with QAOA.\nQAOA 组合优化的完整工作流。",
        "app": "- Logistics (物流)\n- Scheduling (调度)\n- Network design (网络设计)",
        "output": "Optimized solution with multi-backend comparison.\n多后端对比的优化解。",
    },
    "qec_workflow": {
        "problem": "Quantum Error Correction Workflow / 量子纠错工作流",
        "desc": "Complete QEC workflow with noise modeling and decoding.\n带有噪声建模和解码的完整 QEC 工作流。",
        "app": "- Fault-tolerant computing (容错计算)\n- Quantum memory (量子存储)\n- Logical qubits (逻辑比特)",
        "output": "Error correction performance comparison.\n纠错性能对比。",
    },
    "hardware_compile": {
        "problem": "Hardware-Aware Compilation / 硬件感知编译",
        "desc": "Circuit compilation with topology constraints and optimization.\n带有拓扑约束和优化的电路编译。",
        "app": "- NISQ algorithms (NISQ 算法)\n- Hardware targeting (硬件目标)\n- Circuit optimization (电路优化)",
        "output": "Compiled circuit with reduced depth.\n减少深度的编译电路。",
    },
    "ml_workflow": {
        "problem": "Quantum Machine Learning Workflow / 量子机器学习工作流",
        "desc": "Complete QML workflow with training and prediction.\n带有训练和预测的完整 QML 工作流。",
        "app": "- Classification (分类)\n- Regression (回归)\n- Pattern recognition (模式识别)",
        "output": "Trained model with predictions.\n带有预测的训练模型。",
    },
    # Paper reproductions
    "vqe_h2": {
        "problem": "VQE for H₂ Molecule / VQE 计算 H₂ 分子",
        "desc": "Reproduce Peruzzo et al. (2014) ground state energy calculation.\n复现 Peruzzo et al. (2014) 基态能量计算。",
        "app": "- Quantum chemistry (量子化学)\n- Molecular simulation (分子模拟)\n- Benchmark (基准测试)",
        "output": "Ground state energy ≈ -1.137 Hartree.\n基态能量 ≈ -1.137 Hartree。",
    },
    # qaoa_maxcut is already defined above
    "grover_search": {
        "problem": "Grover's Search Algorithm / Grover 搜索算法",
        "desc": "Reproduce Grover (1996) quantum search.\n复现 Grover (1996) 量子搜索。",
        "app": "- Database search (数据库搜索)\n- Cryptography (密码学)\n- Benchmark (基准测试)",
        "output": "Target found with ~99% probability.\n目标以 ~99% 概率找到。",
    },
    # Additional utility examples
    "qaoa": "QAOA algorithm / QAOA 算法",
    "qi": "Quantum Inspire backend / Quantum Inspire 后端",
    "qi_hardware": "Quantum Inspire hardware / Quantum Inspire 硬件",
    "scheduler_demo": "Scheduler demonstration / 调度器演示",
    "cif": "Classical if statement / 经典 if 语句",
    "teleportation": "Quantum teleportation / 量子隐形传态",
}


def update_example(path):
    """Update an example with bilingual documentation."""
    dirname = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path).replace('.py', '')

    # Find matching doc
    doc = DOCS.get(name) or DOCS.get(dirname)
    if doc is None:
        return False

    if isinstance(doc, str):
        # Simple description — convert to full doc format
        problem = doc
        desc = doc
        app = "- Quantum computing (量子计算)\n- Algorithm demonstration (算法演示)\n- Educational (教学)"
        output = "See code comments for output explanation.\n参见代码注释了解输出说明。"
    else:
        problem = doc.get("problem", "")
        desc = doc.get("desc", "")
        app = doc.get("app", "- Quantum computing (量子计算)\n- Algorithm demonstration (算法演示)")
        output = doc.get("output", "See code comments for output explanation.\n参见代码注释了解输出说明。")

    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Check if already has bilingual docs with application scenarios
    if "## Application" in code and "## 应用场景" in code:
        return False

    # Extract code after docstring
    lines = code.split('\n')
    code_start = 0
    in_docstring = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('"""', "'''")):
            if in_docstring:
                code_start = i + 1
                break
            in_docstring = True
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                code_start = i + 1
                break
        elif in_docstring and ('"""' in stripped or "'''" in stripped):
            code_start = i + 1
            break

    actual_code = '\n'.join(lines[code_start:]).lstrip('\n')

    # Build new docstring
    doc_parts = [problem, ""]
    if desc:
        doc_parts.append(desc)
        doc_parts.append("")
    if app:
        doc_parts.append("## Application / 应用场景")
        doc_parts.append(app)
        doc_parts.append("")
    if output:
        doc_parts.append("## Output / 输出")
        doc_parts.append(output)
        doc_parts.append("")

    new_doc = '\n'.join(doc_parts).rstrip()
    new_content = f'"""{new_doc}"""\n\n{actual_code}'

    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


def main():
    updated = 0
    skipped = 0

    # Find all example .py files (including subdirectories)
    for pattern in ["examples/*/[a-z]*.py", "examples/*/*/[a-z]*.py", "examples/*/*/*/[a-z]*.py"]:
        for path in sorted(glob.glob(pattern)):
            if os.path.basename(path).startswith('_'):
                continue
            if update_example(path):
                updated += 1
                print(f"  Updated: {path}")
            else:
                skipped += 1

    print(f"\nTotal: {updated} updated, {skipped} skipped")


if __name__ == "__main__":
    main()
