Title  
Hierarchical Predictive Coding–Driven Active Inference for Sample-Efficient Reinforcement Learning  

Introduction  
Background  
Reinforcement learning (RL) has achieved remarkable success in domains ranging from games (e.g., AlphaZero) to robotics (e.g., dexterous manipulation). However, state-of-the-art model-free and model-based algorithms often demand millions of environment interactions to reach competitive performance. In contrast, biological agents—from rodents to humans—demonstrate extraordinary data efficiency, learning complex sensorimotor and cognitive skills in hours or days. Neuroscience theories of predictive coding and active inference offer a principled account of how the brain builds hierarchical generative models to minimize surprise (“free energy”) and selects actions to reduce uncertainty about future observations. By embedding these neurobiological principles into RL, we aim to design artificial agents that (1) learn compact, compositional world models through prediction error minimization and (2) perform intrinsically driven exploration via expected free-energy minimization, thereby vastly improving sample efficiency.

Research Objectives  
1. Develop a hierarchical predictive coding network (H-PCN) that learns a multi-level generative model of the environment by minimizing variational free energy on sensory inputs.  
2. Integrate active inference by defining an expected free-energy (EFE) objective that balances reward maximization and epistemic (information-seeking) exploration.  
3. Formulate an RL algorithm—Predictive Coding Active Inference RL (PCAI-RL)—that unifies world-model learning, policy optimization, and intrinsic reward generation under the free-energy principle.  
4. Empirically demonstrate that PCAI-RL achieves superior sample efficiency on sparse-reward and hard-exploration benchmarks (e.g., MiniGrid, Montezuma’s Revenge) compared to standard model-free (PPO, DQN) and model-based (Dreamer, PlaNet) baselines.  

Significance  
This project bridges NeuroAI and RL by translating predictive-coding and active-inference insights into a practical RL framework. The expected gains in data efficiency will benefit real-world applications with expensive or risky interactions—robotics, autonomous driving, clinical decision support—and foster new synergies between neuroscience and machine learning.

Methodology  
1. Hierarchical Predictive Coding Network (H-PCN)  
   a. Architecture  
      • A stack of $L$ layers, each corresponding to a latent state $z_{t}^{(l)}$ at time step $t$ and abstraction level $l\in\{1,\dots,L\}$.  
      • Bottom layer receives raw observation $o_{t}$ and produces prediction $\hat{o}_{t+1}\,$. Upper layers capture increasingly abstract features (e.g., object identity, scene context).  
   b. Generative and Recognition Models  
      • Generative model:  
        $$p(z_{t+1}^{(l)},o_{t+1}\mid z_{t}^{(l)},a_{t}) = p(o_{t+1}\mid z_{t+1}^{(1)})\prod_{l=1}^{L}p\bigl(z_{t+1}^{(l)}\mid z_{t}^{(l)},a_{t},z_{t+1}^{(l+1)}\bigr)\,, $$  
        where $z^{(L+1)}$ is a global context vector (e.g., task goal).  
      • Recognition (inference) model:  
        $$q\bigl(z_{t+1}^{(l)}\mid o_{t+1},z_{t}^{(l)},a_{t},z_{t}^{(l+1)}\bigr)\,. $$  
   c. Free-Energy Minimization  
      • The variational free energy for each time step:  
        $$
        \mathcal{F}_{t} 
        = \mathbb{E}_{q(z_{t+1}\mid o_{t+1},a_{t})}\Bigl[\ln q(z_{t+1}\mid o_{t+1},a_{t}) - \ln p(o_{t+1},z_{t+1}\mid z_{t},a_{t})\Bigr]\,.
        $$  
      • Summed over time and layers: $\mathcal{F} = \sum_{t=0}^{T-1}\sum_{l=1}^{L}\mathcal{F}_{t}^{(l)}$.  
      • Parameter updates via stochastic gradient descent on $\mathcal{F}$.  

2. Active Inference and Expected Free Energy (EFE)  
   a. Definition of EFE  
      • For each candidate action sequence (policy) $\pi$, the expected free energy at time $t$ is:  
        $$
        G_{t}(\pi) 
        = \mathbb{E}_{\tilde{o},\tilde{z}\sim q(\cdot\mid\pi)}\Bigl[
          -\ln p(\tilde{o}\mid\pi)
          + \mathrm{D_{KL}}\bigl[q(\tilde{z}\mid\tilde{o},\pi)\,\|\,p(\tilde{z})\bigr]
        \Bigr]\,.  
        $$  
      • The first term encourages goal-directed behavior (extrinsic reward). The second term encourages exploration (epistemic drive).  
   b. Action Selection  
      • At each decision point, the agent samples a set of candidate action sequences $\{\pi^{(i)}\}$, computes $G_{t}(\pi^{(i)})$ via Monte Carlo rollouts through the learned H-PCN, and selects  
        $$
        \pi^{*} = \arg\min_{\pi^{(i)}}G_{t}(\pi^{(i)})\,.  
        $$  
      • In practice, approximate with $n$-step lookahead and model predictive control (MPC).  

3. RL Algorithm: PCAI-RL Pseudocode  
   Initialize H-PCN parameters $\theta$ and policy parameters $\phi$.  
   For each episode:  
     1. Observe $o_{t}$; infer latent $z_{t}$ via recognition model.  
     2. Generate candidate action sequences $\{\pi^{(i)}\}$ (e.g., CEM sampling).  
     3. For each $\pi^{(i)}$, roll out H-PCN to predict $(\hat{o}_{t+1:\,t+n},\hat{z}_{t+1:\,t+n})$; compute $G_{t}(\pi^{(i)})$.  
     4. Select action $a_{t}$ from $\pi^{*}$.  
     5. Execute $a_{t}$ in the environment, receive $o_{t+1},r_{t}$.  
     6. Store transition $(o_{t},a_{t},r_{t},o_{t+1})$ in replay buffer.  
     7. Sample minibatch; update $\theta$ via $\nabla_{\theta}\mathcal{F}$ and $\phi$ via policy gradient on extrinsic + intrinsic reward $-\!G_{t}$.  

4. Experimental Design  
   a. Benchmarks  
      • MiniGrid (KeyCorridor, MultiRoom) for sparse-reward navigation.  
      • Atari Montezuma’s Revenge for hard exploration.  
      • Continuous-control tasks (SparseCartPole, SparseMountainCar).  
   b. Baselines  
      • Model-free: PPO, DQN with count-based exploration.  
      • Model-based: Dreamer, PlaNet, Active Predictive Coding (Rao et al., 2022).  
   c. Metrics  
      • Sample efficiency: environment steps to reach a fixed return threshold.  
      • Final performance: average return over last 10 episodes.  
      • Epistemic gain: average KL divergence between prior and posterior $q(z\mid o)$.  
      • Computational cost: UTD ratio, wall-clock training time.  
      • Statistical significance: report means±std over 5 random seeds; perform pairwise t-tests.  
   d. Ablations  
      • Without hierarchical depth ($L=1$ vs $L>1$).  
      • Without epistemic term in EFE (pure reward maximization).  
      • Varying lookahead horizon $n$.  

5. Implementation Details  
   • Network sizes: 3-layer convolutions for vision tasks; 2 hidden layers for continuous control. Latent dims $\{32,16,8\}$.  
   • Optimizer: Adam with learning rate $3\!\times\!10^{-4}$.  
   • CEM sampling: population 500, elites 50, iterations 5.  
   • Hardware: NVIDIA A100 GPUs; parallel environments with 32 workers.  

Expected Outcomes & Impact  
Expected Outcomes  
• PCAI-RL will achieve 2×–5× improvement in sample efficiency over Dreamer and PPO on sparse-reward benchmarks.  
• Agents will demonstrate more structured exploration trajectories, as evidenced by higher epistemic gain and faster coverage of novel states.  
• Ablations will confirm the critical role of hierarchical predictive coding and active-inference-driven intrinsic rewards in driving data efficiency.  

Broader Impact  
NeuroAI: This project concretely validates how predictive coding and active inference—long studied in neuroscience—can inform next-generation RL. Our hierarchical architecture offers a blueprint for integrating bottom-up sensory prediction with top-down goal priors.  
Real-World RL: Dramatic reductions in required interactions enable safe deployment in robotics, healthcare, and autonomous systems, where data collection is expensive or hazardous.  
Energy Efficiency: By minimizing environment queries and leveraging compact world models, PCAI-RL reduces compute and energy footprints, aligning with sustainable AI goals.  
Future Directions  
• Neuromorphic implementation on spiking hardware (e.g., Intel Loihi) to further lower power consumption.  
• Extension to multi-agent active inference for collaborative exploration and communication.  
• Integration with neuro-symbolic modules for structured reasoning and interpretability.  
In summary, this research proposal outlines a comprehensive plan to harness predictive coding and active inference within a hierarchical RL framework, offering a path toward biologically inspired, sample-efficient learning agents with broad scientific and societal benefits.