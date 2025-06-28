1. Title  
GreenSched: A Deep Reinforcement Learning–Driven Energy-Carbon-Aware Job Scheduler for Sustainable Cloud Datacenters  

2. Introduction  
Background  
Cloud datacenters account for a substantial fraction of global electricity consumption and associated CO₂ emissions. Traditional datacenter job schedulers prioritize performance metrics such as throughput, latency, or resource utilization, without explicitly accounting for real‐time energy prices or grid carbon intensity. Meanwhile, the rapid growth of AI workloads and the rising cost of electricity have heightened the need for scheduling solutions that jointly optimize for operational expenditure (OPEX) and environmental impact. Recent advances in carbon‐aware scheduling (e.g., CarbonClipper [Lechowicz et al., 2024], PCAPS [Lechowicz et al., 2025], CarbonScaler [Hanafy et al., 2023]) demonstrate substantial carbon reductions, yet these methods rely on convex‐optimization heuristics or elastic scaling without leveraging the full expressive power of deep reinforcement learning (DRL) in dynamic environments.

Research Objectives  
GreenSched aims to fill this gap by developing a high‐fidelity, DRL‐based job scheduler that:  
• Ingests real‐time workload features, node utilization metrics, dynamic electricity pricing, and grid carbon forecasts.  
• Learns to assign or defer jobs, adjust power‐cap settings, and trigger live VM/container migrations.  
• Balances multi‐objective rewards combining energy cost, carbon emissions, and service‐level agreement (SLA) violations.  
• Operates seamlessly on Kubernetes and can scale to thousands of GPU/TPU devices.  

Significance  
By integrating DRL with accurate simulator environments and real trace data, GreenSched promises:  
• 15–30% reductions in energy cost, 20–40% lower CO₂ emissions, and maintained or improved job completion times compared to baseline schedulers.  
• A reproducible, open‐source framework that unifies benchmarking for carbon‐aware scheduling research.  
• Insights into policy interpretability and trade‐offs between cost, carbon, and performance for practical deployments.

3. Methodology  
3.1 Problem Formulation  
We model the datacenter job scheduling problem as a Markov Decision Process (MDP) defined by $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:  
- State space $\mathcal{S}$: For each time step $t$, the state $s_t$ comprises  
  • Job queue snapshot $\{(d_i, r_i)\}$ where $d_i$ is job deadline/priority and $r_i$ is resource demand vector (CPU, GPU, memory).  
  • Node utilization vector $u_t\in\mathbb{R}^N$ for $N$ servers.  
  • Current electricity price $p_t$ (USD/kWh).  
  • Forecasted carbon intensity $c_t$ (gCO₂eq/kWh) over a short horizon.  
  • Renewable generation forecast $g_t$ (kW).  
- Action space $\mathcal{A}$: At each step, the agent chooses for each job $i$ either “schedule on server $j$,” “delay,” “set CPU/GPU power cap $z_{i,j}$,” or “migrate to server $k$.” We encode this as a discrete vector of dimension up to $|\mathcal{A}|=M\times N + 1 + P + N$, where $M$ is queue length and $P$ is number of power‐cap levels.  
- Transition function $P(s_{t+1}\mid s_t,a_t)$: Determined by job arrivals (modeled from real traces), job execution progress under the chosen actions, electricity price and carbon intensity time series.  
- Reward function $R(s_t,a_t)$:  
  
  $$  
  r_t = -\alpha\cdot E_t - \beta\cdot C_t - \gamma\cdot \mathrm{SLA}_t,  
  $$  
  
  where  
  • $E_t$ = energy consumed (kWh) during interval $[t,t+1]$.  
  • $C_t = E_t \times c_t$ = carbon emissions (gCO₂eq).  
  • $\mathrm{SLA}_t$ = penalty term for jobs missing deadlines (number of violations or tardiness).  
  • $(\alpha,\beta,\gamma)$ are tunable scalars to trade off cost, carbon, and performance.  
- Discount factor $\gamma \in [0,1]$ balances immediate vs. future objectives.  

3.2 Deep Reinforcement Learning Architecture  
We adopt an actor‐critic paradigm using Proximal Policy Optimization (PPO) with the following components:  
1. Policy network $\pi_\theta(a_t\mid s_t)$:  
   - Input: state embedding from CNN/RNN layers processing the grid of node utilizations and forecasts, concatenated with job queue features.  
   - Hidden layers: two fully connected layers of size 512 with ReLU activations.  
   - Output head: softmax over discrete actions.  
2. Value network $V_\phi(s_t)$ shares the lower layers with $\pi_\theta$ and outputs a scalar estimate of expected return.  
3. Training loss:  
  
  $$  
  L(\theta,\phi) = \mathbb{E}_t\Bigg[\underbrace{-\min\Big(r_t(\theta)\hat{A}_t, \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\Big)}_{\text{Policy surrogate}} + c_1\big(V_\phi(s_t)-R_t\big)^2 - c_2H\big(\pi_\theta(\cdot\mid s_t)\big)\Bigg],  
  $$  
  
  where $r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$, $\hat{A}_t$ is the advantage estimate, $\epsilon$ the clipping parameter, $c_1,c_2$ coefficient hyperparameters, and $H(\cdot)$ the policy entropy bonus.  

3.3 Simulator Design and Data Collection  
We build a high‐fidelity datacenter simulator that faithfully reproduces:  
• Workload dynamics using the Google cluster trace and Microsoft Azure job logs.  
• Electricity price and carbon intensity time series from ERCOT and CAISO markets.  
• Server power models calibrated from SPECpower and NVIDIA GPU power curves.  

Simulation loop pseudocode:  
```
Initialize simulator(S_0) with real trace seeds  
for each episode e=1..E:
    s ← S_0
    for t = 0..T:
        a ← policy(s)
        (s', cost, carbon, sla) ← Simulator.step(s,a)
        r ← -α*cost - β*carbon - γ*sla
        store (s,a,r,s') in buffer
        s ← s'
    update policy and value networks via PPO on buffer
```

3.4 Fine-Tuning on Kubernetes Testbed  
After simulator‐based convergence, we deploy the policy in a controlled Kubernetes cluster with:  
• 20 heterogeneous nodes (mix of CPU, GPU).  
• A job generator reproducing trace patterns.  
• A monitoring stack (Prometheus + Grafana) to collect energy usage via IPMI and carbon data from APIs.  

We use off‐policy correction (importance sampling) to fine‐tune $\pi_\theta$ for real‐world discrepancies and safe exploration strategies (e.g., action masking to prevent catastrophic SLA breaches).  

3.5 Experimental Design and Baselines  
We compare GreenSched against:  
• Round‐Robin and Greedy CPU/GPU schedulers (throughput‐optimized).  
• CarbonClipper (spatiotemporal carbon‐aware convex solver).  
• PCAPS (carbon‐ and precedence‐aware heuristic).  
• CarbonScaler (elastic scaling policy).  

Metrics:  
• Energy cost savings (%) relative to Greedy.  
• CO₂ emission reduction (%) relative to Greedy.  
• Job completion time (mean, tail = 95th percentile).  
• SLA violation rate (% tardy jobs).  
• Scheduler decision latency (ms).  

We conduct experiments across:  
• Three trace scenarios (ML training workloads; batch data‐processing; mixed AI+batch).  
• Two geographical regions (high‐carbon grid vs. low‐carbon mix).  
• Varying job arrival rates (utilization levels 30%, 60%, 90%).  

Each configuration is repeated over five random seeds to compute 95% confidence intervals.  

4. Expected Outcomes & Impact  
4.1 Energy and Carbon Reductions  
We anticipate GreenSched to achieve:  
• 15–30% reduction in energy expenditure over best‐in‐class carbon‐aware schedulers (PCAPS, CarbonScaler).  
• 20–40% lower CO₂ emissions in high‐variability grids due to proactive scheduling around carbon intensity peaks.  

4.2 Performance and SLA Adherence  
By learning end‐to‐end policies that jointly optimize costs and performance, GreenSched should maintain or improve job completion times. Preliminary simulation indicates <1% increase in average completion time and <0.5% SLA violation rate at 90% utilization.

4.3 Scalability and Overhead  
The centralized PPO policy runs in <50 ms per decision on a single CPU core. We will demonstrate scalability to hundreds of nodes by batching decisions and using efficient inference (TensorRT optimization).

4.4 Open-Source Framework and Reproducibility  
We will release:  
• The datacenter simulator with trace preprocessing scripts.  
• Trained policy checkpoints and training code.  
• Kubernetes testbed deployment manifests and monitoring dashboards.  

This end-to-end platform will serve as a unifying benchmark for the ML for Systems community, facilitating reproducible comparisons and future extensions in sustainable scheduling.

4.5 Broader Impacts  
GreenSched addresses the urgent need for carbon‐aware computing at scale. By operationalizing DRL in production‐like environments, we bridge the gap between ML research and systems deployment. The proposed framework will inform:  
• Cloud providers seeking to reduce carbon footprints and OPEX.  
• Grid operators and policymakers aiming to shape demand response programs.  
• Researchers exploring multi‐objective RL, safe exploration, and interpretable scheduling policies.

In summary, GreenSched represents a significant step toward sustainable cloud computing, combining rigorous DRL methodology with practical validation to deliver tangible environmental and economic benefits.