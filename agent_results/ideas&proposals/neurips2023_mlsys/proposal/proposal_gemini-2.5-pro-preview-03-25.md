**1. Title:** GreenSched: A Deep Reinforcement Learning Framework for Energy- and Carbon-Aware Job Scheduling in Cloud Datacenters

**2. Introduction**

*   **Background:** Cloud computing platforms have become indispensable infrastructure, powering a vast array of applications from scientific computing and big data analytics to everyday web services and the rapidly growing domain of Large Language Model (LLM) training and inference. However, the proliferation of datacenters comes at a significant environmental and economic cost. Globally, datacenters account for an estimated 1-2% of total electricity consumption, a figure projected to rise substantially with the increasing demand for computation, particularly for energy-intensive AI workloads. This massive energy consumption translates directly into substantial operational expenditures and, depending on the energy source mix, significant carbon emissions, contributing to global climate change.

    Traditional job schedulers in distributed systems (e.g., Kubernetes default scheduler, YARN, Slurm) primarily focus on optimizing performance metrics like throughput, latency, fairness, or resource utilization. They often rely on heuristics or rule-based systems that are ill-equipped to handle the complex, dynamic, and multi-objective nature of modern datacenter management, especially when incorporating sustainability goals. These schedulers typically lack awareness of real-time factors crucial for sustainability, such as fluctuating electricity prices driven by market dynamics and the varying carbon intensity of the electricity grid, which changes based on the proportion of renewable energy sources available at any given time.

    The ML for Systems community has recognized the potential of machine learning, particularly Deep Reinforcement Learning (DRL), to address complex control problems in computer systems by learning sophisticated policies directly from experience or simulation, often surpassing human-designed heuristics. Applying ML to job scheduling offers the promise of adaptively optimizing resource allocation based on real-time conditions. Recent work, as highlighted in our literature review (e.g., CarbonClipper [1], PCAPS [2], CarbonScaler [3]), has begun exploring carbon-aware workload management. However, challenges remain, including balancing performance with sustainability [1, 2], adapting to dynamic energy markets and carbon intensities [3], handling complex workload dependencies [2], scaling RL solutions [4], and ensuring seamless integration with existing infrastructure. Furthermore, few approaches holistically integrate dynamic energy pricing, carbon intensity forecasts, diverse workload characteristics, and fine-grained power management actions within a unified DRL framework designed for continuous online adaptation in large-scale heterogeneous clusters.

    The rise of LLMs introduces new system challenges and opportunities, aligning with the workshop's focus. Training and serving these massive models require enormous computational resources, primarily GPUs, exacerbating the energy and carbon concerns. An intelligent scheduler must efficiently manage these demanding workloads alongside traditional tasks, potentially leveraging their unique characteristics (e.g., long training durations, potential for checkpointing and migration) for better sustainability outcomes. This research directly addresses the workshop's call for applying ML to systems issues emerging from large-scale training/serving and for compute sustainability, aiming to move beyond simple heuristic replacement towards a learned, adaptive control policy.

*   **Research Objectives:** This research proposes GreenSched, a novel DRL-based job scheduling framework designed to co-optimize energy cost, carbon emissions, and application performance in heterogeneous cloud datacenters. Our primary objectives are:
    1.  **Develop a DRL Agent for Multi-Objective Scheduling:** Design and implement a DRL agent capable of learning an effective scheduling policy that minimizes a weighted combination of energy consumption cost and carbon footprint, while respecting Service Level Agreements (SLAs) for job performance (e.g., completion time deadlines).
    2.  **Comprehensive State and Action Space Design:** Define a rich state representation that captures dynamic workload characteristics (resource demands, queue times), cluster status (node utilization, power consumption, heterogeneity), real-time electricity pricing, and grid carbon intensity forecasts. Design a versatile action space including job placement, delaying, server power capping (CPU/GPU), and potentially VM/container migration.
    3.  **Adaptive Reward Formulation:** Formulate a reward signal that accurately reflects the multi-objective goals and allows for configurable trade-offs between energy cost, carbon emissions, and performance penalties, enabling adaptation to different operator priorities.
    4.  **High-Fidelity Simulation and Real-World Validation:** Develop a robust simulation environment incorporating realistic workload traces (including representative AI/LLM jobs), energy pricing models, carbon intensity data, and heterogeneous hardware models for pre-training the DRL agent. Validate and fine-tune the learned policy on a physical Kubernetes testbed to bridge the sim-to-real gap and assess practical deployability.
    5.  **Quantify Sustainability Gains and Performance Impact:** Rigorously evaluate GreenSched against baseline scheduling strategies (e.g., default Kubernetes scheduler, energy-oblivious policies, simple carbon-aware heuristics) across diverse workloads and environmental conditions, quantifying the achievable reductions in energy cost and carbon emissions, and analyzing the impact on job performance metrics (JCT, slowdown, SLA violations).
    6.  **Promote Reproducibility and Open Science:** Release the GreenSched framework, including the simulator, DRL agent implementation, and evaluation scripts, as an open-source project to foster further research and adoption in sustainable computing, aligning with the workshop's emphasis on reproducible research.

*   **Significance:** This research holds significant potential across multiple dimensions:
    *   **Environmental:** By optimizing for lower energy consumption and prioritizing usage during low-carbon intensity periods, GreenSched can substantially reduce the carbon footprint of datacenter operations, contributing to climate change mitigation efforts.
    *   **Economic:** Minimizing energy cost through price-aware scheduling and consumption reduction directly translates to lower operational expenditures for cloud providers and potentially lower service costs for users.
    *   **Technical:** This work advances the state-of-the-art in ML for Systems by demonstrating the application of DRL to complex, multi-objective resource management problems in dynamic environments. It addresses key challenges identified in the literature regarding balancing competing objectives and adapting to real-time environmental signals. The development of a comprehensive state/action space and reward function for this domain represents a notable contribution.
    *   **Practical:** The open-source GreenSched framework will provide practitioners with a tool to improve the sustainability of their clusters and researchers with a platform for building upon this work. Its validation on a Kubernetes testbed enhances its practical relevance and path towards real-world deployment. This directly aligns with the workshop's goal of bridging ML and Systems communities and fostering usable, reproducible artifacts.

**3. Methodology**

*   **Research Design:** We model the energy-carbon-aware job scheduling problem as a Markov Decision Process (MDP) and employ Deep Reinforcement Learning (DRL) to find an optimal scheduling policy. The DRL agent acts as the scheduler, interacting with the datacenter environment (simulated or real) at discrete time steps or upon specific events (e.g., job arrival, job completion).

*   **MDP Formulation:** The MDP is defined by the tuple $(S, A, P, R, \gamma)$:
    *   **State Space ($S$):** The state $s_t \in S$ at time $t$ provides a snapshot of the system and its context. It will be represented as a high-dimensional vector or potentially a graph structure, capturing:
        *   *Pending Jobs Queue:* For each job $j$ in the queue: resource requests (CPU cores, GPU units, RAM GB, disk I/O), estimated runtime (if available), arrival timestamp, current time in queue, SLA deadline (if applicable), priority level, job type (e.g., batch, interactive, ML training).
        *   *Cluster Node Status:* For each node $n$ (potentially heterogeneous server types): current resource utilization (CPU %, GPU %, RAM %, Network BW), allocated resources, available resources, current power consumption (W), current CPU/GPU frequency or power cap settings.
        *   *Environmental Context:* Current grid electricity price ($€/\text{kWh}$), forecast of electricity prices for the next $H$ hours. Current grid carbon intensity ($\text{gCO2eq}/\text{kWh}$), forecast of carbon intensity for the next $H$ hours. Information about local renewable energy generation (if applicable).
        *   *Time:* Current timestamp or time features (e.g., hour of day, day of week).
        The state representation will involve normalization and potentially feature engineering (e.g., calculating job urgency based on deadline and queue time). We will investigate flat vector representations versus graph neural network (GNN) approaches where nodes represent servers and jobs, capturing relationships more explicitly.

    *   **Action Space ($A$):** The action $a_t \in A$ selected by the agent based on state $s_t$. The action space is hybrid, combining discrete and potentially continuous choices:
        1.  *Placement Decision:* For a newly arrived job or a job at the head of the queue: assign job $j$ to a specific eligible node $n$.
        2.  *Delay Decision:* Defer scheduling job $j$ (keep in queue). This is crucial for time-shifting load to periods with lower energy cost or carbon intensity.
        3.  *Power Management:* Adjust the power cap or operating frequency for a specific node $n$ or a group of nodes (e.g., setting CPU P-states, GPU power limits). This might be discretized into several levels.
        4.  *(Optional/Advanced)* *Migration:* Initiate the migration of a running job (VM/container) $v$ from node $n_1$ to node $n_2$. This action has higher complexity and overhead, potentially explored in later stages.
        The agent might select a composite action (e.g., place job $j$ on node $n$ AND adjust power cap of node $n'$).

    *   **Transition Probability ($P$):** $P(s_{t+1} | s_t, a_t)$ defines the probability of transitioning to state $s_{t+1}$ after taking action $a_t$ in state $s_t$. This is implicitly defined by the datacenter environment simulator or the real system dynamics (job arrivals, completions, system monitoring updates).

    *   **Reward Function ($R$):** The reward $r_t = R(s_t, a_t, s_{t+1})$ signals the quality of action $a_t$ taken in state $s_t$. It's designed to encourage the agent towards the multi-objective goal. A possible formulation is:
        $$
        r_t = - w_{energy} \cdot \text{EnergyCost}_t - w_{carbon} \cdot \text{CarbonEmissions}_t - w_{perf} \cdot \text{PerformancePenalty}_t
        $$
        where:
        *   $\text{EnergyCost}_t$: Total energy cost incurred during the time interval $(t, t+1]$. Calculated as $\sum_{n \in \text{Nodes}} \text{Power}_n(t) \times \Delta t \times \text{Price}(t)$, where $\text{Power}_n(t)$ is the power consumption of node $n$ (modeled based on utilization and power caps) and $\text{Price}(t)$ is the electricity price.
        *   $\text{CarbonEmissions}_t$: Total carbon emissions during the interval. Calculated as $\sum_{n \in \text{Nodes}} \text{Power}_n(t) \times \Delta t \times \text{Intensity}(t)$, where $\text{Intensity}(t)$ is the grid carbon intensity.
        *   $\text{PerformancePenalty}_t$: Penalties incurred due to poor performance. This could include:
            *   SLA Violation Penalty: A large negative value if any job misses its deadline in the interval. $P_{sla} = \sum_{j \in \text{JobsFinishedLate}} C_{sla}(j)$, where $C_{sla}(j)$ is a penalty constant or function of lateness.
            *   (Optional) Job Slowdown Penalty: A term proportional to the average job slowdown to discourage generally poor performance even without strict deadlines. $P_{slowdown} = \sum_{j \in \text{JobsFinished}} \frac{\text{ActualJCT}(j)}{\text{IdealJCT}(j)}$.
        *   $w_{energy}, w_{carbon}, w_{perf}$: Non-negative weights balancing the objectives. These weights are crucial hyperparameters that allow operators to specify priorities (e.g., prioritize carbon reduction over cost savings). We will explore methods for setting these weights, potentially including sensitivity analysis or dynamic adjustment.

    *   **Discount Factor ($\gamma$):** $0 < \gamma \le 1$. Represents the preference for immediate versus future rewards. Typically close to 1 for long-horizon optimization problems.

*   **Deep Reinforcement Learning Algorithm:** Given the potentially large and continuous state space and the hybrid action space, we propose using an Actor-Critic algorithm suitable for complex control tasks, such as Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC).
    *   *PPO:* Known for its stability and good performance across various benchmarks. It optimizes a clipped surrogate objective function using multiple epochs of gradient ascent on sampled data.
    *   *SAC:* Maximizes a trade-off between expected return and policy entropy, encouraging exploration and leading to more robust policies, particularly effective with continuous actions (like power capping).
    *   *Network Architecture:* Both actor (policy) and critic (value function) will be implemented using deep neural networks (e.g., Multi-Layer Perceptrons - MLPs). If using a graph-based state representation, Graph Neural Networks (GNNs) will be employed to capture dependencies between jobs and nodes. Input features will be normalized.

*   **Data Collection and Simulation:**
    *   *Workload Traces:* We will utilize publicly available cluster workload traces (e.g., Google Cluster Data, Azure VM Traces) containing job arrival times, resource requests, and durations. We will supplement these with synthetic traces representing modern workloads, including LLM training (long-running, GPU-intensive) and inference (latency-sensitive, bursty) jobs, potentially generated based on characterizations from recent literature.
    *   *Energy Price Data:* Real-time and day-ahead electricity price data from various markets (e.g., CAISO, Nord Pool) will be used to model dynamic pricing scenarios.
    *   *Carbon Intensity Data:* Real-time and forecast data for grid carbon intensity from sources like Electricity Maps or WattTime will be integrated.
    *   *Simulation Environment:* We will develop a discrete-event simulator (e.g., using Python with SimPy or a custom C++ engine) that models:
        *   Heterogeneous cluster architecture (CPU/GPU nodes with different specs).
        *   Job arrivals based on traces.
        *   Job execution dynamics.
        *   Power models for components (CPU, GPU, RAM) based on utilization levels and power caps (e.g., using models derived from SPECpower benchmarks or empirical measurements).
        *   Dynamic energy pricing and carbon intensity feeds.
        *   The DRL agent's interaction loop (observe state, take action, receive reward).
    This simulator will enable rapid prototyping, safe exploration during training, and large-scale evaluation.

*   **Experimental Design and Validation:**
    1.  **Simulation-Based Evaluation:**
        *   *Baselines:*
            *   Default Kubernetes Scheduler (emulated based on priority/resource packing).
            *   First-Come-First-Served (FCFS).
            *   Basic Heuristics: Energy-aware (place on most efficient idle node), Carbon-aware (delay jobs until low-intensity periods if possible, simple thresholding).
            *   (If feasible) Re-implementations of core ideas from relevant literature (e.g., simplified CarbonScaler heuristic).
        *   *Scenarios:* Vary workload mixes (CPU-bound, GPU-bound, LLM jobs), cluster sizes/heterogeneity, energy price volatility (flat, time-of-use, real-time), and carbon intensity profiles (low-stable, high-variable). Test different settings of reward weights ($w_{energy}, w_{carbon}, w_{perf}$).
        *   *Metrics:* Sustainability (Total Energy kWh, Total Cost $, Total CO2eq kg), Performance (Avg. JCT, Avg. Slowdown, Throughput, SLA Violation Rate %), Utilization (Avg/Peak CPU/GPU/RAM Util %), Overhead (Scheduling latency).
    2.  **Kubernetes Testbed Fine-Tuning and Validation:**
        *   *Setup:* Deploy a small-scale Kubernetes cluster (e.g., 5-10 nodes with CPU/GPU heterogeneity). Integrate monitoring tools (Prometheus, node-exporter, GPU monitoring) to collect real-time state. Develop scheduler plugin or custom controller to enact GreenSched's decisions (job binding, power capping via OS interfaces like `cpufreq`, `nvidia-smi`).
        *   *Procedure:* Load the best policy pre-trained in simulation. Perform limited fine-tuning with real system interactions (if feasible and stable) or conduct zero-shot evaluation. Run representative workloads.
        *   *Metrics:* Collect the same metrics as in simulation, comparing GreenSched against the default Kubernetes scheduler on the physical testbed. Assess practical challenges (integration complexity, control latency, state accuracy).

*   **Reproducibility:** We commit to releasing the GreenSched code, including the simulator, DRL agent implementation (using standard libraries like TensorFlow/PyTorch, Stable Baselines3/RLlib), configuration files, workload generation scripts, and data processing scripts under a permissive open-source license (e.g., Apache 2.0 or MIT). The simulator and agent will be containerized using Docker for ease of setup and replication of results.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A Novel DRL Scheduling Framework:** A fully implemented and validated GreenSched framework demonstrating the feasibility of using DRL for complex, multi-objective (energy, carbon, performance) job scheduling in datacenters.
    2.  **Quantified Sustainability Improvements:** Demonstrable evidence, through extensive simulations and testbed experiments, that GreenSched can achieve significant reductions in energy costs (expected 15-30%) and carbon emissions (expected 20-40%) compared to traditional schedulers, by intelligently leveraging dynamic pricing, carbon intensity variations, and power management capabilities.
    3.  **Performance Trade-off Analysis:** A clear understanding and quantification of the trade-offs involved. We anticipate GreenSched can achieve sustainability gains with minimal degradation (or potentially even improvement in some cases, e.g., through better packing) in job performance metrics like average JCT and SLA compliance, depending on the configured objective weights.
    4.  **Methodological Insights:** Valuable insights into effective state representations, action space designs, and reward formulations for applying DRL to real-world systems resource management problems, particularly in the context of sustainability.
    5.  **Open-Source Artifact:** A publicly available, well-documented, and reproducible open-source software package (simulator, DRL agent, evaluation tools) that serves as a platform for future research and development in sustainable computing and ML for Systems.

*   **Impact:**
    *   **Environmental:** By enabling datacenters to operate more efficiently and utilize cleaner energy sources more effectively, GreenSched directly contributes to mitigating the environmental impact of the rapidly growing digital infrastructure. This aligns with global efforts towards decarbonization and sustainable development.
    *   **Economic:** The potential for significant energy cost savings offers a strong economic incentive for adoption by cloud providers, enterprises managing private clouds, and large HPC centers, ultimately making computing services more affordable.
    *   **Scientific and Research Community:** This research pushes the boundaries of ML for Systems, showcasing a sophisticated DRL application to a critical systems problem. The open-source release will lower the barrier for entry for other researchers, enabling comparative studies, extensions (e.g., incorporating network awareness, thermal management), and validation of new ideas, fostering reproducibility and collaboration as encouraged by the workshop. It provides a concrete step towards standardized benchmarks for sustainable scheduling.
    *   **Industry and Practice:** GreenSched offers a pathway towards more sustainable and cost-effective datacenter operations. The validation on Kubernetes increases its practical relevance, potentially leading to integration into commercial or open-source cluster management platforms, influencing future scheduler designs. Addressing the scheduling needs of LLM workloads within this sustainable framework makes it particularly relevant given current industry trends.

In conclusion, GreenSched represents a timely and impactful research direction at the intersection of machine learning, computer systems, and sustainability. By developing and validating a DRL-based scheduler that co-optimizes for energy, carbon, and performance, this work promises significant environmental and economic benefits while contributing valuable tools and insights to the research community, directly addressing the core themes of the ML for Systems workshop.