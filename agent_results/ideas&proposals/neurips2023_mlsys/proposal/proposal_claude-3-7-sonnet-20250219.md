# GreenScaler: A Carbon-Energy-Aware Reinforcement Learning Framework for Cloud Datacenter Scheduling

## 1. Introduction

### Background
Cloud datacenters are becoming increasingly critical infrastructure in our digital economy, supporting a vast array of applications from web services to artificial intelligence workloads. However, this growing computational demand comes at a significant environmental cost. Datacenters currently consume approximately 1-2% of global electricity and are responsible for about 0.3% of global carbon emissions, with these figures projected to increase substantially over the coming years. The environmental impact is further exacerbated by the rise of computationally intensive workloads like Large Language Models (LLMs), which require massive computational resources for both training and inference.

Traditional datacenter schedulers primarily optimize for performance metrics such as throughput, latency, or resource utilization, with limited consideration for energy consumption and virtually no awareness of carbon emissions. This approach neglects two critical factors that vary significantly over time and across geographic locations: electricity costs and grid carbon intensity. Electricity prices fluctuate based on supply-demand dynamics, while carbon intensity varies with the mix of renewable and non-renewable energy sources in the power grid. These variations create opportunities for intelligent workload management that could simultaneously reduce operational costs and environmental impact without compromising service quality.

Recent research has begun to explore carbon-aware computing, as evidenced by works like CarbonClipper, PCAPS, and CarbonScaler. These approaches have demonstrated the potential for reducing the carbon footprint of datacenter operations through spatiotemporal workload shifting and dynamic resource allocation. However, existing solutions often focus on specific workload types, lack adaptability to changing conditions, or fail to consider the complex interplay between performance requirements, energy consumption, and carbon emissions.

### Research Objectives
This research proposes GreenScaler, a novel reinforcement learning framework for energy and carbon-aware job scheduling in cloud datacenters. Our objectives are to:

1. Develop a comprehensive reinforcement learning approach that optimizes datacenter operations across multiple objectives: energy cost, carbon emissions, and service-level agreement (SLA) compliance.

2. Create a state representation that captures the complex dynamics of workloads, infrastructure states, energy markets, and grid carbon intensity.

3. Design a flexible action space that enables fine-grained control over job placement, resource allocation, and power management decisions.

4. Implement and evaluate the system using both realistic simulations and real-world deployments to demonstrate practical benefits.

5. Provide an open, reproducible framework that can accelerate research in sustainable computing.

### Significance
The significance of this research lies in several dimensions:

First, by dynamically responding to real-time changes in electricity prices and grid carbon intensity, GreenScaler can achieve substantial reductions in both operational costs and environmental impact. Our preliminary analysis suggests potential energy cost savings of 15-30% and carbon emission reductions of 20-40% without compromising performance.

Second, unlike existing approaches that often focus on a single optimization target, GreenScaler employs a multi-objective reinforcement learning framework that balances performance, energy efficiency, and carbon reduction, providing datacenter operators with flexible control over their priorities.

Third, this research addresses the emerging challenges in managing LLM workloads, which have unique resource consumption patterns and scalability requirements that existing schedulers are ill-equipped to handle.

Finally, by releasing an open framework, we aim to establish a benchmark for sustainable computing research, enabling fair comparison of different approaches and accelerating progress in this critical area.

## 2. Methodology

### 2.1 System Architecture

GreenScaler is designed as a hierarchical framework consisting of three main components:

1. **Monitoring and Prediction Module**: Collects real-time data on workload characteristics, server utilization, energy pricing, and grid carbon intensity. It also incorporates prediction models for short-term forecasting of these parameters.

2. **Reinforcement Learning Engine**: The core decision-making component that learns optimal scheduling policies through interaction with the environment.

3. **Scheduling and Control Interface**: Implements the actions decided by the RL engine, interfacing with infrastructure management systems (e.g., Kubernetes) to execute job placements, migrations, and power management decisions.

Figure 1 illustrates the overall architecture and data flow of the GreenScaler system.

### 2.2 Problem Formulation

We formulate the datacenter scheduling problem as a Markov Decision Process (MDP) defined by a tuple $(S, A, P, R, \gamma)$, where $S$ is the state space, $A$ is the action space, $P$ is the transition probability function, $R$ is the reward function, and $\gamma$ is the discount factor.

#### 2.2.1 State Space

The state space $S$ encompasses all relevant information needed for the scheduler to make informed decisions:

$$s_t = [J_t, N_t, E_t, C_t]$$

where:
- $J_t$ represents the set of jobs in the system at time $t$, with each job characterized by a feature vector that includes resource requirements (CPU, memory, GPU), estimated execution time, priority, and deadline.
- $N_t$ describes the current state of compute nodes, including utilization levels, power consumption, and hardware characteristics.
- $E_t$ contains information about current and predicted electricity prices across different regions.
- $C_t$ provides current and forecasted carbon intensity values for the power grid.

For each job $j \in J_t$, we represent its features as:

$$j = [r_j^{cpu}, r_j^{mem}, r_j^{gpu}, e_j, p_j, d_j, w_j]$$

where $r_j$ denotes resource requirements, $e_j$ is the estimated execution time, $p_j$ is the priority, $d_j$ is the deadline, and $w_j$ is the waiting time in the queue.

Similarly, for each node $n \in N_t$:

$$n = [u_n^{cpu}, u_n^{mem}, u_n^{gpu}, p_n, h_n]$$

where $u_n$ represents utilization levels, $p_n$ is the current power consumption, and $h_n$ captures hardware characteristics.

#### 2.2.2 Action Space

The action space $A$ consists of decisions the scheduler can make:

1. **Job Placement**: Assign job $j$ to node $n$ or delay scheduling.
   $$a_{place}(j) \in \{n_1, n_2, ..., n_k, delay\}$$

2. **Resource Allocation**: Determine the amount of resources allocated to each job.
   $$a_{alloc}(j) = [a_j^{cpu}, a_j^{mem}, a_j^{gpu}]$$

3. **Power Management**: Adjust power caps for CPUs and GPUs on each node.
   $$a_{power}(n) = [p_{cap}^{cpu}, p_{cap}^{gpu}]$$

4. **Job Migration**: Move running jobs between nodes to optimize resource utilization and energy efficiency.
   $$a_{migrate}(j) \in \{n_1, n_2, ..., n_k, stay\}$$

#### 2.2.3 Reward Function

The reward function balances multiple objectives:

$$R(s_t, a_t, s_{t+1}) = -w_e \cdot E_{cost}(s_t, a_t) - w_c \cdot C_{emissions}(s_t, a_t) - w_p \cdot P_{violations}(s_t, a_t)$$

where:
- $E_{cost}(s_t, a_t)$ is the total energy cost incurred by the action
- $C_{emissions}(s_t, a_t)$ represents the carbon emissions generated
- $P_{violations}(s_t, a_t)$ captures SLA violations, including missed deadlines and performance degradation
- $w_e$, $w_c$, and $w_p$ are weights that allow flexible prioritization of different objectives

The energy cost is calculated as:

$$E_{cost}(s_t, a_t) = \sum_{n \in N_t} p_n \cdot \Delta t \cdot price_n$$

where $p_n$ is the power consumption of node $n$, $\Delta t$ is the time step duration, and $price_n$ is the electricity price at the node's location.

Carbon emissions are computed as:

$$C_{emissions}(s_t, a_t) = \sum_{n \in N_t} p_n \cdot \Delta t \cdot intensity_n$$

where $intensity_n$ is the carbon intensity (gCO₂eq/kWh) of the electricity at the node's location.

The SLA violation penalty is defined as:

$$P_{violations}(s_t, a_t) = \sum_{j \in J_t} p_j \cdot \max(0, \frac{c_j + w_j - d_j}{d_j}) + \sum_{j \in J_t} p_j \cdot \max(0, \frac{e_j^{target} - e_j^{actual}}{e_j^{target}})$$

where $c_j$ is the completion time, $w_j$ is the waiting time, $d_j$ is the deadline, $e_j^{target}$ is the target execution time, and $e_j^{actual}$ is the actual execution time.

### 2.3 Reinforcement Learning Algorithm

We employ Proximal Policy Optimization (PPO), a state-of-the-art policy gradient method, for training our scheduler. PPO offers several advantages for this application, including sample efficiency, stable learning, and the ability to handle continuous action spaces. The objective function for PPO is:

$$L^{CLIP}(\theta) = \hat{E}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio, $\hat{A}_t$ is the estimated advantage at time $t$, and $\epsilon$ is a hyperparameter that controls the clipping range.

To handle the complex state space, we use a neural network architecture composed of:

1. **Feature Embedding Layers**: Separate embedding networks for job features, node features, and energy/carbon data.
2. **Attention Mechanism**: To model the relationships between jobs and nodes, enabling the scheduler to focus on relevant combinations.
3. **Policy Network**: Outputs a probability distribution over possible actions.
4. **Value Network**: Estimates the expected cumulative reward for state evaluation.

The neural network architecture is illustrated in Figure 2.

### 2.4 Data Collection and Simulation Environment

To train and evaluate GreenScaler, we develop a high-fidelity simulation environment that captures the essential dynamics of datacenter operations. The simulator is built on the following data sources:

1. **Workload Traces**: We use publicly available traces from Google, Alibaba, and Microsoft Azure, supplemented with synthetic LLM training and inference workloads based on published benchmarks.

2. **Energy Price Data**: Historical electricity pricing data from wholesale markets in different regions, including day-ahead and real-time prices.

3. **Carbon Intensity Data**: Grid carbon intensity measurements from sources like ElectricityMap, covering major cloud regions worldwide.

4. **Server Power Models**: Accurate power consumption models for different server types under varying workloads, validated against real hardware measurements.

The simulation environment implements a discrete-event simulation that models:
- Job arrivals and executions
- Resource allocation and utilization
- Server power consumption based on utilization
- Energy costs and carbon emissions calculation
- SLA monitoring and violation detection

### 2.5 Experimental Design

We evaluate GreenScaler through a comprehensive set of experiments designed to assess its performance across different metrics and scenarios:

#### 2.5.1 Simulation Experiments

1. **Baseline Comparison**: We compare GreenScaler against the following baselines:
   - First-Come-First-Served (FCFS)
   - Shortest Job First (SJF)
   - Resource-Aware Scheduler (prioritizing resource utilization)
   - CarbonClipper (representing state-of-the-art carbon-aware scheduling)
   - CarbonScaler (state-of-the-art approach for leveraging workload elasticity)

2. **Workload Scenarios**: We evaluate performance across different workload mixes:
   - Web services (latency-sensitive)
   - Batch processing (throughput-oriented)
   - ML training (resource-intensive, elastic)
   - LLM inference (variable load patterns)

3. **Grid Scenarios**: We simulate different power grid conditions:
   - Stable grids with predictable carbon intensity
   - Variable grids with high renewable penetration
   - Multi-region setups with different carbon profiles

4. **Sensitivity Analysis**: We assess the impact of various factors:
   - Different priority weights in the reward function
   - Prediction error in energy prices and carbon intensity
   - Various degrees of workload heterogeneity

#### 2.5.2 Real-World Deployment

After simulation validation, we implement GreenScaler in a Kubernetes-based testbed consisting of:
- 20 heterogeneous servers (mix of CPU-only and GPU-equipped nodes)
- Real-time energy monitoring using RAPL (Running Average Power Limit)
- Integration with electricity pricing and carbon intensity APIs
- Support for both containerized applications and ML workloads

The real-world experiments focus on:
1. Validating the simulation results in a controlled but realistic environment
2. Assessing the overhead and scalability of the scheduler
3. Measuring actual energy consumption and calculating emission reductions
4. Evaluating the adaptability to unexpected events (e.g., workload spikes, node failures)

### 2.6 Evaluation Metrics

We evaluate GreenScaler using the following key metrics:

1. **Energy Efficiency**:
   - Total energy consumption (kWh)
   - Energy cost ($)
   - Power Usage Effectiveness (PUE)

2. **Environmental Impact**:
   - Carbon emissions (kgCO₂eq)
   - Carbon-adjusted PUE (taking into account grid carbon intensity)
   - Renewable energy utilization percentage

3. **Performance**:
   - Job completion time (average and percentiles)
   - Deadline satisfaction rate
   - Throughput (jobs/hour)
   - Resource utilization (average and peak)

4. **System Overhead**:
   - Scheduling latency
   - Decision-making time
   - Resource consumption of the scheduler itself

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

Based on preliminary analysis and related work in the field, we anticipate the following outcomes from the GreenScaler research:

1. **Energy and Cost Reduction**: We expect to achieve 15-30% reductions in energy expenditure compared to traditional schedulers by intelligently scheduling workloads during periods of lower electricity prices and by dynamically adjusting server power states.

2. **Carbon Emission Reduction**: The carbon-aware scheduling component is anticipated to reduce CO₂ emissions by 20-40% by prioritizing workload execution during periods of higher renewable energy availability and by optimizing the geographical distribution of workloads.

3. **Performance Preservation**: These environmental and economic benefits will be achieved while maintaining or only marginally affecting job completion times, with less than 5% increase in average completion time for non-critical jobs.

4. **LLM Workload Optimization**: For LLM workloads specifically, we expect to demonstrate that carbon-aware scheduling can reduce the environmental footprint of training and inference by up to 45% with appropriate workload partitioning and placement strategies.

5. **Quantifiable Trade-offs**: The research will provide a clear characterization of the trade-offs between performance, energy consumption, and carbon emissions, enabling datacenter operators to make informed decisions based on their priorities.

6. **Open Framework**: The complete GreenScaler framework, including simulation environment, RL algorithms, and Kubernetes integration components, will be released as an open-source project to serve as a foundation for future research in sustainable computing.

### 3.2 Research Impact

The potential impact of this research extends across multiple dimensions:

**Environmental Impact**: As datacenters continue to grow in size and number, even modest percentage improvements in energy efficiency and carbon emissions can translate to significant absolute reductions. If widely adopted, the approaches developed in this research could contribute meaningfully to the tech industry's sustainability goals.

**Economic Impact**: Energy costs represent a substantial portion of datacenter operational expenses. The cost savings achieved through intelligent energy-aware scheduling can improve the economic viability of cloud services and potentially lead to reduced prices for end-users.

**Scientific Impact**: This research advances the state of the art in applying reinforcement learning to complex system optimization problems. The multi-objective formulation and the techniques developed to handle large, heterogeneous state and action spaces will be valuable contributions to the ML for Systems community.

**Practical Impact**: The open-source release of GreenScaler will provide practitioners with practical tools for implementing carbon and energy-aware scheduling in real-world environments. The Kubernetes integration ensures compatibility with widely used container orchestration systems.

**Educational Impact**: The framework will serve as an educational resource for researchers and students interested in sustainable computing, reinforcement learning, and datacenter management.

### 3.3 Future Research Directions

This research lays the groundwork for several promising future directions:

1. **Federated Carbon-Aware Computing**: Extending the approach to coordinate scheduling across multiple datacenters or even different cloud providers.

2. **Hardware-Aware Optimization**: Incorporating detailed hardware-specific power models to further refine energy optimization strategies.

3. **Application-Specific Adaptations**: Developing specialized scheduling policies for emerging workloads such as quantum computing simulations or AI-generated content.

4. **Integration with Carbon Markets**: Exploring mechanisms to connect datacenter operations with carbon offset markets or renewable energy certificate trading.

5. **Energy Storage Integration**: Incorporating battery storage systems into the scheduling decisions to further optimize for grid carbon intensity fluctuations.

In conclusion, GreenScaler represents a significant step toward more sustainable cloud computing infrastructure. By intelligently managing workloads in response to energy and carbon considerations while maintaining performance guarantees, this research addresses a critical challenge at the intersection of computing, energy systems, and environmental sustainability.