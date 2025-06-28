# CarbonGPT: Large Language Models for Carbon-Aware Workload Scheduling in Cloud Computing

## Introduction

Cloud computing has revolutionized the way computational resources are provisioned and utilized globally. However, this technological advancement comes with a significant environmental cost. Cloud datacenters now account for approximately 1% of global electricity consumption, contributing substantially to the world's carbon footprint. As climate change concerns intensify, there is a growing imperative for cloud providers to minimize their environmental impact while maintaining high-performance standards.

Traditional workload scheduling in cloud environments has primarily focused on performance optimization, cost efficiency, and meeting Service Level Agreements (SLAs). Environmental considerations, particularly carbon emissions, have typically been secondary concerns. This approach is increasingly untenable as organizations face mounting pressure from stakeholders, regulators, and consumers to reduce their environmental impact. The challenge lies in developing scheduling systems that intelligently balance performance requirements with carbon emission reduction goals.

Recent research has demonstrated the potential of carbon-aware scheduling to significantly reduce emissions. Systems such as PCAPS have achieved up to 32.9% reduction in carbon footprint (Lechowicz et al., 2025), while CASPER has shown carbon emission reductions of up to 70% for distributed web services without degrading latency performance (Souza et al., 2024). However, these approaches often rely on rule-based systems or conventional machine learning models that struggle to capture the complex interdependencies between workload characteristics, time sensitivity, geographic carbon intensity variations, and renewable energy availability.

The emergence of Large Language Models (LLMs) presents a new opportunity to address these limitations. LLMs excel at integrating diverse data sources, understanding complex patterns, and making sophisticated predictions based on multidimensional inputs. This research proposes CarbonGPT, a novel LLM-based approach to carbon-aware workload scheduling in cloud environments. CarbonGPT will leverage the pattern recognition and contextual understanding capabilities of LLMs to make intelligent scheduling decisions that minimize carbon emissions while maintaining performance requirements.

The research objectives of this proposal are:
1. To develop a specialized LLM architecture for carbon-aware workload scheduling that can integrate and reason across multiple heterogeneous data sources.
2. To design and implement a continuous learning framework that improves scheduling decisions by analyzing outcomes of previous scheduling choices.
3. To create a comprehensive evaluation framework that quantifies both carbon emission reductions and performance impacts across diverse workload types and cloud environments.
4. To demonstrate that LLM-based schedulers can achieve 15-30% greater carbon emission reductions compared to traditional and simpler ML-based scheduling approaches while maintaining SLA compliance.

The significance of this research lies in its potential to provide cloud providers with a powerful new tool for sustainability. By leveraging the sophisticated reasoning capabilities of LLMs, CarbonGPT could establish a new paradigm for carbon-aware computing that goes beyond simple heuristic replacements. Furthermore, the continuous learning framework will enable the system to adapt to evolving workload patterns, energy markets, and carbon intensity fluctuations, providing a robust solution for long-term sustainability goals in cloud computing.

## Methodology

Our methodology for developing CarbonGPT encompasses four main phases: (1) data integration and preparation, (2) LLM architecture design and training, (3) scheduler implementation, and (4) experimental evaluation. The following sections detail each phase.

### 1. Data Integration and Preparation

The effectiveness of our LLM-based scheduler depends on the quality and comprehensiveness of the data it can access. We will integrate four key data sources:

**a. Carbon Intensity Data**:
- Real-time and forecasted carbon intensity data from multiple electricity grids worldwide (gCO2eq/kWh)
- Historical carbon intensity patterns with temporal resolution of 5 minutes
- Data sources: Electricity Map API, WattTime API, and national grid operators' APIs

**b. Workload Characterization Data**:
- Resource consumption patterns (CPU, memory, I/O, network)
- Execution time distributions
- Deadline constraints and priority levels
- Inter-task dependencies
- We will collect this data from public cloud workload traces (e.g., Google, Azure, Alibaba) and create synthetic workloads that represent various application types.

**c. Datacenter Efficiency Metrics**:
- Power Usage Effectiveness (PUE) for each datacenter location
- Server efficiency profiles (power consumption under different load levels)
- Cooling system efficiency metrics
- Infrastructure-level carbon accounting data

**d. Renewable Energy Availability**:
- On-site renewable energy generation forecasts
- Power Purchase Agreement (PPA) data
- Grid-level renewable energy percentage

We will create a unified data schema that normalizes these diverse data sources into a consistent format suitable for LLM training and inference. The data will be temporally aligned and geospatially tagged to enable contextual reasoning.

### 2. LLM Architecture Design and Training

We propose a specialized LLM architecture, CarbonGPT, specifically designed for carbon-aware workload scheduling:

**a. Base Model Selection and Adaptation**:
We will start with a pre-trained transformer-based language model (e.g., GPT-4, LLaMA 3) and adapt it for the scheduling domain through:
- Domain-specific vocabulary augmentation for energy and computing concepts
- Architecture modifications to better handle numerical and temporal data
- Introduction of specialized attention mechanisms for workload-resource matching

**b. Multi-Modal Input Handling**:
The model will process multiple data types:
- Text descriptions of workloads and constraints
- Numerical time series data (carbon intensity, server loads)
- Structured data (workload properties, datacenter specifications)

**c. Fine-tuning Process**:
We will fine-tune the model using a combination of approaches:
- Supervised fine-tuning using historical scheduling decisions and their outcomes
- Reinforcement learning from environmental feedback (carbon emissions, performance metrics)
- Few-shot learning to adapt to new datacenter environments

The training objective function will balance multiple factors:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{carbon}} + \beta \cdot \mathcal{L}_{\text{performance}} + \gamma \cdot \mathcal{L}_{\text{feasibility}}$$

Where:
- $\mathcal{L}_{\text{carbon}}$ represents the carbon emission minimization objective
- $\mathcal{L}_{\text{performance}}$ captures performance objectives (latency, throughput)
- $\mathcal{L}_{\text{feasibility}}$ ensures scheduling decisions are practical and respect constraints
- $\alpha$, $\beta$, and $\gamma$ are weighting coefficients that can be adjusted based on priority

**d. Continual Learning Framework**:
To ensure the model remains effective as conditions change, we will implement a continual learning system that:
- Collects feedback from actual scheduling outcomes
- Periodically retrains on recent data to capture changing patterns
- Uses a knowledge distillation process to incorporate new insights without catastrophic forgetting

### 3. Scheduler Implementation

The CarbonGPT scheduler will be implemented as a complete system with the following components:

**a. Input Processing Pipeline**:
- Data collectors for each source (carbon intensity, workload metrics, etc.)
- Preprocessing modules for data cleaning and normalization
- Feature engineering to extract relevant signals

**b. Scheduling Decision Engine**:
The core scheduling algorithm operates as follows:

1. For each scheduling interval $t$:
   - Collect the current state $S_t$ of the system (available resources, running jobs)
   - Gather the queue $Q_t$ of pending workloads
   - Obtain carbon intensity forecasts $C_t$ for all datacenter locations
   - Estimate energy consumption profiles $E_{i,j}$ for each workload $i$ on resource $j$

2. The LLM generates a scheduling plan $P_t$ that maps workloads to resources and execution times by solving:

$$P_t = \arg\min_{P} \sum_{i \in Q_t} \sum_{j \in R} \sum_{\tau=t}^{t+H} E_{i,j,\tau} \cdot C_{j,\tau} \cdot x_{i,j,\tau}$$

Subject to constraints:
- Resource capacity constraints: $\sum_{i} r_{i,k} \cdot x_{i,j,\tau} \leq R_{j,k,\tau}$ for each resource type $k$
- Deadline constraints: $\sum_{\tau=t}^{D_i} x_{i,j,\tau} \cdot p_{i,j,\tau} \geq w_i$ for each workload $i$
- Precedence constraints: $s_i \geq c_j$ if workload $i$ depends on workload $j$

Where:
- $x_{i,j,\tau}$ is a binary decision variable indicating if workload $i$ is scheduled on resource $j$ at time $\tau$
- $H$ is the planning horizon
- $r_{i,k}$ is the requirement of workload $i$ for resource type $k$
- $R_{j,k,\tau}$ is the capacity of resource $j$ for resource type $k$ at time $\tau$
- $D_i$ is the deadline for workload $i$
- $p_{i,j,\tau}$ is the processing rate of workload $i$ on resource $j$ at time $\tau$
- $w_i$ is the work requirement of workload $i$
- $s_i$ and $c_j$ are the start and completion times of workloads $i$ and $j$

3. The LLM approach allows us to incorporate complex reasoning that purely mathematical optimization would struggle with, such as:
   - Workload urgency assessment
   - Carbon intensity trend analysis
   - Prediction of resource contention
   - Consideration of migration costs

**c. Execution and Monitoring System**:
- Implementation interfaces with popular cloud orchestrators (Kubernetes, OpenStack)
- Real-time monitoring of executed schedules
- Feedback collection for continuous improvement

### 4. Experimental Evaluation

We will evaluate CarbonGPT through a comprehensive set of experiments:

**a. Simulation Environment**:
We will develop a simulation environment that models:
- Multiple geographically distributed datacenters
- Realistic carbon intensity variations based on historical data
- Diverse workload types with varying resource profiles and constraints
- Renewable energy availability patterns

**b. Workload Scenarios**:
We will test the system under various workload scenarios:
- Batch processing workloads with flexible deadlines
- Interactive services with strict latency requirements
- Mixed workloads that represent real-world cloud environments
- Specialized workloads like AI training jobs with unique characteristics

**c. Comparative Analysis**:
We will compare CarbonGPT against:
- Carbon-agnostic schedulers (e.g., FIFO, fair-share)
- Rule-based carbon-aware schedulers
- Traditional ML-based schedulers (e.g., using regression or reinforcement learning)
- State-of-the-art systems from literature (CASPER, PCAPS, CarbonClipper)

**d. Evaluation Metrics**:
We will assess performance across multiple dimensions:

*Carbon Efficiency Metrics:*
- Total carbon emissions (gCO2eq)
- Carbon efficiency: useful work per carbon emitted (operations/gCO2eq)
- Carbon displacement: emissions saved compared to baseline

*Performance Metrics:*
- Job completion time
- SLA compliance rate
- Resource utilization
- Throughput

*System Metrics:*
- Scheduling latency
- Scalability with increasing workload and datacenter size
- Adaptability to changing conditions

**e. Real-world Deployment**:
Following successful simulation results, we will implement a prototype deployment in a real cloud environment:
- Multi-region setup with at least 3 geographically distributed datacenters
- Integration with actual carbon intensity data sources
- Testing with representative workloads
- Measurement of actual energy consumption and carbon emissions

This comprehensive evaluation framework will allow us to quantify both the carbon reduction potential and the performance characteristics of CarbonGPT under diverse conditions, providing robust evidence for its effectiveness.

## Expected Outcomes & Impact

This research is expected to produce several significant outcomes with far-reaching impact on sustainable cloud computing:

### Technical Outcomes

1. **Novel LLM Architecture for Carbon-Aware Scheduling**: We will develop a specialized LLM architecture designed specifically for integrating and reasoning across heterogeneous data sources relevant to carbon-aware scheduling. This architecture will demonstrate how LLMs can move beyond text processing to solve complex system optimization problems.

2. **Carbon-Aware Scheduling Algorithm**: Our research will deliver a comprehensive scheduling algorithm that leverages LLM capabilities to make sophisticated decisions balancing carbon reduction with performance requirements. We expect this algorithm to achieve 15-30% greater carbon emission reductions compared to existing approaches while maintaining SLA compliance.

3. **Continual Learning Framework**: The continuous learning system will provide an adaptive scheduling solution that improves over time by learning from actual outcomes. This framework will establish a methodological foundation for self-improving systems optimization in dynamic cloud environments.

4. **Open-Source Reference Implementation**: We will release an open-source implementation of CarbonGPT, including data collection interfaces, model architecture, and scheduling components. This will enable cloud providers and researchers to adopt, extend, and build upon our approach.

5. **Evaluation Dataset and Benchmark**: The comprehensive evaluation framework and associated datasets will serve as a benchmark for future research in carbon-aware computing, allowing for standardized comparison of different approaches.

### Environmental Impact

1. **Carbon Emission Reduction**: Based on our preliminary analysis and the results of related work, we estimate that widespread adoption of CarbonGPT could reduce cloud computing carbon emissions by 15-30% compared to carbon-agnostic approaches. Given that cloud datacenters account for approximately 1% of global electricity consumption, this represents a significant potential impact on global carbon emissions.

2. **Renewable Energy Utilization**: By intelligently scheduling workloads to align with renewable energy availability, CarbonGPT will increase the effective utilization of renewable energy sources. This contributes to the economic viability of renewable energy investments and accelerates the transition to cleaner energy sources.

3. **Energy Efficiency Improvements**: Beyond carbon reduction, our approach will identify and implement energy efficiency optimizations that reduce overall power consumption, providing economic benefits alongside environmental ones.

### Industry and Research Impact

1. **New Paradigm for Sustainable Computing**: This research will establish a new paradigm for how AI, particularly LLMs, can be applied to sustainability challenges in computing infrastructure. It demonstrates a path forward that leverages advanced AI for environmental goals.

2. **Practical Tools for Cloud Providers**: Cloud providers will gain practical tools and methodologies to reduce their carbon footprint while maintaining competitive performance, helping them meet increasingly stringent sustainability targets and regulatory requirements.

3. **Cross-Disciplinary Research Advancement**: This work bridges multiple domains including machine learning, systems research, and environmental science. The methodologies and findings will contribute to advancing research at these intersections.

4. **Educational Resources**: The open-source implementation and documentation will serve as educational resources for training the next generation of researchers and practitioners in sustainable computing.

### Broader Societal Impact

1. **Corporate Sustainability Goals**: As organizations increasingly adopt cloud computing, CarbonGPT will provide them with a transparent mechanism to reduce the environmental impact of their digital operations, supporting corporate sustainability commitments.

2. **Policy and Standards Development**: The metrics, methodologies, and results from this research can inform the development of standards and policies related to sustainable computing, providing evidence-based foundations for regulatory frameworks.

3. **Public Awareness**: By quantifying and demonstrating the environmental impact of cloud computing and potential pathways to improvement, this research contributes to public awareness and discourse on digital sustainability.

In summary, CarbonGPT represents a significant advancement in sustainable cloud computing that leverages the latest developments in AI to address pressing environmental challenges. The expected 15-30% reduction in carbon emissions, if widely adopted, would make a meaningful contribution to global carbon reduction efforts while establishing new methodological approaches for applying LLMs to systems optimization problems.