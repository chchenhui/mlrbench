Title  
LLM-Driven Carbon-Aware Workload Scheduling for Sustainable Cloud Computing  

1. Introduction  
Background  
Cloud datacenters today consume roughly 1% of global electricity, and this share is rising as AI-driven services scale. Traditional schedulers focus on performance (latency, throughput) and cost, but largely ignore the spatiotemporal variability of grid carbon intensity, renewable availability, and complex workload interdependencies. Recent work (PCAPS, CASPER, CarbonClipper, CarbonScaler) has shown that even rule-based or classical ML methods can achieve 15–70% carbon reductions by shifting or throttling jobs. However, these approaches often rely on handcrafted heuristics, simple predictors, or decoupled optimization steps, and they struggle to capture rich, unstructured data or adapt rapidly to evolving conditions.  

Research Objectives  
This proposal aims to develop and evaluate a novel scheduling system—GreenSched—that leverages large language models (LLMs) to predict carbon emissions, energy consumption, and performance metrics for candidate placement decisions, and then jointly optimize carbon impact and SLA compliance in real time. Our specific objectives are:  
1. Construct an LLM-based predictor $f_\theta$ that, given a structured prompt describing current workload features, grid carbon data, renewable forecasts, and datacenter efficiency metrics, outputs per-job estimates of energy use $E$ and service time $s$ for each candidate datacenter.  
2. Integrate $f_\theta$ into an online scheduling optimizer that, at each timeslot $t$, assigns incoming jobs to datacenters $d\in D$ by minimizing a weighted objective combining predicted carbon cost and SLA penalties.  
3. Design a continuous learning loop: collect actual $E_{\mathrm{real}}$, $s_{\mathrm{real}}$ after execution, and fine-tune $f_\theta$ to improve prediction accuracy over time.  
4. Evaluate end-to-end performance via trace-driven simulation and small-scale cluster experiments, comparing carbon reduction, SLA adherence, and overhead against state-of-the-art baselines (PCAPS, CASPER, CarbonClipper, CarbonScaler, plus a pure ML regressor).  

Significance  
GreenSched will be the first system to harness the representational power of LLMs—trained on heterogeneous time-series, tabular, and textual metadata—to model complex dependencies among workload types, grid variability, and datacenter characteristics. By unifying prediction and optimization within a single model architecture, we expect to achieve 15–30% additional carbon savings over classical schedulers without sacrificing SLAs. The resulting software and datasets will be open-source, accelerating sustainable cloud practices across industry and research.  

2. Methodology  
Overview  
GreenSched consists of three main components: (a) data ingestion and prompt construction, (b) LLM-based prediction and scheduling optimizer, and (c) continuous learning and feedback. We describe each in detail, including mathematical formulation, algorithmic steps, and experimental design.  

2.1 Data Collection and Preprocessing  
• Workload traces: We will use publicly available traces (Google cluster, Azure, Alibaba) containing job arrival times, CPU/GPU/memory requirements, deadlines or latency SLOs, and historical placement decisions.  
• Carbon intensity: Real-time and historical grid carbon data will be sourced from electricityMap and national grid APIs, yielding $I(t,d)\,[\mathrm{gCO_2/kWh}]$ for each datacenter location.  
• Renewable forecasts: Short-term forecasts of wind and solar availability $R(t,d)$ will be obtained from regional ISOs.  
• Datacenter metrics: Efficiency measures (PUE, utilization cap) and cost parameters.  

Preprocessing merges these streams into synchronized time slices $\{t_1,\dots,t_T\}$ at a chosen granularity (e.g. 5 min). For each job $w$ arriving at time $t$, we assemble a feature vector $x_{t,w}$ containing:  
– Job descriptors: $(\mathrm{CPU},\mathrm{GPU},\mathrm{memory},\mathrm{deadline})$  
– Real-time grid data: $\{I(t,d)\}_{d\in D}$  
– Renewable forecasts: $\{R(t+\Delta,d)\}_{\Delta\in\{5,15,60\},\,d\in D}$  
– Datacenter efficiencies: $\mathrm{PUE}(d)$, network latency profiles, etc.  

2.2 LLM-Based Prediction Model  
We adopt a pre-trained transformer language model (e.g. LLaMA-2, GPT-style) and fine-tune it to perform joint prediction of energy consumption and service time across candidate datacenters.  

Prompt format (textual):  
“Time: 2025-03-01T12:05. Job: CPU=4 cores, GPU=0, Mem=8 GB, Deadline=30 min. Grid carbon: [us-east:250, us-west:200, eu-central:300] gCO₂/kWh. Renewables: [us-east: wind=40%, solar=20%; …]. PUE: [us-east=1.1, …]. Where should this job run? Please predict for each region: energy_kWh and service_time_min.”  

Let $f_\theta(x_{t,w})$ produce for each candidate datacenter $d$:  
$$
(E_{t,w}(d),\;s_{t,w}(d)) \;=\; f_\theta(x_{t,w}, d).
$$  

Training objective: we fine-tune on historical tuples $\{(x_{t,w}, d_{\mathrm{chosen}}, E_{\mathrm{real}}, s_{\mathrm{real}})\}$ to minimize a composite loss:  
$$
\mathcal{L}(\theta)\;=\;\frac{1}{N}\sum_{i=1}^N\Big[
\lambda_E\,(E_i^{(\mathrm{pred})}-E_i^{(\mathrm{real})})^2
\;+\;\lambda_s\,(s_i^{(\mathrm{pred})}-s_i^{(\mathrm{real})})^2
\;+\;\lambda_a\,\mathrm{CE}(p_i,p_i^{(\mathrm{oracle})})
\Big],
$$  
where CE is cross-entropy on scheduling actions (when fine-tuning to mimic high-quality decisions), and $\{\lambda_E,\lambda_s,\lambda_a\}$ are weighting hyperparameters.  

2.3 Scheduling Optimization  
At inference time, for each job $w$ at time $t$, we query $f_\theta(x_{t,w})$ to obtain $(E_{t,w}(d),s_{t,w}(d))$ for all $d\in D$. We then solve a lightweight per-job assignment:  
$$
a_{t,w}\;=\;\arg\min_{d\in D}\; 
\underbrace{E_{t,w}(d)\times I(t,d)}_{\text{carbon cost}}
\;+\;\underbrace{\alpha\,\max\!\bigl(0,s_{t,w}(d)-\tau_w\bigr)}_{\text{SLA penalty}},
$$  
where $\tau_w$ is the deadline/SLA threshold, and $\alpha$ trades off carbon versus performance. This greedy step runs in $O(|D|)$ and can be parallelized across jobs. For bursty arrivals, we also explore a batch assignment variant using integer programming:  
$$
\min_{\{a_{t,w}\}}\sum_{w} E_{t,w}(a_{t,w})\,I(t,a_{t,w})
+ \alpha\,\max(0,s_{t,w}(a_{t,w})-\tau_w)
\quad
\text{s.t. capacity constraints on each }d.
$$  

2.4 Continuous Learning Loop  
After execution, we record $(E_{\mathrm{real}},s_{\mathrm{real}})$ and update $f_\theta$ periodically (e.g. daily or weekly) using collected logs. This online fine-tuning corrects model drift due to changing workloads or hardware upgrades.  

2.5 Experimental Design and Evaluation Metrics  
Benchmarks  
– Public traces: Google cluster, Azure HPC jobs.  
– Simulated geo-distributed cloud with 10 datacenters spanning North America, Europe, and Asia.  

Baselines  
1. Performance-only scheduler (minimize predicted latency).  
2. PCAPS (carbon + precedence heuristics).  
3. CASPER (threshold-based carbon alignment).  
4. CarbonClipper (optimal online allocation).  
5. CarbonScaler (elasticity ML).  
6. Pure ML regressor (XGBoost) replacing LLM in our pipeline.  

Metrics  
• Carbon Reduction (%):  
$$
100\times\bigl(1-\tfrac{\mathrm{CO}_2^{\mathrm{method}}}{\mathrm{CO}_2^{\mathrm{baseline}}}\bigr).
$$  
• SLA Violation Rate: fraction of jobs with $s_{\mathrm{real}}>\tau_w$.  
• Energy Consumption (kWh) and cost (\$).  
• Scheduling Overhead: time per scheduling decision.  
• Prediction Error: RMSE on $E$ and $s$.  

Ablations  
– Remove carbon term ($\alpha=0$).  
– Disable continuous learning.  
– Limit LLM context to show benefit of richer prompts.  

Statistical Analysis  
We will run each experiment for 30 independent seeds and report mean ± 95% confidence intervals. Significance will be tested via paired t-tests.  

Implementation Details  
• Base LLM: LLaMA-2 7B, fine-tuned with LoRA.  
• Training: batch size 16, learning rate 5e-5, up to 10 epochs.  
• Scheduler: Python + PyTorch + OR-Tools for batch IP.  
• Simulator: built on CloudSim extended with carbon APIs.  

3. Expected Outcomes & Impact  
Expected Outcomes  
– Prediction accuracy: RMSE ≤ 5% for energy, ≤ 3% for service time after fine-tuning.  
– Carbon savings: 15–30% relative to PCAPS and CarbonScaler, 5–10% beyond pure ML baseline.  
– SLA compliance: ≥ 99% jobs meeting deadlines.  
– Scheduling overhead: sub-10 ms per job in greedy mode, sub-1 s per batch in IP mode.  
– Continuous learning yields a 10% error reduction over static models.  

Scientific Contributions  
1. Demonstration that LLMs can serve as unified predictors for spatiotemporal scheduling under heterogeneous data.  
2. Novel joint prediction-optimization framework with continuous adaptation.  
3. Open-source release of code, simulated geo-distributed testbed, and synthetic trace generator.  

Practical Impact  
Cloud providers can incorporate GreenSched to meet corporate sustainability goals, reduce carbon taxes or purchase fewer offsets, and advertise low-carbon compute offerings. The system’s flexibility allows rapid integration of new data sources (e.g. local air quality, real-time pricing), paving the way for broader AI-driven resource management.  

4. Conclusion and Future Work  
We propose GreenSched, the first LLM-driven carbon-aware scheduler that jointly models energy, carbon, and performance in a unified prediction-optimization loop. By fine-tuning a large language model on rich, heterogeneous data and embedding it within a lightweight online optimizer, we anticipate significant carbon savings without harming SLAs.  

Future extensions include:  
– Reinforcement-learning formulations where the LLM acts as a policy network and directly optimizes long-term carbon objectives.  
– Integration of spot-market electricity prices for cost-aware, carbon-aware co-optimization.  
– Deployment on a real cloud testbed (e.g. Kubernetes cluster across multi-region nodes) to validate end-to-end viability.  

This project will establish a new paradigm—LLM-assisted systems for sustainable computing—and drive progress toward greener, AI-powered datacenters.