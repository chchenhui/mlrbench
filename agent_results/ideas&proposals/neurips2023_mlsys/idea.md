Title: GreenSched – Reinforcement Learning–Driven Energy-Carbon-Aware Job Scheduling

Motivation:  
Cloud datacenters consume enormous energy and incur significant carbon emissions, yet existing schedulers optimize for throughput or latency without accounting for real-time energy costs or grid carbon intensity. An adaptive, ML-based scheduler can reduce both operational expenses and environmental impact while respecting service-level objectives.

Main Idea:  
We propose GreenSched, a Deep Reinforcement Learning (DRL) scheduler that continuously ingests workload features, node utilization, dynamic electricity pricing, and grid carbon intensity forecasts.  
• State Representation: per-job resource demands, queue times, current utilization, short-term renewable generation and carbon intensity predictions.  
• Action Space: assign or delay jobs across heterogeneous servers, adjust CPU/GPU power caps, trigger VM migrations.  
• Reward Signal: weighted combination of negative energy cost, negative carbon emissions, and SLA-violation penalties.  
Training occurs in a high-fidelity simulator seeded with real trace data; the learned policy is then fine-tuned in a Kubernetes testbed. We anticipate 15–30% reductions in energy expenditure and 20–40% lower CO₂ footprint without degrading job completion times. GreenSched will be released as an open, reproducible framework to accelerate sustainable computing research.