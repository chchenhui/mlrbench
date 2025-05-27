1. Title  
Adaptive Closed-Loop Generative Modeling and Bayesian Optimization for Efficient Protein Engineering  

2. Introduction  
Background  
The artificial design of proteins with tailored functions has transformative potential across medicine, industry and environmental applications. Generative machine learning (ML) methods—including variational autoencoders (VAEs), language models and diffusion models—have recently achieved striking success in proposing de novo protein sequences (Winnifrith et al., 2023; Kouba et al., 2024). However, most efforts remain “in silico,” optimizing static benchmarks without directly guiding wet-lab exploration. As a result, many computationally generated candidates fail experimental validation, leading to high false-positive rates and wasted resources (Calvanese et al., 2025).  

Research Objectives  
We propose to develop an adaptive, closed-loop framework that tightly couples generative ML with Bayesian optimization and real-time experimental feedback. Our objectives are threefold:  
• To build a generative model (VAE) that captures the complex landscape of protein sequence–function relationships, enriched by successive rounds of wet-lab data.  
• To integrate a Bayesian optimization module that selects a small batch of sequences per iteration, balancing exploration of unknown sequence regions with exploitation of high-potential candidates.  
• To implement a closed-loop optimization pipeline that iteratively updates both the generative model and the selection policy based on experimental measurements, thereby focusing resources on the most promising design regions.  

Significance  
By combining generative ML and adaptive experimental design, this work directly addresses two critical challenges in protein engineering: (1) the vastness of sequence space (>10^130 possible 100-residue proteins) and (2) the high cost/time of wet-lab screening. Our method is expected to reduce experimental assays by ≥80% compared to conventional high-throughput screening, accelerate discovery of functional proteins, and establish a general paradigm for ML-driven biomolecular design that is immediately translatable to industrial and academic laboratories.  

3. Methodology  
3.1 Overview of the Adaptive Closed-Loop Framework  
Our framework (Figure 1) operates in iterations \(t = 1\dots T\). Each iteration comprises four steps: (i) generation of candidate sequences via a VAE, (ii) surrogate modeling and acquisition to select a small batch for testing, (iii) wet-lab functional assays, and (iv) model updates. Over successive rounds, the system refines its internal models, concentrating exploration on regions with high functional potential.  

3.2 Data Collection and Preprocessing  
We will begin with a seed dataset \(D_0 = \{(x_i, y_i)\}_{i=1}^{N_0}\) comprising \(\sim\!10^4\) protein sequences \(x_i\) of length 80–120 amino acids, each annotated with experimental fitness measurements \(y_i\) (e.g., catalytic rate, binding affinity). Sources include public enzyme databases (BRENDA, Uniprot) and high-throughput microfluidic assays (Lee & Kim, 2023). Sequences are one-hot encoded into vectors in \(\{0,1\}^{L\times20}\) and subsequently embedded via learned representations in a continuous latent space.  

3.3 Generative Model: Variational Autoencoder  
We employ a VAE to model the sequence distribution. Let \(x\) denote a one-hot sequence and \(z\in\mathbb{R}^d\) the latent code. The encoder \(q_\phi(z|x)\) and decoder \(p_\theta(x|z)\) are parameterized by deep neural networks. We train by maximizing the evidence lower bound (ELBO):  
$$  
\mathcal{L}(\theta,\phi) \;=\; \sum_{i=1}^{N_t} \Bigl[ \mathbb{E}_{q_\phi(z|x_i)}[\log p_\theta(x_i|z)] \;-\; D_{KL}\bigl(q_\phi(z|x_i)\,\|\,p(z)\bigr)\Bigr],  
$$  
where \(p(z)=\mathcal{N}(0,I)\) is the prior. After each experimental round, the VAE is fine-tuned on the augmented dataset \(D_t\), biasing generation toward regions exhibiting functional improvements (Johnson & Williams, 2024).  

3.4 Surrogate Modeling and Acquisition Function  
To quantify the relationship between sequence \(x\) and function \(y\), we train a Gaussian process (GP) or deep kernel model \(g_\psi(x)\) on \(D_t\). The model predicts a mean \(\mu_t(x)\) and uncertainty \(\sigma_t(x)\) for any candidate. We define an acquisition function that balances exploration (high \(\sigma\)) and exploitation (high \(\mu\)) while enforcing diversity via a penalty term \(\Delta(x, S_{t-1})\):  
$$  
\alpha_t(x) = \mu_t(x) + \kappa\,\sigma_t(x)\;-\;\lambda\,\Delta(x, S_{t-1}).  
$$  
Here \(\kappa\) controls exploration–exploitation trade-off and \(\lambda\) penalizes redundancy. We measure diversity \(\Delta\) as the minimum Hamming distance to previously selected sequences.  

3.5 Closed-Loop Iterative Algorithm  
Algorithm 1 outlines the complete pipeline.  

Algorithm 1: Adaptive Design Space Exploration  
Input: Initial dataset \(D_0\), iteration budget \(T\), batch size \(k\).  
for \(t = 1\) to \(T\) do  
  1. Sample latent codes \(z_j\sim p(z)\), generate \(M\gg k\) candidates \(\{x_j\}\sim p_\theta(x|z_j)\).  
  2. For each \(x_j\), compute \(\alpha_t(x_j)\).  
  3. Select top \(k\) sequences \(S_t = \arg\max_{S:|S|=k}\sum_{x\in S}\alpha_t(x)\).  
  4. Synthesize and assay \(S_t\) in wet lab; measure true fitness values \(\{y_j\}\).  
  5. Update dataset: \(D_t = D_{t-1}\cup\{(x,y)\mid x\in S_t\}\).  
  6. Retrain/fine-tune surrogate \(g_\psi\) on \(D_t\).  
  7. Fine-tune VAE parameters \((\theta,\phi)\) on \(D_t\).  
end for  

3.6 Experimental Design  
We will validate the framework on two case studies: (a) an industrially relevant hydrolase (length ≈100), and (b) a high-affinity binder to a viral antigen (length ≈120). For each case:  
• Iterations: \(T=5\) rounds.  
• Batch size: \(k=50\) sequences per round.  
• Wet-lab assay: microfluidic droplet screening for enzymatic turnover (Case a) or yeast display & flow cytometry for binding (Case b).  
• Replication: each sequence measured in triplicate to estimate mean and variance.  
• Cost Estimate: \(\$50\) per sequence; total \(\approx\$12{,}500\) per case, versus \(\sim\$100{,}000\) for exhaustive screening of 1{,}000 sequences.  

3.7 Evaluation Metrics and Baselines  
We will compare our adaptive method against three baselines:  
1. Random Screening: random sampling of \(k\) sequences per round.  
2. Static Generative Model: VAE generation without feedback, top-\(k\) by predicted \(\mu\).  
3. Pure Bayesian Optimization: BO directly over sequence space (no VAE prior).  

Performance will be assessed by:  
• Discovery Efficiency: number of true “hits” (exceeding a functional threshold) discovered per experiment.  
• Time to Threshold: total experiments needed to reach a pre-defined fitness level \(y^*\).  
• Predictive Accuracy: correlation (Spearman’s \(\rho\)) between surrogate predictions \(\mu_t(x)\) and true outcomes.  
• Cost Reduction: ratio of experiments required vs. exhaustive screening.  

Statistical significance will be evaluated via bootstrapping (1{,}000 resamples), reporting 95% confidence intervals.  

4. Expected Outcomes & Impact  
4.1 Anticipated Technical Outcomes  
We anticipate that our closed-loop framework will:  
• Achieve ≥5× faster discovery of functional proteins compared to static generative or random strategies.  
• Reduce the number of wet-lab assays by ≥80% to reach industrially relevant performance thresholds.  
• Demonstrate increasing predictive fidelity of the surrogate model (\(\rho>0.8\)) over iterations, indicating effective feedback integration.  

4.2 Scientific and Practical Impact  
This work will establish a generalizable paradigm for ML-guided biomolecular design, addressing the critical gap between in silico modeling and experimental validation highlighted by the GEM workshop. By open-sourcing the pipeline—including code for VAE training, surrogate modeling and acquisition, and wet-lab protocols—we will enable wider adoption in both academic and industrial settings.  

4.3 Broader Implications  
Beyond protein engineering, our framework can be extended to nucleic acids, small molecules and materials design, wherever the combinatorial explosion of candidates and limited experimental budgets pose bottlenecks. The method promotes a virtuous cycle—ML models propose candidates, experiments validate and refine models—that can accelerate discovery across the life sciences.  

In summary, this proposal outlines a rigorous, mathematically principled and experimentally validated closed-loop system for protein engineering that bridges the disconnect between generative ML and wet-lab experimentation. The expected gains in efficiency, cost reduction and discovery speed will have lasting impact on how novel biomolecules are designed and deployed.