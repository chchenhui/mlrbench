1. Title  
Adaptive Model-Assisted Dataset Construction with Diversity-Aware Feedback Loops  

2. Introduction  
Background  
Recent advances in large-scale foundation models have largely been driven by richer architectures and ever-growing compute.  However, as model capacity soars, the bottleneck has shifted from model design to the quality and diversity of data.  Workshop themes such as data provenance, dataset drifts, and model-assisted curation underscore that indiscriminate scaling of uncurated data often induces bias, reduces robustness under distribution shifts, and incurs excessive annotation costs.  Pioneering studies—Wyllie et al. (2024) on fairness feedback loops, Erfanian et al. (2024) on multi-modal augmentation, Yu et al. (2023) on diverse feedback for language alignment, and Taori & Hashimoto (2022) on stability of data-model ecosystems—collectively reveal four core challenges:  
- Bias amplification in synthetic feedback loops  
- Quality degradation in large-scale augmentation  
- Difficulty integrating heterogeneous feedback signals  
- Instability in continuous model-data interactions  

Research Objectives  
This proposal targets these challenges by developing an adaptive, iterative framework for dataset construction that explicitly monitors and maximizes diversity while preserving quality.  Our objectives are:  
1. To design a diversity-aware feedback loop that identifies underrepresented patterns in the latent embedding space and generates targeted synthetic samples.  
2. To integrate active learning–based human validation that maintains high annotation efficiency and ensures fidelity to the true data distribution.  
3. To define continuous metrics of diversity, coverage, and cross-model consistency, and use them to drive loop convergence.  
4. To demonstrate, via rigorous experiments in domains such as biomedical imaging and climate science, that our approach reduces annotation costs by 30–50% and improves downstream robustness under domain shift.  

Significance  
By combining principles from fairness-aware augmentation (Chameleon), feedback-loop stability analysis (Data Feedback Loops), and constructive diverse feedback (CDF), this research will:  
- Advance data-centric methodology for foundation models beyond vision and language into emerging domains.  
- Provide a modular pipeline that practitioners can adapt to any domain with minimal human labeling.  
- Promote ethical dataset curation through explicit bias monitoring and mitigation.  

3. Methodology  
3.1 Overview of the Iterative Framework  
Our framework proceeds in discrete iterations $t=0,1,2,\dots$ over a dataset $D_t$.  Each iteration comprises four stages:  
A. Model Initialization  
B. Diversity-Aware Synthetic Data Generation  
C. Active Learning–Driven Human Validation  
D. Metric Computation and Loop Update  

3.2 Data Collection and Initialization  
We assume access to:  
- A small seed dataset $D_0 = \{(x_i,y_i)\}_{i=1}^{N_0}$, labeled in domain of interest (e.g., histopathology images).  
- A large unlabeled pool $U = \{x_j\}_{j=1}^{M}$.  
Stage A trains an initial foundation model $M_0$ on $D_0$ using standard empirical risk minimization.  For classification tasks we optimize:  
$$
\min_{\theta} \frac{1}{N_0}\sum_{i=1}^{N_0} \mathcal{L}\bigl(M_0(x_i;\theta),\,y_i\bigr)\,,
$$  
where $\mathcal{L}$ is cross-entropy loss.  

3.3 Diversity-Aware Synthetic Data Generation  
We embed both $D_t$ and $U$ into a latent space via the penultimate layer of $M_t$, producing embeddings $E_t = \{e_i\}$.  We apply K-means clustering into $K$ clusters $\{C_k\}_{k=1}^K$.  Let $n_{t,k}=|C_k\cap E_t|$ be the count of real samples in cluster $k$.  Define a cluster rarity score:  
$$
r_{t,k} \;=\;\frac{\max_j n_{t,j} - n_{t,k}}{\max_j n_{t,j}}\,,\quad 0\le r_{t,k}\le1.
$$  
Clusters with $r_{t,k}>\tau$ (e.g.\ $\tau=0.5$) are deemed underrepresented.  For each such cluster, we generate $S_{t,k}$ synthetic samples via the foundation model’s generative head (e.g.\ diffusion or autoregression) conditioned on cluster centroid embedding $\mu_{t,k}$.  The synthetic pool at iteration $t$ is  
$$
S_t = \bigcup_{k\,:\,r_{t,k}>\tau} \bigl\{x^{\mathrm{syn}}_{k,1},\dots,x^{\mathrm{syn}}_{k,m}\bigr\},  
$$  
where $m$ is selected to balance overall dataset scale.  

3.4 Active Learning–Driven Human Validation  
To verify quality and fill critical gaps, we perform active sampling from $S_t\cup U$.  For each candidate $x$, compute predictive entropy under an ensemble of $L$ models $\{M_t^{(\ell)}\}$:  
$$
H(x) = -\sum_{c=1}^C \overline{p}_c(x)\,\log \overline{p}_c(x)\,,\quad 
\overline{p}_c(x)=\frac1L\sum_{\ell=1}^L P\bigl(y=c\mid x;M_t^{(\ell)}\bigr).
$$  
We select the top $B$ high-entropy samples and present them to human annotators.  The newly annotated set $H_t$ is added to the dataset:  
$$
D_{t+1} = D_t \,\cup\, \bigl\{(x,y)\,:\,x\in H_t\bigr\}\,\cup\, \bigl\{(x,y)\,:\,x\in S_t,\,y\ \text{model-predicted with high confidence}\bigr\}.
$$  
We then retrain or fine-tune $M_t$ to obtain $M_{t+1}$.  

3.5 Continuous Metrics and Convergence Criteria  
We monitor three key metrics:  
1. Diversity ($\mathcal{D}_t$): measured by Shannon entropy of cluster distribution:  
   $$
   \mathcal{D}_t = -\sum_{k=1}^K \frac{n_{t,k}}{\sum_j n_{t,j}}\;\log\!\Bigl(\frac{n_{t,k}}{\sum_j n_{t,j}}\Bigr).
   $$  
2. Coverage ($\mathcal{C}_t$): Jensen-Shannon divergence between current cluster distribution and uniform distribution $U_k=1/K$:  
   $$
   \mathcal{C}_t = \mathrm{JSD}\!\bigl(\{n_{t,k}\}\,\|\,\{U_k\}\bigr).
   $$  
3. Quality ($\mathcal{Q}_t$): average cross-model agreement (negative average KL divergence):  
   $$
   \mathcal{Q}_t = -\frac{2}{L(L-1)}\sum_{\ell<i}\mathrm{KL}\bigl(P_\ell\|P_i\bigr).
   $$  
The loop terminates when improvements $\Delta\mathcal{D}_t,\Delta\mathcal{C}_t,\Delta\mathcal{Q}_t$ fall below thresholds $\epsilon_D,\epsilon_C,\epsilon_Q$ or a labeling budget is exhausted.  

3.6 Algorithm  
Pseudocode for iteration $t$:  
1. Compute embeddings $E_t$ and clusters $\{C_k\}$.  
2. Identify underrepresented clusters $\{k:r_{t,k}>\tau\}$.  
3. Generate synthetic pool $S_t$ from cluster centroids.  
4. Compute ensemble entropy $H(x)$ for $x\in S_t\cup U$.  
5. Select top-$B$ samples for human labeling; obtain $H_t$.  
6. Form new dataset $D_{t+1}$, retrain to get $M_{t+1}$.  
7. Compute metrics $\mathcal{D}_{t+1},\mathcal{C}_{t+1},\mathcal{Q}_{t+1}$.  
8. Check convergence; if not, increment $t$ and repeat.  

3.7 Experimental Design  
Datasets and Domains  
- Biomedical Imaging: histopathology microscopy (seed: 1K labeled images; pool: 50K unlabeled).  
- Climate Science: satellite multispectral time series (seed: 500 labeled scenes; pool: 20K scenes).  

Baselines  
- Static Model-Assisted Construction (no diversity loop).  
- Fairness-aware augmentation (Chameleon).  
- Random sampling + human labeling.  

Implementation Details  
- Models: Vision Transformer (ViT) or U-Net for segmentation; ensembles of size $L=5$.  
- Synthetic generation via conditional diffusion (1000 steps).  
- Clustering: K-means with $K=50$.  
- Label budget per iteration $B=200$.  

Evaluation Metrics  
- Downstream task performance: classification accuracy, F1-score, IoU for segmentation.  
- Robustness under distribution shifts: test on held-out sites/time periods.  
- Diversity & coverage metrics as defined above.  
- Annotation cost: number of labels vs. performance.  
- Fairness measures: demographic parity or group-wise accuracy in biomedical subpopulations.  

Reproducibility  
We will open-source code, synthetic data scripts, and trained models.  Experiments will be run on standard GPU clusters; seeds and hyperparameters will be published.  

4. Expected Outcomes & Impact  
Expected Outcomes  
- Demonstration that our diversity-aware loop yields datasets with 20–30% higher entropy $\mathcal{D}$ and 25–40% lower JSD against uniform cluster distribution, compared to static baselines.  
- 30–50% reduction in human labeling cost to reach a target accuracy threshold on both biomedical and climate tasks.  
- Improved robustness: models trained on our datasets will exhibit 5–10% fewer errors under novel domain shifts.  
- Quantitative analysis of bias amplification, showing that our framework stabilizes data-model ecosystems (smaller variance in $\mathcal{Q}_t$).  

Broader Impact  
- A generalizable, modular pipeline for data-centric ML that can be adapted to new domains (e.g.\ robotics, remote-sensing).  
- Enhanced ethical data practices through explicit diversity metrics and bias monitoring, addressing concerns raised by recent works on fairness feedback loops.  
- Contributions to HCI in data curation by integrating human validation in active loops, reducing cognitive load and focusing on critical samples.  
- Benchmarks and open datasets released for the community, spurring further research on data-centric foundation models.  

Conclusion  
This proposal advances the frontier of data-centric machine learning by embedding diversity and quality monitoring at the heart of dataset construction.  By uniting model-assisted generation, clustering-driven diversity targeting, active human validation, and rigorous metrics, we will produce high-value datasets that empower foundation models to operate fairly and robustly in emerging and high-stakes domains.