# **Research Proposal: Adaptive Bayesian Design Space Exploration for Protein Engineering Using Generative Models and Experimental Feedback**

## 1. Introduction

### 1.1 Background
Proteins are fundamental biological macromolecules, performing a vast array of functions essential for life. Engineering proteins with novel or enhanced functionalities holds transformative potential for medicine (e.g., therapeutic antibodies, enzyme replacement therapies), industry (e.g., biocatalysis, biosensors), and environmental remediation (e.g., plastic degradation). However, the sheer scale of the protein sequence space presents a formidable challenge. Even for a small protein of 100 amino acids, the number of possible sequences ($20^{100}$) vastly exceeds our capacity for experimental exploration. Traditional protein engineering methods, often relying on random mutagenesis or rational design based on limited structural knowledge, struggle to navigate this combinatorial complexity efficiently.

In recent years, generative machine learning (ML) models, such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Protein Language Models (PLMs), have emerged as powerful tools for *in silico* protein design (Winnifrith et al., 2023). These models can learn patterns from known protein sequences and generate novel sequences predicted to possess desired properties. Despite significant progress, a critical gap persists between computational predictions and experimental validation, a central theme highlighted by the Generative and Experimental perspectives in bioMolecular design (GEM) workshop. Many generative models achieve high performance on static computational benchmarks but often generate sequences with high false-positive rates when tested experimentally (Calvanese et al., 2025), leading to wasted resources and hindering translation to real-world applications. Furthermore, most approaches operate in an open-loop fashion: models are trained once, generate candidates, and experimental validation follows without systematically informing subsequent design cycles.

To bridge this gap, adaptive experimental design strategies, often employing principles from active learning and Bayesian optimization (BO), are gaining traction. These methods aim to create a closed loop where experimental results iteratively refine computational models and guide the selection of subsequent experiments towards the most promising regions of the design space (Doe & Smith, 2024; Lee & Kim, 2023). Recent studies have demonstrated the potential of integrating experimental feedback with generative models (Johnson & Williams, 2024; Green et al., 2024) and developing closed-loop systems (Chen et al., 2025; Martinez & Johnson, 2025). These approaches promise to significantly reduce the experimental burden while accelerating the discovery of functional biomolecules.

### 1.2 Research Objectives
This research aims to develop and validate a novel adaptive experimental design framework, termed Adaptive Bayesian Generative Exploration (ABGE), specifically tailored for protein engineering. ABGE integrates a generative model (VAE) with Bayesian optimization (BO) in a closed-loop system, leveraging experimental feedback to dynamically guide the exploration of protein sequence space.

The specific objectives are:
1.  **Develop the ABGE framework:** Integrate a VAE for candidate generation with a GP-based BO module for intelligent sequence selection based on predicted fitness and uncertainty. Establish a closed-loop workflow enabling iterative model refinement using experimental feedback.
2.  **Implement uncertainty-aware and diversity-driven selection:** Incorporate uncertainty quantification from the BO surrogate model and explicit diversity metrics to ensure a balance between exploiting high-potential regions and exploring novel areas of the sequence space during batch selection.
3.  **Incorporate adaptive model updates:** Implement mechanisms to update both the BO surrogate model and the underlying VAE generative model based on incoming experimental data, allowing the system to learn from experience and improve its generative and predictive capabilities over time.
4.  ***In silico* validation:** Rigorously evaluate the performance of the ABGE framework using established computational oracles (simulated fitness landscapes) for protein properties (e.g., stability, fluorescence, binding affinity). Compare its efficiency and effectiveness against baseline methods.
5.  **Demonstrate potential for wet-lab application:** Outline a clear path and methodology for applying the ABGE framework to a specific, relevant protein engineering task, detailing the required experimental assays and feedback integration.

### 1.3 Significance
This research directly addresses the critical challenge of efficiently navigating the vast protein design space and bridging the gap between computational prediction and experimental validation, aligning perfectly with the goals of the GEM workshop. By developing the ABGE framework, we anticipate the following significant contributions:

1.  **Accelerated Discovery:** The adaptive, closed-loop nature of ABGE is expected to significantly speed up the discovery of novel, high-functioning proteins compared to traditional methods or open-loop generative approaches.
2.  **Optimized Resource Allocation:** By intelligently selecting sequences for experimental validation based on uncertainty and potential, ABGE aims to drastically reduce the number of costly and time-consuming wet-lab experiments required, potentially achieving significant cost savings (targeting up to 80% reduction, as hypothesized in the initial idea).
3.  **Improved Generative Models:** Integrating direct experimental feedback for model refinement (both BO and VAE) promises to create more accurate and reliable generative models for protein design, reducing the prevalent issue of high false-positive rates (Calvanese et al., 2025).
4.  **Bridging Computation and Experiment:** This work provides a concrete methodology for synergistic integration of state-of-the-art ML with experimental biology, fostering closer collaboration between computationalists and experimentalists.
5.  **Generalizability:** While focused on protein engineering, the principles of the ABGE framework could potentially be adapted for designing other biomolecules like RNA or small molecules, broadening its impact.

Ultimately, this research contributes to advancing ML's role in biology beyond benchmark performance, enabling the efficient engineering of bespoke biomolecules to tackle pressing challenges in health, industry, and the environment.

## 2. Methodology

### 2.1 Overall Framework
The proposed Adaptive Bayesian Generative Exploration (ABGE) framework operates as an iterative closed-loop system (Figure 1 - conceptual description). Each iteration consists of the following key steps:

1.  **Candidate Generation:** A Variational Autoencoder (VAE), trained on known protein sequences and potentially conditioned on desired properties, generates a diverse pool of novel candidate sequences.
2.  **Fitness Prediction & Uncertainty Estimation:** A Bayesian Optimization (BO) module, typically using a Gaussian Process (GP) as a surrogate model, predicts the fitness (e.g., stability, binding affinity, catalytic activity) and associated uncertainty for each candidate sequence based on data from previous experimental rounds.
3.  **Intelligent Candidate Selection:** An acquisition function (e.g., Upper Confidence Bound - UCB) leverages the GP's predictions and uncertainties to score candidates. A batch of sequences is selected to maximize the acquisition function while promoting diversity, ensuring a balance between exploiting promising regions (high predicted fitness) and exploring uncertain areas (high uncertainty).
4.  **Experimental Validation:** The selected batch of protein sequences is synthesized and experimentally tested (either via wet-lab assays or high-fidelity computational oracles in the validation phase) to measure their true fitness values.
5.  **Model Update:** The experimental results (sequence-fitness pairs) are used to update the BO surrogate model (GP). Critically, this feedback is also used to refine the VAE generative model, improving its ability to generate promising candidates in subsequent iterations.
6.  **Iteration:** The loop repeats from Step 1 with the updated models until a predefined stopping criterion is met (e.g., budget exhaustion, convergence to desired fitness level, number of iterations reached).

**(Conceptual Figure 1 Description: A diagram showing a loop. Arrow from "VAE Generator" points to "Candidate Pool". Arrow from "Candidate Pool" points to "BO Selector (GP + Acquisition Fn + Diversity)". Arrow from "BO Selector" points to "Selected Batch for Experiment". Arrow from "Selected Batch" points to "Experimental Assay / Oracle". Arrow from "Experiment" feeds "Results (Sequence, Fitness)" back to "BO Selector" (Update GP) and potentially back to "VAE Generator" (Update VAE). The loop then repeats.)**

### 2.2 Data Collection and Preparation
*   **Initial Dataset:** The framework requires an initial dataset of protein sequences with associated fitness labels relevant to the target engineering task (e.g., sequences and their measured thermostability, fluorescence brightness, or binding affinity scores). Sources can include public databases (e.g., ProTherm, FireProtDB, Skempi), literature data, or results from previous large-scale screening campaigns. Sequences will be preprocessed (e.g., filtering, alignment if necessary) and encoded, typically using one-hot encoding or embeddings from pre-trained protein language models (e.g., ESM, ProtBERT) for input into the ML models.
*   **Experimental Feedback Data:** During the iterative process, the framework ingests data points of the form $(x_i, y_i)$, where $x_i$ is a tested protein sequence (or its representation) and $y_i$ is its experimentally determined fitness score. The nature of $y_i$ depends on the specific assay (e.g., continuous score, binary functional/non-functional).

### 2.3 Algorithmic Details

#### 2.3.1 Variational Autoencoder (VAE) for Candidate Generation
*   **Architecture:** We will employ a VAE architecture suitable for sequence data. This typically involves an encoder network (e.g., LSTM or CNN) that maps an input sequence $x$ to a latent distribution $q_\phi(z|x)$, usually parameterized as a multivariate Gaussian $\mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$, and a decoder network (e.g., LSTM or CNN) that reconstructs the sequence from a latent sample $z \sim q_\phi(z|x)$, defining $p_\theta(x|z)$.
*   **Training:** The VAE is initially trained by maximizing the Evidence Lower Bound (ELBO) on the initial dataset:
    $$ \mathcal{L}_{VAE}(\phi, \theta; x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) || p(z)) $$
    where $p(z)$ is the prior distribution over the latent space (typically $\mathcal{N}(0, I)$), $D_{KL}$ is the Kullback-Leibler divergence, and $\beta$ is a weighting factor (potentially annealed during training).
*   **Generation:** New candidates are generated by sampling $z \sim p(z)$ and passing it through the decoder: $x_{new} = \text{decoder}_\theta(z)$. A large pool ($N_{cand} \gg N_{batch}$) of candidates is generated in each iteration.

#### 2.3.2 Bayesian Optimization (BO) Module
*   **Surrogate Model (GP):** A Gaussian Process (GP) will be used to model the mapping from a sequence representation (either the VAE latent vector $z$ or a fixed sequence descriptor) to the observed fitness $y$. Given a set of $n$ observations $D_n = \{(x_i, y_i)\}_{i=1}^n$, the GP provides a posterior distribution over the fitness function $f$: $p(f|D_n) = \mathcal{GP}(\mu_n(x), k_n(x, x'))$. The posterior mean $\mu_n(x)$ represents the predicted fitness, and the posterior variance $\sigma_n^2(x) = k_n(x, x)$ quantifies the uncertainty. We will use standard kernels suitable for the chosen sequence representation, such as the RBF kernel for latent vectors or potentially string kernels for sequences directly, with hyperparameters optimized by maximizing the marginal likelihood.
*   **Acquisition Function:** To guide the selection of the next batch experiments, we will use the Upper Confidence Bound (UCB) acquisition function:
    $$ \alpha_{UCB}(x) = \mu_n(x) + \kappa \sigma_n(x) $$
    where $\kappa$ is a hyperparameter balancing exploitation (high $\mu_n$) and exploration (high $\sigma_n$). Sequences maximizing $\alpha_{UCB}$ are prioritized.
*   **Batch Selection with Diversity:** Since experiments are often performed in batches, we need to select $N_{batch}$ sequences simultaneously. Simply picking the top $N_{batch}$ points according to $\alpha_{UCB}$ can lead to sampling redundant points in the same region. To ensure diversity, we will employ a batch selection strategy. After generating the candidate pool $X_{cand}$ and computing their UCB scores, we will iteratively select sequences:
    1. Select $x_1^* = \arg\max_{x \in X_{cand}} \alpha_{UCB}(x)$.
    2. For $j=2$ to $N_{batch}$: Select $x_j^* = \arg\max_{x \in X_{cand} \setminus \{x_1^*, ..., x_{j-1}^*\}} [\alpha_{UCB}(x) - \lambda D(x, \{x_1^*, ..., x_{j-1}^*\})]$, where $D(x, S)$ is a diversity term penalizing similarity to already selected points in the batch (e.g., based on distance in latent space or sequence space) and $\lambda$ controls the diversity pressure. Alternatively, methods like Batch BALD or diverse top-k sampling based on clustering in latent space can be explored.

#### 2.3.3 Experimental Feedback Integration and Model Update
*   **GP Update:** After obtaining experimental results $(X_{batch}, Y_{batch})$ for the selected batch, the dataset $D_n$ is augmented to $D_{n+N_{batch}}$, and the GP hyperparameters are re-optimized, yielding updated $\mu_{n+N_{batch}}(x)$ and $\sigma_{n+N_{batch}}(x)$.
*   **VAE Update (Adaptive Refinement):** This is a key adaptive component. The new high-performing sequences discovered through experimentation contain valuable information. We will investigate methods to incorporate this feedback into the VAE:
    *   **Weighted Fine-tuning:** Retrain/fine-tune the VAE (parameters $\phi, \theta$) using the augmented dataset, potentially up-weighting the newly discovered high-fitness sequences. This guides the VAE to generate more sequences similar to empirically validated successful designs (inspired by Calvanese et al., 2025; Johnson & Williams, 2024).
    *   **Conditional Generation:** Modify the VAE architecture or training objective to allow conditioning on desired fitness levels, potentially by incorporating the GP predictions or observed fitness values into the VAE's objective function or architecture.
    *   **Latent Space Guidance:** Use the GP model to identify promising regions in the latent space $Z$ and bias the VAE's sampling process $z \sim p(z)$ towards these regions during candidate generation.

The most effective VAE update strategy will be determined during the initial *in silico* validation phase.

### 2.4 Experimental Design for Validation

#### 2.4.1 *In Silico* Validation
*   **Simulated Fitness Landscapes (Oracles):** We will use well-established computational oracles that provide a fitness score for any given protein sequence, simulating the experimental outcome without wet-lab costs. Potential oracles include:
    *   Rosetta folding energy landscape (predicting protein stability).
    *   Pre-trained predictors for specific functions (e.g., GFP fluorescence predictors based on deep learning models trained on large experimental datasets like Sarkisyan et al., 2016).
    *   Simplified lattice protein models with known energy functions.
*   **Baseline Methods for Comparison:** The performance of ABGE will be compared against:
    1.  **Random Sampling:** Sequences randomly selected from the space accessible by the VAE or mutations.
    2.  **Generative Model + Top-K:** Generate sequences using the initial VAE and select the top-K predicted sequences based on a static property predictor (trained once on the initial data) without active learning.
    3.  **Standard BO:** Bayesian optimization using the GP and UCB but without the adaptive VAE (VAE is trained once and used only for initial candidate pool generation).
    4.  **VAE + Feedback only (No BO):** Using experimental feedback to update the VAE but selecting candidates randomly or based solely on VAE likelihood from the updated generator.
*   **Evaluation Metrics:**
    *   **Efficiency:**
        *   *Convergence Speed:* Number of experimental rounds (batches tested) or total sequences tested to reach a predefined target fitness threshold.
        *   *Cumulative Best Fitness:* Plot of the maximum fitness found versus the number of sequences tested. A steeper curve indicates higher efficiency.
        *   *Hit Rate:* Percentage of tested sequences exceeding a specific fitness threshold within a given budget.
    *   **Effectiveness:**
        *   *Maximum Fitness Achieved:* The highest fitness value discovered within a fixed experimental budget (e.g., 500 tested sequences).
        *   *Diversity of Top Sequences:* Measure the sequence diversity (e.g., average pairwise distance) among the top-performing sequences discovered.
    *   **Resource Reduction:** Quantify the reduction in the number of required experiments compared to baselines to achieve the same performance level (e.g., reach 95% of the maximum achievable fitness on the oracle landscape).

#### 2.4.2 Potential Wet-Lab Case Study Design
Pending successful *in silico* validation, we propose applying ABGE to a relevant protein engineering challenge, such as:
*   **Target Task:** Optimizing the thermostability of an enzyme (e.g., a lipase or protease) while maintaining its catalytic activity.
*   **Initial Data:** Sequences of homologous enzymes with known stability (e.g., melting temperature $T_m$) and activity data.
*   **Experimental Assay:**
    *   Gene synthesis and recombinant protein expression for selected candidates.
    *   Differential Scanning Fluorimetry (DSF) or Circular Dichroism (CD) melt curves to measure $T_m$ (stability fitness).
    *   Enzymatic activity assays using a standard substrate to measure specific activity (maintain as constraint or secondary objective).
*   **Feedback:** The measured $T_m$ values constitute the primary fitness feedback $y_i$ for the ABGE loop. Activity data can be used as a filter or incorporated into a multi-objective optimization framework.
*   **Outcome:** Demonstrate the discovery of novel enzyme variants with significantly enhanced thermostability compared to the wild-type or variants found through baseline methods, using a comparable experimental budget.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes
We anticipate the following outcomes from this research:
1.  **A Robust ABGE Framework:** A fully implemented and documented software framework integrating VAEs, BO, and adaptive model updating for protein engineering, potentially released as open-source to benefit the research community.
2.  ***In Silico Proof-of-Concept:** Quantitative results from simulations demonstrating that ABGE significantly outperforms baseline methods in terms of efficiency (requiring fewer experimental evaluations to reach high-fitness regions, potentially >5x improvement) and effectiveness (discovering proteins with higher fitness scores within a fixed budget). We aim to validate the initial hypothesis of potentially achieving substantial (e.g., approaching 80%) reduction in experimental effort compared to non-adaptive or random approaches for certain landscapes.
3.  **Novel High-Performing Sequences:** Identification of novel protein sequences with optimized properties within the simulated landscapes, showcasing the framework's discovery potential.
4.  **Validated Adaptive Strategies:** Insights into the most effective strategies for updating the generative model (VAE) based on sparse experimental feedback within a closed-loop system.
5.  **Pathway to Experimental Validation:** A detailed protocol and design for applying ABGE to a specific wet-lab protein engineering task, paving the way for future experimental validation.
6.  **Publications and Dissemination:** Peer-reviewed publications in leading ML conferences (e.g., NeurIPS, ICML, ICLR) and/or bioinformatics/computational biology journals. Presentation at relevant workshops like GEM, potentially leading to consideration for the associated Nature Biotechnology fast-track process.

### 3.2 Impact
This research project holds the potential for significant impact:
*   **Scientific Impact:** It advances the state-of-the-art in machine learning for scientific discovery, particularly in applying adaptive and generative methods to complex biological design problems. It provides a principled approach to address key challenges in the field, such as high false-positive rates, data scarcity in later optimization stages, and efficient exploration-exploitation trade-offs (addressing challenges identified in Calvanese et al., 2025; Lee & Kim, 2023; Doe & Smith, 2024). By systematically integrating experimental feedback to refine generative models, it directly tackles a core limitation of current *in silico* design approaches.
*   **Practical Impact:** The ABGE framework, if successful, offers a powerful tool to accelerate the design and optimization of proteins for diverse applications. This can drastically reduce the time and cost associated with developing new enzymes for green chemistry, engineering antibodies for diagnostics and therapeutics, and creating novel biomaterials. The potential for significant resource optimization makes advanced protein engineering more accessible.
*   **Community Impact:** This work directly contributes to the goals of the GEM workshop by developing and evaluating a methodology that explicitly links generative ML with adaptive experimental design crucial for real-world biological applications. By sharing the framework and findings, we aim to foster further research and collaboration at the intersection of ML and experimental biology, promoting the translation of computational methods into tangible biological outcomes.

In conclusion, the proposed research on Adaptive Bayesian Generative Exploration (ABGE) offers a promising path towards more efficient and effective data-driven protein engineering, bridging the computational-experimental divide and unlocking new possibilities in biomolecular design.

**(Approximate Word Count: ~2100 words)**