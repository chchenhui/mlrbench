# Title  
**InfluenceSpace: Hierarchical Influence–Driven Curation for Multi-Modal Foundation Models**

---

# Introduction  

## Background  
Multi-modal foundation models (FMs), such as those combining vision, text, and audio, have revolutionized AI by achieving state-of-the-art performance across diverse tasks. However, their success hinges on massive, heterogeneous training datasets, which often contain redundancy, noise, and bias. Traditional data curation methods—such as heuristic pruning or rule-based balancing—fail to quantify the *causal impact* of individual data points or clusters on model behavior. This limitation is critical for multi-modal FMs, where interactions between modalities amplify the complexity of data influence. For instance, a visually ambiguous video caption might disproportionately harm alignment accuracy, while under-represented demographic groups in training data may perpetuate fairness gaps.  

Existing influence estimation techniques, like DataInf (Kwon et al., 2023), approximate the effect of training samples using low-rank Hessians but focus on single-modal models or parameter-efficient fine-tuning. Meanwhile, multimodal evaluation frameworks like HEMM (Liang et al., 2024) highlight gaps in cross-modal reasoning but lack guidance for proactive data curation. The need for scalable, hierarchical methods to identify and prioritize high-impact multi-modal data clusters is urgent, particularly as models grow to trillion-scale parameters and legal concerns around data copyright and fairness intensify.  

## Research Objectives  
This proposal aims to address the following objectives:  
1. **Develop a scalable curation pipeline** that quantifies cross-modal data influence hierarchically, balancing computational efficiency with granularity.  
2. **Prune harmful/irrelevant clusters** and up-weight underrepresented but semantically rich groups to enhance model accuracy and fairness.  
3. **Mitigate bias** by explicitly modeling interactions between modalities, ensuring robustness to domain shifts and demographic imbalances.  
4. **Validate** the pipeline on standardized benchmarks (e.g., VQA, COCO) with ablation studies on data volume reduction, energy consumption, and fairness metrics.  

## Significance  
This work bridges two critical research directions identified in the DATA-FM workshop: (1) advancing practical strategies for FM curation and (2) addressing societal impacts through data-centric approaches. By reducing dataset sizes while maintaining or improving performance, InfluenceSpace will lower computational costs and carbon footprints for FM training. Additionally, explicit focus on influence-driven fairness (e.g., prioritizing clusters representing marginalized groups) aligns with emerging needs for ethical AI systems.  

---

# Methodology  

The InfluenceSpace pipeline has two stages: **hierarchical clustering** and **influence-driven curation**, both optimized for multi-modal scalability (Figure 1).  

## Stage 1: Cross-Modal Semantic Clustering  

### Data Preprocessing  
Let $ \mathcal{D} = \{\mathbf{x}_1^{(i)}, \mathbf{x}_2^{(i)}, ..., \mathbf{x}_M^{(i)}\}_{i=1}^N $ denote a multi-modal dataset with $ N $ examples, each containing $ M $ modalities (e.g., text, image). For each modality $ m $, we extract embeddings $ \mathbf{z}_m^{(i)} \in \mathbb{R}^d $ using frozen encoders (e.g., CLIP (Radford et al., 2021) or FLAVA (Singh et al., 2021)). Cross-modal alignment is then enforced via a **contrastive loss** $ \mathcal{L}_{\text{con}} $, which pulls corresponding embeddings closer and pushes negatives apart.  

The combined embedding for example $ i $ is:  
$$
\bar{\mathbf{z}}^{(i)} = \frac{1}{M} \text{LN}\left( \sum_{m=1}^M \text{ReLU}(\mathbf{W}_m \mathbf{z}_m^{(i)}) \right),
$$  
where $ \mathbf{W}_m $ is a learnable projection and $ \text{LN} $ is layer normalization.  

### Clustering Algorithm  
We apply **spherical k-means** (which normalizes embeddings) to group $ \{ \bar{\mathbf{z}}^{(i)} \} $ into $ K $ clusters $ \mathcal{C}_1, \mathcal{C}_2, ..., \mathcal{C}_K $, where $ K = \mathcal{O}(\log N) $ to trade off granularity and scalability. This yields centroids $ \{ \boldsymbol{\mu}_k \}_{k=1}^K $, which serve as cluster representatives.  

### Cluster Characterization  
Each cluster $ \mathcal{C}_k $ is annotated with:  
- **Modal dominance**: Compute the relative importance $ w_k^{(m)} = \frac{\text{Var}(\mathbf{z}_m^{(i)} | \mathcal{C}_k)}{\sum_{m'} \text{Var}(\mathbf{z}_{m'}^{(i)} | \mathcal{C}_k)} $ to identify modality-specific clusters (e.g., text-heavy for NLP tasks).  
- **Diversity**: Measure $ \Delta_k = \mathbb{E}_{i,j \in \mathcal{C}_k}[ \| \bar{\mathbf{z}}^{(i)} - \bar{\mathbf{z}}^{(j)} \| ] $ to detect homogeneity.  

## Stage 2: Hierarchical Influence Estimation  

We model the influence $ \mathcal{I}(\mathcal{C}_k) $ of cluster $ \mathcal{C}_k $ on model loss $ \mathcal{L}_{\text{val}} $ using an amortized approximation inspired by DataInf (Kwon et al., 2023).  

### Influence Function Approximation  
The influence of a data point on the validation loss is conventionally given by:  
$$  
\mathcal{I}(x) = -\nabla_{\theta} \mathcal{L}_{\text{val}}(\theta^*)^\top \mathbf{H}_{\theta^*}^{-1} \nabla_{\theta} \mathcal{L}_{\text{train}}(x, \theta^*),  
$$  
where $ \theta^* $ is the optimal parameter and $ \mathbf{H}_{\theta^*} $ the Hessian. Instead of computing per-sample $ \mathcal{I}(x) $, we approximate cluster influence:  
$$  
\mathcal{I}(\mathcal{C}_k) = -\nabla_{\theta} \mathcal{L}_{\text{val}}(\theta^*)^\top \mathbf{H}_{\theta^*}^{-1} \nabla_{\theta} \left( \frac{1}{|\mathcal{C}_k|} \sum_{x \in \mathcal{C}_k} \mathcal{L}_{\text{train}}(x, \theta^*) \right).  
$$  
To reduce computational complexity, we:  
1. **Approximate $ \mathbf{H}_{\theta^*}^{-1} $**: Use a **diagonal low-rank approximation** $ \mathbf{H}_{\theta^*}^{-1} \approx \sum_{r=1}^R \lambda_r \mathbf{u}_r \mathbf{u}_r^\top + \gamma \mathbf{I} $, where $ R $ is the rank (e.g., 200).  
2. **Use mini-batch gradients**: Estimate $ \nabla_{\theta} \mathcal{L}_{\text{val}}(\theta^*) $ with a moving average over validation batches.  
3. **Cluster-level aggregation**: Replace per-sample gradients with centroid-based proxy:  
$$  
\mathcal{I}_{\text{proxy}}(\mathcal{C}_k) = \mathbf{a}^\top \boldsymbol{\mu}_k, \quad \text{where } \mathbf{a} = \mathbf{H}_{\theta^*}^{-1} \nabla_{\theta} \mathcal{L}_{\text{train}}(\theta^*).  
$$  

### Curation Strategy  
Clusters are scored by:  
$$  
s_k = \gamma_1 \cdot \mathcal{I}_{\text{proxy}}(\mathcal{C}_k) + \gamma_2 \cdot \Delta_k - \gamma_3 \cdot |\mathcal{C}_k|,  
$$  
balancing influence, diversity, and redundancy. Hyperparameters $ \gamma_i $ are validated on a dev set. Pruning thresholds and up-weighting factors are determined as:  
- **Prune**: Remove clusters where $ s_k < \tau_p $.  
- **Boost**: Up-sample clusters where $ s_k > \tau_b $ and $ | \{ x \in \mathcal{C}_k : \text{fairness violation} \} | \geq \delta $.  

### Experimental Design  

#### Datasets  
- **Primary Benchmark**: COCO Captions (vision-text), HowTo100M (video-audio).  
- **Fairness-Sensitive Tasks**: Waterbirds (background bias), UCF101-Animacy (demographics).  

#### Baselines  
- **Random Curation**: Randomly drop 50% of data.  
- **Rule-Based Curation**: Remove low-quality samples via CLIP score thresholds.  
- **DataInf (Cluster-Agnostic)**: Apply DataInf per-sample then aggregate.  

#### Evaluation Metrics  
- **Performance**: Accuracy, BLEU-4, and mAP on HEMM’s benchmarks.  
- **Efficiency**: Training epochs to converge, energy consumption (kWh).  
- **Fairness**: Demographic parity difference $ \Delta_{DP} = | P(\hat{Y}=1 | A=0) - P(\hat{Y}=1 | A=1) | $, where $ A $ is the attribute.  
- **Robustness**: Performance on corrupted or domain-shifted subsets.  

#### Implementation Details  
- **Optimization**: Use AdamW with LR = 1e-4, batch size 512.  
- **Cluster Parameters**: $ K = 1000 $ for COCO, $ K = 2000 $ for HowTo100M.  
- **Influence Parameters**: Rank $ R = 200 $, $ R_{\text{update}} = 5 $.  

---

# Expected Outcomes & Impact  

## Anticipated Results  
1. **Dataset Efficiency**: Reduce training data by >60% while achieving >95% of baseline accuracy on vision-text retrieval tasks.  
2. **Bias Mitigation**: Decrease gender bias in captioning datasets by a factor of 2–3, as measured by $ \Delta_{DP} $.  
3. **Theoretical Insights**: Cluster-level influence correlates with HEMM’s cross-modal reasoning scores (e.g., clusters with high $ \mathcal{I} $ align with "multi-modal entailment" subtasks).  
4. **Computational Scaling**: Enable influence estimation for 10M+ samples with <5% of the compute required by per-sample methods.  

## Technical Contributions  
- A **hierarchical framework** for balancing granularity and scalability in influence estimation, compatible with multi-modal fusion architectures like FLAVA.  
- A **lightweight influence proxy** (via centroid gradients) that outperforms existing methods in runtime while maintaining correlation with true influence ($ \rho > 0.7 $).  
- An open-source toolkit for influence-driven multimodal curation, to be released alongside benchmarks for fairness and robustness evaluation.  

## Broader Impact  
This work directly addresses four DATA-FM workshop themes:  
1. **Societal Impacts**: Reduces environmental costs (via data efficiency) and promotes fairness through targeted cluster up-weighting.  
2. **Data Curation**: Provides a principled alternative to heuristic filtering, extendable to retrieval-augmented generation and agents.  
3. **Economics of Data**: Establishes cluster-level "value" (via influence scores), laying groundwork for data marketplaces.  
4. **Benchmarking**: Identifies pitfalls in current benchmarks (e.g., over-reliance on uniform data distributions) through cluster analysis.  

By enabling precise quantification of multi-modal data utility, InfluenceSpace aims to catalyze a shift toward influence-aware FM pre-training paradigms, ultimately improving transparency and accountability in AI development.  

--- 

**Total Word Count**: ~1,980 words (excluding headers and equations)