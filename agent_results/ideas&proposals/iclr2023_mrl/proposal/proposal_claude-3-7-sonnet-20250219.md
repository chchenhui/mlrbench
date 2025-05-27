# Geometric Optimal Transport for Robust Cross-Modal Manifold Alignment in Multimodal Representation Learning

## 1. Introduction

Multimodal representation learning has emerged as a critical approach in artificial intelligence, enabling machines to process and understand information from different perceptual modalities simultaneously. The integration of information from diverse modalities, such as vision, language, and audio, offers significant advantages over unimodal approaches, including improved robustness, enhanced generalization, and more comprehensive understanding of complex phenomena. This integration is particularly important as real-world information naturally spans multiple modalities, making multimodal learning essential for applications ranging from cross-modal retrieval and generation to multimodal sentiment analysis and medical diagnosis.

Despite impressive advances in multimodal learning, fundamental challenges persist in effectively aligning and integrating information across modalities. Different modalities exhibit distinct statistical properties, semantic structures, and geometric characteristics in their native representation spaces. When these heterogeneous representations are naively combined or projected into a shared space, misalignments can occur that undermine the effectiveness of the resulting multimodal representation. Traditional approaches have predominantly focused on instance-level alignment (e.g., through contrastive learning) or simple concatenation/fusion techniques, without explicitly addressing the underlying geometric incompatibilities between modality manifolds.

Recent literature has begun to explore this problem from various angles. Cicchetti et al. (2024) introduced the Gramian Representation Alignment Measure (GRAM) to align multiple modalities in higher-dimensional space by minimizing the Gramian volume. Cai et al. (2025) examined how misalignment shapes multimodal representation learning, particularly in image-text pairs. Jiang et al. (2023) challenged the notion that exact modality alignment is always optimal, proposing alternative approaches including deep feature separation and geometric consistency. Tjandrasuwita et al. (2025) investigated the implicit emergence of alignment in multimodal learning, finding that its relationship with task performance depends on data characteristics like modality similarity and information redundancy.

While these studies have advanced our understanding of multimodal alignment, they have not fully addressed the geometric foundations of cross-modal transfer or developed comprehensive frameworks that can systematically align the manifold structures of different modalities. Furthermore, the scalability of alignment techniques to more than two modalities remains insufficiently explored, and the impact of geometric alignment on downstream tasks requires further investigation.

This research proposal addresses these gaps by introducing a novel framework called Geometric Optimal Transport for Manifold Alignment (GOTMA). GOTMA leverages principles from optimal transport theory and Riemannian geometry to explicitly align the geometric structures of different modality manifolds within a joint embedding space. Rather than relying solely on instance-level alignment, our approach preserves important topological and geometric properties across modalities, facilitating more effective cross-modal transfer and improved performance on downstream tasks.

The objectives of this research are threefold:
1. Develop a comprehensive theoretical framework for understanding and quantifying geometric misalignment between modality-specific representation manifolds.
2. Design novel training objectives based on optimal transport and Riemannian geometry that explicitly promote structural similarity between modality manifolds in the shared embedding space.
3. Evaluate the impact of geometric alignment on representation quality and performance across a diverse set of multimodal tasks, with particular focus on robustness to modality noise, missing modalities, and cross-modal transfer.

The significance of this research lies in its potential to advance fundamental understanding of multimodal representation spaces and to develop more effective techniques for learning joint representations that preserve the rich geometric structure of each modality. By addressing the geometric foundations of multimodal learning, this work will contribute to the development of more robust, interpretable, and effective multimodal systems capable of handling the complexity of real-world multimodal data.

## 2. Methodology

### 2.1 Theoretical Framework for Geometric Alignment

We begin by formalizing the problem of geometric alignment in multimodal representation learning. Consider a multimodal dataset with $M$ modalities, where each sample $i$ has observations from different modalities $\{x_i^1, x_i^2, ..., x_i^M\}$. For each modality $m$, we have an encoder $f_m$ that maps the input to a representation $\mathbf{z}_i^m = f_m(x_i^m) \in \mathbb{R}^d$. These representations can be viewed as points on modality-specific manifolds $\mathcal{M}_m$ embedded in a common latent space.

Our key insight is that effective multimodal learning requires not just alignment of individual instances across modalities, but alignment of the geometric structures of the manifolds themselves. We define geometric misalignment using the following metrics:

1. **Manifold Discrepancy**: We quantify the discrepancy between manifolds using the Gromov-Wasserstein distance, which measures the difference in the intrinsic geometries of the manifolds:

$$D_{GW}(\mathcal{M}_1, \mathcal{M}_2) = \min_{\pi \in \Pi(\mu_1, \mu_2)} \sum_{i,j,k,l} |d_1(z_i^1, z_j^1) - d_2(z_k^2, z_l^2)|^2 \pi_{i,k}\pi_{j,l}$$

where $\mu_m$ is the empirical distribution of points on manifold $\mathcal{M}_m$, $\Pi(\mu_1, \mu_2)$ is the set of all transport plans between these distributions, and $d_m$ is a distance metric on manifold $\mathcal{M}_m$.

2. **Local Neighborhood Preservation**: For each point $\mathbf{z}_i^m$, we define its $k$-nearest neighbors in modality $m$ as $\mathcal{N}_k(\mathbf{z}_i^m)$. We measure neighborhood preservation across modalities as:

$$L_{NP}(f_1, f_2) = \frac{1}{N} \sum_{i=1}^N \frac{|\mathcal{N}_k(f_1(x_i^1)) \cap \mathcal{N}_k(f_2(x_i^2))|}{k}$$

3. **Curvature Alignment**: We compute the sectional curvatures of the manifolds at corresponding points and measure their alignment:

$$L_{CA}(f_1, f_2) = \frac{1}{N} \sum_{i=1}^N ||K_1(\mathbf{z}_i^1) - K_2(\mathbf{z}_i^2)||^2$$

where $K_m(\mathbf{z})$ represents the sectional curvature tensor at point $\mathbf{z}$ on manifold $\mathcal{M}_m$.

### 2.2 Geometric Optimal Transport for Manifold Alignment (GOTMA)

Building on the theoretical framework, we propose GOTMA, which consists of the following components:

#### 2.2.1 Modality-Specific Encoders

For each modality $m$, we design an encoder network $f_m$ that maps inputs to a shared $d$-dimensional representation space. The architecture of each encoder is tailored to the specific modality:

- For images: ResNet or Vision Transformer
- For text: BERT or RoBERTa
- For audio: Conformer or AST (Audio Spectrogram Transformer)

#### 2.2.2 Optimal Transport Alignment Loss

We introduce an alignment loss based on optimal transport theory that explicitly promotes geometric similarity between modality manifolds:

$$\mathcal{L}_{OT} = \sum_{m_1 < m_2} D_{OT}(\mathcal{M}_{m_1}, \mathcal{M}_{m_2})$$

where $D_{OT}$ is a computationally efficient approximation of the Gromov-Wasserstein distance using entropic regularization:

$$D_{OT}(\mathcal{M}_{m_1}, \mathcal{M}_{m_2}) = \min_{\pi \in \Pi(\mu_{m_1}, \mu_{m_2})} \sum_{i,j} c(z_i^{m_1}, z_j^{m_2}) \pi_{i,j} - \epsilon H(\pi)$$

Here, $c(z_i^{m_1}, z_j^{m_2})$ is a cost function that measures the geometric discrepancy between points, and $H(\pi)$ is the entropy of the transport plan.

#### 2.2.3 Riemannian Consistency Loss

To ensure that the geometric properties of the manifolds are preserved consistently across modalities, we introduce a Riemannian consistency loss:

$$\mathcal{L}_{RC} = \sum_{m_1 < m_2} \frac{1}{N} \sum_{i=1}^N \sum_{j \in \mathcal{N}_k(z_i^{m_1})} |d_R(z_i^{m_1}, z_j^{m_1}) - d_R(z_i^{m_2}, z_j^{m_2})|^2$$

where $d_R$ is the Riemannian distance between points on the manifold, approximated using geodesic distances in the embedding space.

#### 2.2.4 Instance-Level Contrastive Loss

While our focus is on geometric alignment, we also incorporate instance-level alignment through a modified contrastive loss:

$$\mathcal{L}_{CL} = \sum_{m_1 < m_2} \frac{1}{N} \sum_{i=1}^N -\log \frac{\exp(z_i^{m_1} \cdot z_i^{m_2} / \tau)}{\sum_{j=1}^N \exp(z_i^{m_1} \cdot z_j^{m_2} / \tau)}$$

where $\tau$ is a temperature parameter.

#### 2.2.5 Total Training Objective

The final training objective combines these losses with appropriate weighting:

$$\mathcal{L}_{total} = \lambda_{OT} \mathcal{L}_{OT} + \lambda_{RC} \mathcal{L}_{RC} + \lambda_{CL} \mathcal{L}_{CL} + \lambda_{task} \mathcal{L}_{task}$$

where $\mathcal{L}_{task}$ is a task-specific loss (e.g., classification or regression loss) and $\lambda$ values are hyperparameters controlling the contribution of each loss component.

### 2.3 Experimental Design

We will evaluate our approach through a comprehensive set of experiments across multiple datasets and tasks:

#### 2.3.1 Datasets

1. **MS-COCO**: A large-scale dataset containing images with corresponding text descriptions.
2. **AudioSet**: A collection of audio clips with associated labels and textual descriptions.
3. **CMU-MOSEI**: Multimodal sentiment analysis dataset with video, audio, and text.
4. **MIMIC-III**: Multimodal medical dataset with clinical notes, physiological measurements, and imaging data.

#### 2.3.2 Tasks and Evaluation Metrics

1. **Cross-Modal Retrieval**:
   - Image-to-text and text-to-image retrieval on MS-COCO
   - Metrics: Recall@K (K=1,5,10), Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG)

2. **Multimodal Classification**:
   - Sentiment analysis on CMU-MOSEI
   - Audio event classification on AudioSet
   - Metrics: Accuracy, F1-score, Area Under ROC Curve (AUC)

3. **Cross-Modal Translation**:
   - Image-to-text generation and text-to-image generation
   - Metrics: BLEU, METEOR, CIDEr for text; FID, IS for images

4. **Robustness Evaluation**:
   - Performance under modality corruption (e.g., adding noise, masking portions)
   - Performance with missing modalities
   - Metrics: Relative performance degradation compared to baseline models

5. **Geometric Analysis**:
   - Visualization of manifold alignment using t-SNE and UMAP
   - Quantitative assessment of geometric properties (curvature, geodesic distances)
   - Topological data analysis using persistent homology

#### 2.3.3 Baselines

We will compare GOTMA against the following baselines:

1. Unimodal models for each modality
2. Simple multimodal fusion approaches (concatenation, attention)
3. CLIP (Contrastive Language-Image Pretraining)
4. GRAM (Gramian Multimodal Representation Learning)
5. ALBEF (Align before Fuse)
6. MAC (Modality-Agnostic and -Specific Representations)

#### 2.3.4 Ablation Studies

We will conduct ablation studies to assess the contribution of each component:

1. Impact of optimal transport loss vs. conventional contrastive loss
2. Effect of Riemannian consistency loss
3. Scaling behavior with number of modalities (2, 3, and 4 modalities)
4. Sensitivity to hyperparameters (learning rate, loss weights, etc.)
5. Impact of different manifold distance approximations

#### 2.3.5 Implementation Details

We will implement our approach using PyTorch and conduct experiments on multiple NVIDIA A100 GPUs. For optimal transport computations, we will leverage the POT (Python Optimal Transport) library with GPU acceleration. Training will be performed using AdamW optimizer with a cosine learning rate schedule. We will use mixed-precision training to improve efficiency.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

1. **Theoretical Advances**: We expect to develop a rigorous theoretical framework for understanding geometric alignment in multimodal representation learning. This will include formal definitions and metrics for quantifying misalignment, as well as analytical results on the relationship between geometric alignment and downstream task performance.

2. **Algorithmic Innovations**: The proposed GOTMA framework will provide a novel approach to multimodal representation learning that explicitly addresses geometric alignment. We anticipate that this approach will outperform existing methods, particularly in scenarios involving complex modality interactions or when robustness to missing or corrupted modalities is required.

3. **Empirical Insights**: Through our comprehensive experimental evaluation, we expect to gain insights into:
   - The relationship between geometric alignment and downstream task performance
   - The relative importance of different alignment objectives (optimal transport, Riemannian consistency, contrastive)
   - How alignment requirements vary across different types of tasks and modality combinations
   - The geometric properties of well-aligned multimodal representations

4. **Open-Source Contributions**: We will release a comprehensive open-source implementation of GOTMA, including tools for analyzing and visualizing multimodal representation spaces. This will facilitate further research in this area and enable practitioners to apply our methods to their specific domains.

### 3.2 Broader Impact

The implications of this research extend beyond the immediate technical contributions:

1. **Advancing Multimodal AI**: By addressing fundamental geometric challenges in multimodal learning, our work will contribute to the development of more capable and robust multimodal AI systems. This has applications in areas such as assistive technologies, healthcare diagnostics, robotics, and human-computer interaction.

2. **Enhanced Cross-Modal Understanding**: Improved geometric alignment will enable better cross-modal transfer of knowledge, potentially reducing the amount of paired multimodal data required for training effective models. This could make multimodal learning more accessible for low-resource domains.

3. **Interpretability and Fairness**: By explicitly modeling the geometric structure of modality manifolds, our approach may lead to more interpretable multimodal representations. This could help identify and mitigate biases that might be present in one modality but not others.

4. **Theoretical Foundations**: Our work will contribute to the theoretical foundations of multimodal learning by formalizing the concept of geometric alignment and establishing its importance. This will provide a framework for future research in this area.

5. **Cross-Disciplinary Impact**: The geometric perspective on multimodal alignment has connections to fields such as differential geometry, information theory, and cognitive science. Our work may inspire cross-disciplinary research that further enriches the field of multimodal learning.

In conclusion, the proposed GOTMA framework represents a significant step forward in multimodal representation learning by explicitly addressing the geometric foundations of cross-modal alignment. By promoting structural similarity between modality manifolds, we expect to achieve more robust, transferable, and semantically meaningful multimodal representations that outperform existing approaches on a wide range of tasks.