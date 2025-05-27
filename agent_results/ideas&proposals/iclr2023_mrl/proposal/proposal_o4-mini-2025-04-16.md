Title  
“Geometric Alignment for Robust Cross-Modal Representation Learning”  

1. Introduction  
Background  
Multimodal representation learning aims to embed data from heterogeneous sources (e.g., images, text, audio) into a common latent space so that semantically similar concepts across modalities lie close together. Recent advances—such as contrastive learning (e.g. CLIP), graph-based fusion, and diffusion‐map methods—have demonstrated strong empirical performance on retrieval, classification, and generation tasks. However, these methods largely focus on instance-level alignment (matching positive pairs) without explicitly enforcing higher‐order geometric consistency between modality‐specific manifolds. As a result, the learned shared space can exhibit distortions: local neighborhoods in one modality may not correspond to those in another, leading to suboptimal fusion, brittleness under missing or noisy modalities, and poor generalization to downstream tasks requiring fine‐grained cross-modal reasoning.

Research Objectives  
This proposal investigates how to explicitly align the geometry of unimodal latent manifolds during joint training. Our key goals are:  
• To formalize geometric misalignment and design loss functions that encourage structural similarity—preserving local neighborhoods, global volume, and distributional shape—across modalities.  
• To integrate Optimal Transport (OT) and Riemannian-geometry-inspired objectives with standard contrastive or reconstruction losses, yielding a composite training criterion.  
• To empirically validate that geometry-aware alignment leads to more robust representations, measured by cross-modal retrieval accuracy, generation quality, and resilience to missing/perturbed modalities.

Significance  
A principled geometry‐aware framework tackles fundamental questions posed by the MRL workshop: (1) What properties of multimodal representations matter? (2) How can we promote robustness to modality dropout and adversarial noise? (3) What evaluation metrics best capture representation quality? By unifying instance-level and structural alignment, our work will:  
– Improve downstream performance on retrieval and generation tasks.  
– Offer new diagnostics (e.g., manifold continuity, trustworthiness) for multimodal embeddings.  
– Guide the design of architectures that scale gracefully to three or more modalities.

2. Methodology  
Our approach consists of three components: (A) encoder architecture and shared space, (B) geometric alignment losses, and (C) experimental validation protocol.  

A. Model Architecture and Data  
We assume $M$ modalities indexed by $m\in\{1,\dots,M\}$. For each modality $m$, let $\mathcal{X}_m$ be its input space (e.g. image pixels, tokenized text, audio spectrograms). We learn an encoder network $f_m:\mathcal{X}_m\to\mathbb{R}^d$ parameterized by weights $\theta_m$. All embeddings are normalized to unit length: $\|f_m(x)\|=1$. The shared latent space is thus the unit hypersphere $S^{d-1}$.  
Data: We use standard multimodal benchmarks:  
• Image–text: MSCOCO (images + captions), Flickr30K.  
• Audio–video: VGGSound (video + audio clips).  
• Three-way: image + text + audio subsets (e.g. HowTo100M).  

B. Geometric Alignment Losses  
We introduce three complementary objectives, in addition to a base contrastive loss $L_{\text{con}}$ (e.g. InfoNCE):  
 1. Instance-level contrastive alignment:  
    $$L_{\text{con}}=\frac{1}{N}\sum_{i=1}^N\Big[-\log\frac{\exp(\langle z_i^{(m)},z_i^{(n)}\rangle/\tau)}{\sum_{j}\exp(\langle z_i^{(m)},z_j^{(n)}\rangle/\tau)}\Big],$$  
    where $z_i^{(m)}=f_m(x_i^{(m)})$, $\tau$ is a temperature, and $(m,n)$ range over modality pairs.  

 2. Distributional alignment via Optimal Transport (OT):  
    Let $\{z_i^{(m)}\}_{i=1}^N$ be embeddings from modality $m$. Define empirical distributions  
    $$P_m=\frac{1}{N}\sum_{i=1}^N \delta_{z_i^{(m)}},\quad P_n=\frac{1}{N}\sum_{i=1}^N \delta_{z_i^{(n)}}.$$  
    The OT distance (Earth Mover’s Distance) is  
    $$W(P_m,P_n)=\inf_{\gamma\in\Gamma(P_m,P_n)}\mathbb{E}_{(u,v)\sim\gamma}\big[\|u-v\|\big].$$  
    We approximate $W$ via entropic-regularized Sinkhorn iterations and minimize  
    $$L_{\text{OT}}=\sum_{m<n}W(P_m,P_n).$$  

 3. Local neighborhood preservation:  
    For each modality $m$, let $\mathcal{N}_k^{(m)}(i)$ be the indices of the $k$ nearest neighbors of $z_i^{(m)}$ by cosine similarity. We define a neighborhood‐matching loss that penalizes discrepancy in neighborhood graphs:  
    $$L_{\text{NN}}=\frac{1}{NM}\sum_{m<n}\sum_{i=1}^N\big|\mathcal{N}_k^{(m)}(i)\,\triangle\,\mathcal{N}_k^{(n)}(i)\big|,$$  
    where $\triangle$ is the symmetric set difference.  

 4. Gramian volume alignment (inspired by GRAM [Cicchetti et al., 2024]):  
    Collect modality embeddings into matrices $Z^{(m)}\in\mathbb{R}^{N\times d}$; compute Gram matrices $G^{(m)}=Z^{(m)}(Z^{(m)})^\top$. Enforce volumetric consistency by:  
    $$L_{\text{GRAM}}=\sum_{m<n}\|G^{(m)}-G^{(n)}\|_F^2.$$  

Overall Objective:  
$$\min_{\{\theta_m\}_{m=1}^M}L_{\text{total}}=L_{\text{con}}+\lambda_{\text{OT}}L_{\text{OT}}+\lambda_{\text{NN}}L_{\text{NN}}+\lambda_{\text{GRAM}}L_{\text{GRAM}}.$$  
Hyperparameters $\{\lambda\}$ tune the trade-off between instance and structural alignment.

C. Algorithmic Training Procedure  
Pseudocode:  
```
Initialize encoder weights {θ_m} randomly or from pretrained (e.g. CLIP)  
for epoch in 1…E do  
  for batch B = {x_i^{(1)},…,x_i^{(M)}}_{i=1}^b do  
    compute embeddings z_i^{(m)} = f_m(x_i^{(m)})  
    compute L_con over all positive/negative pairs  
    approximate L_OT via mini-batch Sinkhorn  
    compute kNN graphs and L_NN  
    compute Gram matrices and L_GRAM  
    L_total = L_con + λ_OT L_OT + λ_NN L_NN + λ_GRAM L_GRAM  
    update θ_m ← θ_m − η ∇_{θ_m} L_total  
  end for  
end for  
```

D. Experimental Design and Evaluation Metrics  
We evaluate on:  
• Cross-modal retrieval: image→text and text→image recall@K (R@1, R@5, R@10), mean average precision (mAP).  
• Cross-modal generation: text→image generation quality measured by FID and CLIP‐Score.  
• Robustness: simulate missing modalities by randomly dropping one modality at test time; measure drop in retrieval accuracy.  
• Adversarial resilience: add Gaussian or adversarial perturbations to embeddings; measure performance degradation.  
• Manifold quality diagnostics: trustworthiness and continuity scores [Venna & Kaski, 2001], local continuity meta-criterion (LCMC).  

Ablation Studies:  
• Vary λ_OT, λ_NN, λ_GRAM individually to quantify their contribution.  
• Compare to baselines: pure contrastive, deep feature separation loss [Jiang et al., 2023], and alternating diffusion maps [Lederman & Talmon, 2019].  
• Scale to $M=3$ modalities; measure how losses generalize beyond pairs.

3. Expected Outcomes & Impact  
We anticipate the following results:  
1. Performance gains: Geometry-aware models will outperform pure contrastive baselines by 3–7% in retrieval R@1 and reduce FID in generation tasks.  
2. Improved robustness: Under missing or noisy modalities, our method will exhibit <10% drop in accuracy versus >20% for baselines.  
3. Manifold alignment metrics: Trustworthiness and continuity will increase by 15–20% over baselines, confirming better local and global geometry matching.  
4. Insights into loss synergies: Ablations will reveal that OT and Gramian terms address global shape alignment, while neighborhood loss preserves fine local structure.

Scientific Impact  
This work will:  
– Offer the first systematic integration of OT, Gramian‐volume, and neighborhood alignment for multimodal embeddings.  
– Provide new evaluation protocols (manifold metrics + robustness tests) that can become standard in the community.  
– Enhance theoretical understanding of when and why structural alignment improves downstream transfer.  
– Release a PyTorch library implementing all proposed losses, with pretrained checkpoints and evaluation scripts, to foster reproducibility.

Broader Impact  
Improved multimodal representations will benefit numerous applications—cross‐modal search engines, assistive technologies, and AI-driven content generation—by making models more reliable when inputs are noisy or partially missing. By open-sourcing our code and detailed benchmarks, we aim to accelerate progress in the MRL community and encourage principled geometry-aware design.