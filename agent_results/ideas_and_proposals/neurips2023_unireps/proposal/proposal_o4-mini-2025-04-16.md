Title  
Optimal Transport-Based Cross-Modal Representation Alignment for Seamless Model Merging (“AlignMerge”)

1. Introduction  
Background  
Recent advances in both neuroscience and artificial intelligence have revealed a striking phenomenon: when exposed to similar sensory inputs, distinct neural systems—whether biological brains or independently initialized artificial networks—tend to converge on analogous internal representations. In multimodal machine learning, this manifests as semantically aligned embeddings for co‐occurring signals (e.g. an image and its caption). Harnessing and unifying these naturally emergent alignments promises  
  • Parameter‐efficient fusion of pre‐trained, uni‐modal models,  
  • Reduced training cost via model reuse instead of end‐to‐end training,  
  • Deeper theoretical understanding of identifiability and invariances in representation space, and  
  • New capabilities in robotics, embodied AI, and cross‐modal reasoning.  

Research Objectives  
This proposal introduces AlignMerge, a principled framework that (i) computes an optimal‐transport (OT) alignment between latent distributions of two pre‐trained uni‐modal models, (ii) learns an invertible mapping to preserve each model’s functionality, and (iii) fuses the aligned representations via lightweight, adaptive cross‐attention layers—avoiding full re‐training. Our key objectives are:  
  1. Design and implement an entropic‐regularized OT pipeline for large‐scale, paired cross‐modal features.  
  2. Parameterize and train an invertible neural mapping $T_\theta$ to pull one modality’s features onto the other’s latent manifold, with theoretical identifiability guarantees.  
  3. Fuse the aligned embeddings through adaptive cross‐attention modules for downstream joint tasks (e.g. Visual Question Answering, image‐text retrieval).  
  4. Rigorously validate alignment quality, invertibility, and task performance against jointly trained baselines and state‐of‐the‐art merging methods.  

Significance  
By avoiding costly joint pre‐training and preserving individual model integrity, AlignMerge can:  
  • Democratize multimodal system construction in low‐compute settings,  
  • Provide a modular “plug‐and‐play” paradigm for pre‐trained vision/language models,  
  • Illuminate the geometric and information‐theoretic underpinnings of representation similarity, and  
  • Bridge theory (identifiability in neural maps) with practice (efficient multimodal fusion).  

2. Related Work  
Cross‐Modal OT Alignment  
  • AlignMamba [Yan et al., 2024] uses token‐level OT and MMD.  
  • CMOT [Zhou et al., 2023] applies OT‐based token mixup for speech‐text, improving BLEU.  
  • Smith et al. (2023) and Martinez et al. (2023) propose global OT on latent features but do not guarantee invertibility.  

Model Merging & Stitching  
  • Sung et al. (2023) empirically study transformer merging across modalities, introducing weight‐distance metrics.  
  • DecAlign (Qian et al., 2025) hierarchically decomposes features into common/unique parts with prototype‐guided OT.  

Identifiability in Neural Maps  
  • Recent work in functional identifiability [Raghu et al., 2024] shows non‐convex models admit only permutation‐scaling ambiguities under mild conditions.  
  • Invertible neural networks (e.g. RealNVP, NICE) offer a route to bijective cross‐modal mappings but have not been combined with OT alignment.  

Gap Analysis  
Existing OT‐based alignment methods either lack invertible mappings, rely on ad‐hoc global statistics (e.g. MMD) without local semantic matching, or require significant finetuning of large backbone weights. There is, to our knowledge, no framework that (1) computes precise
pair‐wise OT couplings on large batches, (2) parameterizes a provably invertible map, and (3) fuses via lightweight attention without retraining full models.  

3. Methodology  
Overview  
Given two pre‐trained, frozen feature extractors  
  $$f_v: \mathcal{X}_{\rm image} \to \mathbb{R}^d,\quad f_t: \mathcal{X}_{\rm text}\to \mathbb{R}^d,$$  
and a paired dataset  
  $$\mathcal{D} = \{(x^v_i,x^t_i)\}_{i=1}^N,$$  
we will learn:  
  1. An entropic OT coupling $\gamma^\star$ between empirical distributions of $\{z^v_i = f_v(x^v_i)\}$ and $\{z^t_i = f_t(x^t_i)\}$.  
  2. An invertible mapping $T_\theta:\mathbb{R}^d\!\to\!\mathbb{R}^d$ that aligns $z^v$ to $z^t$ under $\gamma^\star$.  
  3. A lightweight fusion module $\Phi_\phi$ (cross‐attention) over $(T_\theta(z^v),z^t)$ for downstream tasks.  

3.1 Entropic‐Regularized Optimal Transport  
Define empirical measures  
  $$\mu_v = \frac1N\sum_{i=1}^N \delta_{z^v_i},\quad \mu_t = \frac1N\sum_{j=1}^N \delta_{z^t_j},$$  
and cost matrix $C\in\mathbb{R}^{N\times N}$ with  
  $$C_{ij} = \|z^v_i - z^t_j\|_2^2.$$  
We solve  
  $$
    W_\varepsilon(\mu_v,\mu_t) 
    = \min_{\gamma\in\Pi(\mu_v,\mu_t)} \langle \gamma,\,C\rangle
      - \varepsilon\,H(\gamma),
  $$
where $H(\gamma) = -\sum_{ij}\gamma_{ij}\log\gamma_{ij}$ and $\Pi(\mu_v,\mu_t)$ enforces row‐/column‐sums $1/N$.  
We implement the Sinkhorn‐Knopp algorithm:  
  1. $K = \exp(-C/\varepsilon)$  
  2. Iterate $u\leftarrow 1/(K v),\;v\leftarrow 1/(K^\top u)$ for 50–100 iters.  
  3. Return $\gamma = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)$.  

3.2 Invertible Mapping $T_\theta$  
We parameterize $T_\theta$ as a coupling of $L$ RealNVP/Glow‐style steps ensuring $\det\!\bigl(\partial T/\partial z^v\bigr)\neq0$.  
Objective:  
  $$
    \min_\theta\;\sum_{i,j}\gamma^\star_{ij}\,\big\|T_\theta(z^v_i)-z^t_j\big\|_2^2
      \;+\;\lambda\,\mathcal{R}(\theta),
  $$  
where $\mathcal{R}$ is weight‐decay. This encourages $T_\theta$ to respect the OT coupling.  
Invertibility Guarantee  
Under the architecture of RealNVP, $T_\theta$ is bijective by construction. A direct consequence is that for any $z^v$ we can recover  
  $$z^v = T^{-1}_\theta\bigl(T_\theta(z^v)\bigr).$$  

3.3 Identifiability Analysis  
Proposition.  Suppose $f_v,f_t$ produce feature clouds whose empirical distributions have full support and $T_\theta$ exactly minimizes the OT alignment objective. Then, up to measure‐zero permutations within degenerate clusters, $T_\theta$ is the unique bijective map that preserves the coupling.  
Proof Sketch.  Follows from Brenier’s theorem on unique gradients of convex functions for squared‐Euclidean cost, extended to the entropic‐regularized setting [Genevay et al., 2018].  

3.4 Adaptive Cross‐Attention Fusion  
We fuse the aligned features for downstream tasks using a multi‐head cross‐attention block (MHCA):  
  $$  
    \mathrm{MHCA}(Q,K,V) 
      = \mathrm{Concat}_{h=1}^H\bigl(\mathrm{softmax}(\tfrac{QW^Q_h (KW^K_h)^\top}{\sqrt{d_h}})\,VW^V_h\bigr)\,W^O,  
  $$  
where we set  
  $$Q = T_\theta(z^v),\quad K = z^t,\quad V = z^t,$$  
and learn weights $\{W^Q_h,W^K_h,W^V_h,W^O\}$. Optionally we include a symmetric block with roles swapped. A small feed‐forward network and residual connections follow each attention layer.  

3.5 Training & Losses  
We freeze $f_v,f_t$ and jointly train $T_\theta$ and MHCA parameters $\phi$ with the composite loss:  
  $$  
    \mathcal{L} 
    = W_\varepsilon(\mu_v,\mu_t) 
      + \alpha\,\mathcal{L}_{\rm task}\bigl(\Phi_\phi(T_\theta(z^v),z^t)\bigr)
      + \beta\,\mathcal{L}_{\rm rec},  
  $$  
  • $\mathcal{L}_{\rm task}$: cross‐entropy for VQA, contrastive loss for retrieval, etc.  
  • $\mathcal{L}_{\rm rec} = \frac1N\sum_i\big\|T^{-1}_\theta(T_\theta(z^v_i))-z^v_i\big\|_2^2$ (reconstruction penalty).  

3.6 Experimental Design  
Datasets  
  • Retrieval: MS‐COCO captions (images vs. captions), Flickr30k.  
  • VQA: VQA v2.0 benchmarks.  

Backbones  
  • Vision: ResNet‐50 / ViT‐Base.  
  • Language: BERT‐Base / RoBERTa.  

Baselines  
  1. Jointly trained multimodal transformers.  
  2. Naïve merging: linear Procrustes mapping without OT.  
  3. AlignMamba (local OT + MMD).  
  4. DecAlign (hierarchical OT).  

Metrics  
  • Task performance: top‐1 accuracy (VQA), recall@1/5/10 (retrieval).  
  • Alignment quality: average post‐alignment Wasserstein distance, cosine similarity increase, CKA score.  
  • Invertibility: average reconstruction error $\mathcal{L}_{\rm rec}$.  
  • Efficiency: GPU hours vs. full joint training.  

Implementation Details  
  • Dimension $d=512$, entropic $\varepsilon=0.05$, learning rate $1\!\times\!10^{-4}$ (Adam).  
  • Batch size 256 pairs, train for 50 epochs.  
  • Ablations: amount of paired data, $\alpha,\beta$ weights, number of attention heads $H$.  

4. Expected Outcomes & Impact  
Expected Outcomes  
  1. **Alignment Quality**  
     – 40–60% reduction in Wasserstein cost vs. naïve mapping.  
     – 20–30% relative increase in cosine alignment and CKA similarity.  
  2. **Task Performance**  
     – Within 1–2% accuracy of fully jointly trained multimodal networks on VQA v2.  
     – Comparable retrieval Recall@1@5@10 to joint baselines.  
  3. **Invertibility**  
     – Reconstruction error $\mathcal{L}_{\rm rec}<10^{-3}$, demonstrating bijectivity.  
  4. **Efficiency**  
     – 70–80% reduction in GPU compute compared to end‐to‐end pre‐training.  

Broader Impact  
  • **Compute‐Efficient AI**: Enables resource‐constrained labs to build high‐performance multimodal systems without massive joint pre‐training.  
  • **Modular Robotics & Embodied AI**: Plug vision and language modules into a unified agent on‐the‐fly.  
  • **Theoretical Insights**: Advances understanding of identifiability and invariances in learned neural representations, with potential cross‐pollination to neuroscience.  
  • **Sustainability**: Reduces carbon footprint of large‐scale AI by promoting model reuse.  
  • **Ethical Considerations**: By modularizing models, one can audit, debug or replace single modalities independently, enhancing system transparency and fairness.  

In summary, AlignMerge offers a novel, theoretically grounded, and practically efficient paradigm for unifying representations across modalities. It stands to transform how we merge, reuse, and interpret pre‐trained neural models—paving the way to more sustainable, transparent, and powerful multimodal AI systems.