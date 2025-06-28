Title  
Adversarial Counterfactual Augmentation for Spurious Correlation Robustness  

1. Introduction  
Background  
Machine learning models trained by empirical risk minimization (ERM) frequently exploit shortcuts—spurious correlations present in training data that are not causally related to the target task. For example, in chest‐X-ray diagnosis, models may key on scanner artifacts or hospital‐specific markers instead of pulmonary pathology; in natural language inference (NLI), models may rely unduly on lexical overlap between premise and hypothesis; and in polygenic risk scoring, models trained on European‐ancestry cohorts generalize poorly to other populations. Such reliance on spurious features leads to brittle out-of-distribution (OOD) performance and raises concerns in safety‐critical and fairness‐sensitive applications.  

Research Objectives  
This proposal develops a unified framework—Adversarial Counterfactual Augmentation (ACA)—to systematically identify spurious features in a pretrained model, synthesize semantically faithful counterfactual examples that perturb only those features, and retrain the model under a consistency‐enforcing objective. Concretely, our objectives are:  
1. Design a robust, annotation-free procedure for spurious feature discovery via gradient‐based attribution and influence functions.  
2. Develop a conditional generative method (e.g., CycleGAN or diffusion model) that produces high-fidelity counterfactuals, modifying only identified spurious regions or attributes.  
3. Integrate the counterfactuals into model retraining with a consistency loss that enforces invariance to spurious perturbations.  
4. Theoretically and empirically validate that ACA reduces reliance on spurious cues, improves worst‐group accuracy, and closes the OOD performance gap.  

Significance  
ACA addresses two critical bottlenecks in prior work: the need for expensive group or spurious‐attribute annotations, and difficulties of achieving full invariance when spurious features are unknown or complex. By combining automated feature identification with generative augmentation and consistency‐based invariance training, ACA is expected to yield robust, stable models across vision, language, and genomics, thereby advancing best practices for spurious-correlation mitigation.  

2. Methodology  
2.1 Problem Formulation  
Let D = { (x_i, y_i) }_{i=1}^N be i.i.d. samples from a distribution P over X×Y. We assume each input x can be decomposed into causal features c and spurious features s so that y ⟂ s | c, but ERM models learn f̂ that may depend on s to minimize training loss. Our goal is to learn f_θ: X→Y that satisfies:  
  f_θ(c, s₁) ≈ f_θ(c, s₂)  ∀ s₁,s₂,  
thus achieving invariance to s and robustness under shifts in P(s|c).  

2.2 Spurious Feature Identification  
We employ two complementary techniques to detect candidate spurious features in a pretrained ERM model f_ψ:  
1. Gradient Attribution. Compute saliency map A_g(x) ∈ R^d via  
   A_g(x) = ∂ℓ(f_ψ(x),y)/∂x.  
   We threshold |A_g(x)| to obtain a mask m_g(x) ∈ {0,1}^d indicating regions with large gradients that may reflect shortcut reliance.  
2. Influence Functions. Using the influence‐function approximation (Koh & Liang, 2017), estimate the effect of each training point z = (x,y) on the loss at a test point z':  
   I(z,z') = -∇_ψ ℓ(f_ψ(z'),y')^T H_ψ^{-1} ∇_ψ ℓ(f_ψ(z),y),  
where H_ψ is the Hessian of the empirical risk. High‐magnitude influences identify training examples that shape the model’s decision via spurious correlations. We cluster such points and derive a mask m_i(x) by averaging saliency over the cluster.  

We combine masks:  
  m(x) = m_g(x) ∨ m_i(x).  

2.3 Counterfactual Generation  
Given x and its spurious mask m(x), we train a conditional generator G_φ that synthesizes x' = G_φ(x, m) with altered spurious content while preserving causal content. Two architectures are considered:  
• CycleGAN‐Style Translator. We treat (x, m) as two‐channel input and learn mapping to the counterfactual domain with cycle consistency. The objective is:  
  L_GAN = E_x[ log D(G(x,m)) ] + E_{x'}[ log (1−D(x')) ],  
  L_cyc = E_x[ ||(1−m)⊙G(G(x,m),m) − (1−m)⊙x||₁ ],  
  L_rec = E_x[ ||(1−m)⊙G(x,m) − (1−m)⊙x||₁ ].  
Total:  
  L_gen = L_GAN + λ_cyc L_cyc + λ_rec L_rec.  

• Diffusion‐based Editor. Condition a denoising diffusion probabilistic model on (x, m) to sample x' by selectively corrupting masked regions, then denoising. We use the loss  
  L_diff = E_{t,x}[ ||ϵ_θ(x_t, m, t) − ϵ||₂² ],  
where x_t is a noisy version at timestep t.  

2.4 Invariance-Enforcing Retraining  
We augment the original dataset with generated counterfactuals:  
  D_aug = D ∪ { ( G_φ(x_i, m_i), y_i ) }_{i=1}^N.  
We train a fresh model f_θ using the combined classification+consistency objective:  
  L_CE = E_{(x,y)∼D_aug}[ ℓ(f_θ(x), y) ],  
  L_cons = E_{(x,y)∼D}[ d( f_θ(x), f_θ(G_φ(x, m(x))) ) ],  
where d is an ℓ₂ distance between logit vectors. The final objective is:  
  L_total(θ) = L_CE + α L_cons.  

2.5 Theoretical Justification  
Under a simplified linear model f_θ(x)=θᵀx and perfect generator coverage of the spurious subspace S, the consistency loss enforces θ⊥S. Formally, if span(S) = { s } and G spans all directions in S, minimization of L_cons yields:  
  θᵀ s = 0, ∀ s ∈ S,  
thus guaranteeing f_θ(x) depends only on causal subspace C. We will extend this analysis to non­linear feature extractors using Rademacher‐type generalization bounds under adversarial perturbations restricted to S.  

2.6 Experimental Design  
Datasets and Tasks  
1. Waterbirds (vision): bird species classification with bird ∥ background spurious correlation.  
2. CelebA (vision): attribute classification (e.g., “Smiling”) spurious w.r.t. hair color.  
3. MNLI (language): natural language inference with lexical overlap spuriousity.  
4. Polygenic Risk (genomics): disease risk across ancestry groups.  

Baselines  
• ERM, IRM (Invariant Risk Minimization), GroupDRO, Spectral Decoupling, JTT (Just Train Twice).  
• Recent annotation‐free methods: EVaLS, self‐guided mitigation, subnetwork extraction.  

Metrics  
• Average accuracy on ID and OOD splits.  
• Worst‐group accuracy (min over spurious groups).  
• OOD gap: Δ = Acc_ID − Acc_OOD.  
• Calibration error (ECE) and stability‐to‐perturbations.  

Ablations  
• Without feature identification (use random masks).  
• Without consistency loss (only augmented classification).  
• Vary α in L_total.  
• Generator architectures (CycleGAN vs diffusion).  

Implementation Details  
We will use ResNet‐50 for vision tasks and BERT‐base for NLI, fine‐tuned with ACA. Generators will be U-Net diffusion models or 9-block ResNet translators. Training uses AdamW with learning rates tuned on a validation OOD split.  

3. Expected Outcomes & Impact  
We anticipate that ACA will achieve:  
1. Significant improvements in worst‐group accuracy (≥10% over ERM) on Waterbirds and CelebA without any group labels.  
2. Reduced OOD generalization gap Δ by 50% relative to IRM and GroupDRO.  
3. Empirical demonstration of invariance: feature‐space correlation with spurious attributes drops to near zero.  
4. Theoretical bounds validating that consistency training removes linear dependence on spurious subspaces under idealized conditions.  

Broader Impact  
ACA provides a practical recipe for spurious‐correlation robustness without requiring costly annotations. In healthcare, it can mitigate scanner‐ and hospital‐specific shortcuts, leading to safer diagnostic models. In language, it curbs superficial lexical reliance, resulting in genuinely reasoning-based NLI. In genomics, it promotes equitable risk scoring across ancestries. More generally, ACA combines causal insights, generative modeling, and consistency‐based invariance, offering a unifying framework applicable to vision, language, and structured data. We expect our code, pretrained generators, and benchmark splits to become a valuable resource for the community, fostering future research on reliable, stable ML under spurious correlations.  

References  
[References to the ten reviewed papers, Koh & Liang (influence functions), cycleGAN, diffusion models, IRM, GroupDRO, JTT, etc., will be included in the full submission.]