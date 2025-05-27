Title  
=======  
Regulation-Sensitive Dynamic Differential Privacy for Federated Learning: Balancing Model Utility and GDPR Compliance  

Introduction  
============  
Background  
----------  
Federated learning (FL) enables multiple clients (e.g., hospitals, banks) to collaboratively train a global model without sharing raw data, addressing data-locality and compliance concerns. To defend against information leakage through model updates, differential privacy (DP) has emerged as the gold-standard: each client perturbs its updates with carefully calibrated noise, ensuring that the inclusion or exclusion of any single record has only a bounded effect on the model. However, standard DP mechanisms in FL typically treat all model parameters or data dimensions with a uniform privacy budget, ignoring that real-world features differ greatly in their legal sensitivity. Meanwhile, regulations such as the EU’s General Data Protection Regulation (GDPR) classify personal data into tiers of risk (e.g., “special category” data vs. benign metadata), impose data-minimization principles, and require accountability through audit trails.  

Research Objectives  
-------------------  
1. Develop an end-to-end FL framework that automatically tags features by regulatory sensitivity, using metadata and lightweight NLP.  
2. Design a dynamic DP budget allocator that assigns per-feature (or per-feature-group) privacy parameters $(\epsilon_g,\delta_g)$ in proportion to their legal sensitivity, while satisfying a global privacy constraint $(\epsilon_{\text{total}},\delta_{\text{total}})$.  
3. Build a secure aggregator that enforces this tailored noise injection and produces immutable audit logs for every training round, enabling third-party verification of compliance with GDPR’s data-minimization and accountability clauses.  
4. Empirically demonstrate on healthcare and financial datasets that our regulation-sensitive scheme improves model utility by up to 30% over uniform-DP baselines at equal total privacy cost, while providing provable compliance guarantees.  

Significance  
------------  
Our proposal bridges a critical gap between legal requirements and technical practice in privacy-preserving ML. By aligning noise injection with feature risk levels mandated by regulation, we both maximize model performance and ensure adherence to GDPR's principles of data-minimization (Art. 5(1)(c)) and accountability (Art. 5(2)). The immutable audit log fosters transparency, enabling regulators or auditors to verify privacy spending and noise calibration without accessing raw data. This interdisciplinary approach paves the way for wider, compliant adoption of FL in sensitive domains.  

Methodology  
===========  
Our methodology comprises five major components: (1) Feature Sensitivity Classification, (2) Dynamic Privacy Budget Allocation, (3) Differentially Private Secure Aggregation, (4) Privacy Accounting & Audit Logging, and (5) Experimental Validation.  

1. Feature Sensitivity Classification  
-------------------------------------  
• Input: A schema describing each feature’s name, metadata tags (e.g., “age,” “diagnosis code,” “transaction amount”), and optionally free-text annotations.  
• NLP-Based Tagging: We train a lightweight transformer or single-layer BiLSTM classifier $C_{\text{sens}}$ that maps feature descriptions to a sensitivity score $s_g\in[0,1]$, where $s_g\approx1$ denotes “highly regulated” (e.g., health diagnosis) and $s_g\approx0$ denotes “low risk” (e.g., device ID). The classifier is pre-trained on a small manually labeled corpus reflecting GDPR categories.  
• Feature Grouping: For models with high-dimensional inputs, we cluster features into $G$ groups (e.g., using domain knowledge or $k$-means on embeddings), assigning each group a representative sensitivity $s_g$.  

2. Dynamic Privacy Budget Allocation  
------------------------------------  
Given a global privacy budget $(\epsilon_{\text{total}},\delta_{\text{total}})$ and sensitivity scores $\{s_g\}_{g=1}^G$, we allocate per-group budgets $(\epsilon_g,\delta_g)$ such that  
  
$$  
\epsilon_g = \frac{s_g}{\sum_{h=1}^G s_h}\,\epsilon_{\text{total}},  
\qquad  
\delta_g = \frac{s_g}{\sum_{h=1}^G s_h}\,\delta_{\text{total}}.  
$$  

This ensures that high-sensitivity groups receive more privacy protection (smaller $\epsilon_g$) relative to their regulated risk, while the overall mechanism remains $(\epsilon_{\text{total}},\delta_{\text{total}})$-DP by the basic composition theorem.  

3. Differentially Private Secure Aggregation  
-------------------------------------------  
We extend the standard Federated Averaging (FedAvg) protocol as follows:  

  3.1. Local Update & Clipping  
  • At round $t$, each selected client $i$ computes its local gradient $g_i^t$ on its private data.  
  • We clip per-feature-group gradients to norm $C_g$ (to bound sensitivity):  
    
  $$  
  \bar{g}_{i,g}^t = g_{i,g}^t \;/\;\max\!\Bigl(1,\frac{\|g_{i,g}^t\|_2}{C_g}\Bigr),  
  $$  
    
  where $g_{i,g}^t$ is the subvector corresponding to group $g$.  

  3.2. Noise Injection  
  • For each group $g$, we compute Gaussian noise scale  

  $$  
  \sigma_g = \frac{C_g\sqrt{2\ln(1.25/\delta_g)}}{\epsilon_g}.  
  $$  

  • The client perturbs its clipped update:  

  $$  
  \widetilde{g}_{i,g}^t = \bar{g}_{i,g}^t + \mathcal{N}(0,\,\sigma_g^2\,I).  
  $$  

  3.3. Secure Aggregation  
  • Clients secret-share their noisy updates to a set of non-colluding servers (using Shamir’s Secret Sharing).  
  • The servers reconstruct the aggregated update without learning individual contributions:  

  $$  
  \widetilde{G}_g^t = \frac{1}{|\mathcal{C}_t|}\sum_{i\in\mathcal{C}_t}\widetilde{g}_{i,g}^t.  
  $$  

  • The global model is updated via standard SGD: $w^{t+1} = w^t - \eta\,\widetilde{G}^t$.  

4. Privacy Accounting & Audit Logging  
--------------------------------------  
4.1. RDP Accountant  
We use Rényi Differential Privacy (RDP) to track the cumulative privacy loss across $T$ rounds. For each group $g$, the Gaussian mechanism in one round yields $\alpha$-order RDP guarantee $\rho_g(\alpha)$; after $T$ rounds:  

$$  
\rho_{g,\text{total}}(\alpha) = T\,\rho_g(\alpha).  
$$  

We then convert back to $(\epsilon_{\text{total}},\delta_{\text{total}})$ via standard RDP-to-DP bounds.  

4.2. Immutable Audit Log  
After each round, the aggregator produces a log entry containing:  
• Round index $t$.  
• Allocated budgets $(\epsilon_g,\delta_g)$.  
• Clipping bounds $C_g$.  
• Noise seeds or hashed commitments.  
• Aggregated update hash $h(\widetilde{G}^t)$.  
• Timestamp and digital signature of the server.  

These entries are appended to a tamper-evident ledger (e.g., implemented via a permissioned blockchain or a Merkle tree). Auditors or regulators can inspect the log to verify that the sum of allocated $\epsilon_g$ matches the declared $\epsilon_{\text{total}}$, that noise commitments are consistent, and that no undeclared privacy spending occurred.  

5. Experimental Validation  
--------------------------  
5.1. Datasets  
• Healthcare: MIMIC-III with features such as vital signs (low sensitivity), diagnoses (high sensitivity), demographic metadata (medium).  
• Finance: UCI German Credit with transactional features (medium), credit history (high), device metadata (low).  

5.2. Baselines  
• Uniform-DP FedAvg: $\epsilon_g = \epsilon_{\text{total}}/G$.  
• Time-adaptive DP (Kiani et al., 2025): budgets vary by round but uniformly across features.  
• Static per-feature DP: budgets set by domain expert but not data-driven.  

5.3. Evaluation Metrics  
• Model Utility: classification accuracy, AUC-ROC, F1.  
• Privacy-Utility Trade-off: utility vs. total $\epsilon_{\text{total}}$.  
• Privacy Leakage Risk: success rate of membership inference attacks on test points.  
• Compliance Metrics: audit log completeness score, regulator satisfaction in a user study.  

5.4. Protocol  
For each dataset and each mechanism (ours vs. baselines):  
1. Sweep $\epsilon_{\text{total}}\in\{0.1,0.5,1.0,2.0\}$ with fixed $\delta=10^{-5}$.  
2. Train for $T=100$ rounds, clipping norms $C_g$ tuned by grid search.  
3. Repeat each experiment 10× with different random seeds; report mean±std.  
4. Conduct membership inference attacks (e.g., Shokri et al.) to assess leakage on each feature group.  
5. Present empirical RDP tracks vs. theoretical budgets to validate the privacy accountant.  
6. Recruit privacy officers and policy-makers for a small usability study: present them the audit logs and assess whether they can verify GDPR compliance.  

Expected Outcomes & Impact  
==========================  
Scientific Contributions  
------------------------  
1. A novel dynamic DP budget allocator that tailors noise levels to regulatory sensitivity, improving the privacy-utility frontier in federated learning.  
2. A formal privacy accounting scheme combining group-wise Gaussian mechanisms with advanced composition and RDP, offering tight bounds on cumulative privacy loss.  
3. An operational audit-log protocol that makes DP budgets and noise realization publicly verifiable without compromising client data, paving the way for regulatory certification.  

Empirical Findings  
------------------  
• We anticipate up to 30% improvement in classification accuracy or AUC over uniform-DP baselines at equal total $\epsilon_{\text{total}}$.  
• Membership inference risk on high-sensitivity features will be at least 20% lower than in uniform schemes.  
• User studies with auditors will show that our audit logs enable quick verification of privacy spending and compliance with GDPR Article 30 (records of processing activities) and Article 5(1)(c) (data minimization).  

Broader Impact  
--------------  
• Adoption: Our open-source framework (code, pre-trained sensitivity classifier, audit-log tool) will accelerate deployment of privacy-preserving FL in healthcare, finance, and beyond.  
• Policy: By concretely mapping DP budgets to regulatory requirements, we provide a template for regulators to define “privacy knobs” in ML systems and for vendors to justify compliance.  
• Trust & Transparency: Immutable audit trails will strengthen trust among data subjects, organizations, and oversight bodies, fostering wider collaboration in sensitive domains.  

In summary, this research plan offers a technically rigorous, legally informed, and experimentally validated approach to privacy-preserving federated learning, directly addressing the dual demands of model performance and regulatory accountability. We expect our regulation-sensitive DP framework to become a cornerstone in the responsible deployment of AI in regulated industries.