1. Title  
Task-Conditioned Functional Alignment for Efficient Cross-Architecture Model Merging  

2. Introduction  
Background   
Recent advances in deep learning and neuroscience have revealed that independently trained systems—whether biological or artificial—often learn similar internal representations when exposed to comparable stimuli. In machine learning, this phenomenon underpins techniques such as “model stitching” and “representation merging,” which can enable efficient reuse of pre-trained models, reduce computational costs, and improve interpretability. However, existing methods typically assume identical architectures or rely on heavy fine-tuning. When architectures differ or training distributions vary slightly, naïve parameter averaging or full retraining often fail or become prohibitively expensive.  

Research Objectives   
We propose Task-Conditioned Functional Alignment (TCFA), a novel framework for aligning and merging pre-trained neural networks with different architectures or trained on slightly different task distributions. TCFA focuses on activation‐space alignment conditioned on downstream task properties, rather than direct parameter interpolation. Our objectives are:  
• To develop algorithms that compute minimal transformations aligning activation manifolds across models per task condition.  
• To construct lightweight “stitching” layers based on those transformations, enabling merged models to perform end-to-end inference with minimal additional parameters.  
• To theoretically characterize the conditions under which functional alignment is possible, leveraging Optimal Transport (OT) theory and Canonical Correlation Analysis (CCA).  
• To empirically validate TCFA on standard vision benchmarks and demonstrate its efficiency and generalization across architectures and tasks.  

Significance   
TCFA addresses two pressing challenges in model reuse: combining heterogeneous architectures and handling small discrepancies in task distributions. By focusing on activation manifolds rather than raw weights, TCFA can exploit underlying invariances emerging during training, providing both practical gains—reduced training time, lower memory footprint—and theoretical insights into the nature of neural representations. A successful TCFA framework would facilitate modular deep‐learning pipelines, efficient model versioning, and cross-modal applications, thereby accelerating research and deployment in resource-constrained environments.  

3. Methodology  

3.1 Overview  
Given two pre-trained models \(M_1\) and \(M_2\) with potentially different architectures but addressing related tasks (e.g., image classification on slightly different distributions), TCFA proceeds in three main stages:  
  1. Task-conditioned activation probing to collect activation sets per layer.  
  2. Computation of alignment transformations using OT and CCA.  
  3. Construction and training of lightweight stitching layers.  

3.2 Data Collection and Task Conditions  
We assume access to a dataset \(\mathcal{D} = \{(x_j, y_j)\}_{j=1}^N\) that covers all task conditions of interest (e.g., classes, styles, augmentations). We partition \(\mathcal{D}\) into disjoint subsets \(\mathcal{D}_c\), each corresponding to a task condition \(c\in\mathcal{C}\). For each model \(M_i\) and each layer \(l\), we record the activations on \(\mathcal{D}_c\):  
  \[
    A_i^l(c) = \{\,a_{i,j}^l(c) = f_{i}^l(x_j)\mid (x_j,y_j)\in\mathcal{D}_c\}\subset \mathbb{R}^{d_i^l}
  \]  
where \(f_{i}^l(x)\) denotes the output of layer \(l\) in model \(M_i\), and \(d_i^l\) its feature dimension.  

3.3 Optimal Transport–Based Alignment  
We aim to find a mapping \(T_{1\to2}^l\) that aligns \(A_1^l(c)\) to \(A_2^l(c)\) with minimal cost. Let empirical distributions \(\mu_{1,c}^l\) and \(\mu_{2,c}^l\) be uniform over their samples. We solve the entropic‐regularized OT problem:  
  $$
    \mathrm{OT}_\varepsilon\bigl(\mu_{1,c}^l,\,\mu_{2,c}^l\bigr)
      = \min_{P\in\Pi(\mu_{1,c}^l,\mu_{2,c}^l)}
        \langle P,\,C\rangle \;-\;\varepsilon H(P),
  $$  
  where \(C_{jk} = \|a_{1,j}^l(c)-a_{2,k}^l(c)\|^2\) is the cost matrix,  
  \(H(P)=-\sum_{jk}P_{jk}\log P_{jk}\), and \(\Pi\) the transportation polytope.  
Using Sinkhorn’s algorithm, we compute the optimal coupling \(P^*\). We then derive a local barycentric mapping:  
  $$
    T_{1\to2}^l(a_{1,j}^l) 
      = \sum_k P^*_{jk}\,a_{2,k}^l\;\Big/\;\sum_k P^*_{jk}\,.
  $$  
By doing this for each \(c\) and averaging (or weighting by class frequency), we obtain a global mapping \(T_{1\to2}^l\).  

3.4 CCA-Based Subspace Alignment  
As an alternative or complement to OT, we apply Canonical Correlation Analysis (CCA) to find linear projections that maximize cross-model correlation. For data matrices \(X = [a_{1,j}^l(c)]\in\mathbb{R}^{d_1^l\times n}\) and \(Y=[a_{2,j}^l(c)]\in\mathbb{R}^{d_2^l\times n}\), we compute covariance blocks \(\Sigma_{XX},\Sigma_{YY},\Sigma_{XY}\). CCA solves  
  $$
    \max_{u,v}\,
    \frac{u^\top\Sigma_{XY}v}
         {\sqrt{u^\top\Sigma_{XX}u}\;\sqrt{v^\top\Sigma_{YY}v}}
  $$  
yielding pairs \(\{(u_k,v_k)\}_{k=1}^r\). We define a linear mapping  
  $$
    T_{1\to2}^l\;=\;V_r\,U_r^\dagger,
  $$  
where \(U_r=[u_1,\dots,u_r]\), \(V_r=[v_1,\dots,v_r]\), and \(U_r^\dagger\) is the pseudoinverse. This yields a low-rank linear transform aligning the dominant subspaces.  

3.5 Stitching Layer Construction  
For each layer \(l\), we insert a stitching layer \(S^l\) parameterized by \(\theta^l\) immediately after \(M_1\)’s original activation:  
  \[
    \hat a^l = S^l\bigl(a_1^l\bigr)\approx T_{1\to2}^l\bigl(a_1^l\bigr),\quad
    \hat a^l\in\mathbb{R}^{d_2^l}.
  \]  
We parametrize \(S^l\) as either:  
• A fixed linear map using \(T_{1\to2}^l\).  
• A small neural network (e.g., two-layer MLP) initialized to approximate \(T_{1\to2}^l\).  

We then merge \(M_1\) and \(M_2\) by replacing layers \(l\) in \(M_1\) with the sequence “[original layers \(\to\) stitching \(S^l\)]” and feeding into the remaining layers of \(M_2\). The resulting network \(M_{\mathrm{merged}} = M_2^{>l}\circ S^l\circ M_1^{\le l}\) can be fine-tuned end-to-end.  

3.6 End-to-End Fine-Tuning and Loss  
We train \(M_{\mathrm{merged}}\) on \(\mathcal{D}\) with cross‐entropy loss:  
  $$
    \mathcal{L}(\theta)
      = -\frac1N\sum_{j=1}^N\sum_{c\in\mathcal C}
        \mathbf{1}[y_j=c]
        \log p_{\theta}(y_j\mid x_j),
  $$  
updating only \(\{\theta^l\}\) (and optionally last-layer weights) to reduce computational cost.  

3.7 Experimental Design  

Datasets and Architectures  
– Vision tasks: CIFAR-10, CIFAR-100, a 100-class subset of ImageNet.  
– Architectures: ResNet-50 vs. DenseNet-121; VGG-16 vs. ResNet-18; optionally ViT vs. ResNet.  

Baselines  
– Naïve parameter averaging.  
– Model stitching via random permutation alignment.  
– Full fine-tuning of Model 2 on Dataset of Model 1.  

Evaluation Metrics  
– Classification accuracy of \(M_{\mathrm{merged}}\) vs. original models.  
– Alignment quality: CKA similarity  
  $$
    \mathrm{CKA}(X,Y) 
      = \frac{\|K_XHK_YH\|_F}{\|K_XH\|_F\|K_YH\|_F},
  $$  
  where \(K_X\) is Gram matrix of activations and \(H\) is centering.  
– Parameter and FLOPs overhead: additional parameters in stitching layers vs. full fine-tuning.  
– Convergence speed: epochs to reach within 1% of original accuracy.  

Ablations  
– OT vs. CCA vs. hybrid alignment.  
– Different numbers of task conditions \(|\mathcal C|\).  
– Rank \(r\) in CCA, regularization \(\varepsilon\) in OT.  

4. Expected Outcomes & Impact  

We anticipate that TCFA will enable merged models to achieve >98% of the parent models’ accuracy while requiring ≤5% of the extra parameters and 20–50% of the fine-tuning epochs. The final merged networks should exhibit high CKA similarity (>0.9) to the target architecture, confirming effective alignment.  

Impact on Research and Practice:  
• Resource Efficiency: By avoiding full retraining, TCFA promises significant savings in GPU time, energy, and memory.  
• Model Modularity: Practitioners can build complex systems by merging pre-trained components, facilitating transfer across modalities and tasks.  
• Theoretical Insights: Our use of OT and CCA will shed light on the geometry of activation manifolds and the conditions under which different architectures converge to similar representations.  
• Neuroscience Connections: Understanding task-conditioned alignment may inspire new experiments in biological systems, informing theories of canonical representations.  

Broader Implications:  
TCFA could become a cornerstone methodology in model versioning, continual learning, and federated learning, where heterogeneous architectures must interoperate. Moreover, insights gained may guide the design of architectures inherently amenable to alignment, bridging gaps between AI and neuroscience in our understanding of representation emergence.