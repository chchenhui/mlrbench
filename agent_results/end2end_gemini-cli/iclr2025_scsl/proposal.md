Here is the research proposal based on your inputs.

***

### **1. Title: Spurious-Correlation-Aware Adapters (SCA-Adapters): Efficiently Robustifying Foundation Models via Orthogonal Gradient Projection**

### **2. Introduction**

#### **2.1. Background**
The advent of large-scale foundation models, including Large Language Models (LLMs) and Large Multimodal Models (LMMs), has marked a paradigm shift in artificial intelligence, demonstrating unprecedented capabilities in language understanding, generation, and cross-modal reasoning. These models, pre-trained on vast internet-scale datasets, serve as powerful backbones that can be fine-tuned for a multitude of downstream applications. However, their remarkable performance is often shadowed by a fundamental and pervasive vulnerability: the reliance on spurious correlations and shortcut learning.

Spurious correlations arise when a model learns to associate a target label with features that are statistically predictive in the training data but are not causally related to the underlying concept. For instance, a model classifying animals might learn to associate the "cow" label with "green pasture" backgrounds rather than the intrinsic features of a cow. This "simplicity bias" (Shah et al., 2020) causes models to fail unexpectedly and unreliably when deployed in real-world scenarios where data distributions differ from the training set—a phenomenon known as out-of-distribution (OOD) generalization failure. Such failures can have serious consequences, from perpetuating social biases in NLP systems (e.g., associating certain demographics with toxicity) to diagnostic errors in medical imaging (e.g., correlating disease with hospital-specific artifacts).

Existing methods to mitigate spurious correlations predominantly fall into two categories, both with significant limitations. The first category involves training models on datasets with explicit group annotations, which demarcate samples based on both the core and spurious attributes (e.g., a "cow on a beach" vs. a "cow on a pasture"). Methods like Group Distributionally Robust Optimization (GroupDRO) (Sagawa et al., 2019) then up-weight the loss on the worst-performing groups to force the model to learn invariant features. However, obtaining these group annotations is expensive, not scalable, and often impossible for spurious correlations that are unknown or not easily perceived by humans. The second category includes methods that aim to regularize the model's representations (e.g., ElRep by Wen et al., 2025) or retrain the model to be invariant to automatically detected shortcuts (e.g., ShortcutProbe by Zheng et al., 2025). While promising, these approaches often require full model retraining, which is computationally prohibitive for foundation models with billions of parameters. This high computational barrier drastically limits the accessibility and practicality of building robust AI systems.

Recently, Parameter-Efficient Fine-Tuning (PEFT) techniques, such as adapters (Houlsby et al., 2019) and LoRA (Hu et al., 2021), have emerged as a solution to the high cost of fine-tuning. These methods freeze the pre-trained model and only tune a small number of additional parameters. While PEFT makes fine-tuning accessible, current applications have largely focused on task adaptation rather than explicitly addressing robustness to spurious correlations. There remains a critical gap in developing a method that is both parameter-efficient *and* explicitly designed to instill OOD robustness.

#### **2.2. Research Objectives**
To bridge this gap, we propose **Spurious-Correlation-Aware Adapters (SCA-Adapters)**, a novel PEFT framework designed to efficiently robustify foundation models. Our approach explicitly disentangles the learning of core and spurious features during the fine-tuning process. We introduce two lightweight adapter modules: a "task" adapter trained to solve the primary task and a "spurious" adapter trained to solve the same task using only shortcut features. The core innovation lies in our training mechanism: we project the gradients of the task adapter to be orthogonal to the gradient direction induced by the spurious features. This manipulation nullifies updates that rely on shortcuts, compelling the task adapter to learn from core, invariant patterns.

The primary objectives of this research are:
1.  To develop the SCA-Adapter framework, a novel parameter-efficient technique for mitigating spurious correlations in large foundation models.
2.  To design and implement a method for automatically identifying spurious features from data without requiring manual group labels, which will be used to train the spurious adapter.
3.  To empirically validate the effectiveness and efficiency of SCA-Adapters on established vision and language benchmarks for spurious correlation, comparing its performance against standard fine-tuning, full-model robustification methods, and other PEFT techniques.
4.  To conduct a comprehensive analysis of the learned representations and model behavior to provide insights into how orthogonal gradient projection effectively disentangles spurious and core feature learning.

#### **2.3. Significance**
This research is significant for several reasons. **Scientifically**, it introduces a new paradigm for robust fine-tuning, combining the efficiency of PEFT with the targeted intervention of gradient-based disentanglement. It contributes to the foundational understanding of shortcut learning by demonstrating a practical mechanism to control and counteract the model's learning dynamics. **Practically**, SCA-Adapters will offer a scalable, low-cost solution for deploying robust and reliable foundation models. By dramatically reducing the computational resources required for robustification, our work will democratize the development of trustworthy AI, making it accessible to researchers and practitioners with limited resources. Ultimately, this research will pave the way for more dependable AI systems in high-stakes domains such as medicine, finance, and autonomous systems, where robustness to distributional shifts is not just desirable but essential.

### **3. Methodology**

Our proposed research will be conducted in three main stages: (1) formalizing the SCA-Adapter framework and its orthogonal gradient training algorithm, (2) developing a procedure for automatic spurious feature identification, and (3) executing a rigorous experimental validation and analysis.

#### **3.1. The SCA-Adapter Framework**
Let a pre-trained foundation model be denoted by $f_{\theta}$, with its parameters $\theta$ frozen. We augment this model by inserting two parallel, lightweight adapter modules at specific layers (e.g., after the transformer blocks).
*   **Task Adapter:** $A_{task}$ with trainable parameters $\phi_{task}$.
*   **Spurious Adapter:** $A_{spurious}$ with trainable parameters $\phi_{spurious}$.

For a given input $x$, the features from the backbone model $h = f_{\theta}(x)$ are fed into both adapters. The final prediction for the main task is derived from the task adapter's output, passed through a classification head $C_{task}$. The spurious adapter has its own classification head, $C_{spurious}$.

The core of our methodology is the training procedure, which is designed to make the knowledge learned by the task adapter and the spurious adapter mutually exclusive.

#### **3.2. Automatic Identification of Spurious Features**
A key prerequisite for our method is training the spurious adapter on data that isolates shortcut features. Since we assume no group labels are available, we propose a two-stage, data-driven approach to automatically generate this training signal.

**Stage 1: Identifying Spurious-Dominated Samples.** We posit that samples easily solvable via shortcuts are learned much faster and with lower loss during initial training. We perform a preliminary, standard fine-tuning of the model (with a single adapter) for a few epochs. We then identify a set of "easy" samples, $D_{easy}$, containing data points for which the model quickly achieves high confidence and low loss. These samples are strong candidates for being reliant on spurious correlations.

**Stage 2: Isolating Spurious Features.** For each sample $x \in D_{easy}$, we aim to create a "spurious-only" counterpart, $x_{spurious}$, which will be used to train the spurious adapter.
*   **For Vision Tasks:** Using the preliminary fine-tuned model, we compute a saliency map for $x$ using an attribution method like Grad-CAM. This map highlights the image regions most influential for the model's prediction. For shortcut-reliant samples, these regions will correspond to spurious features (e.g., the background). We then create a masked image, $x_{spurious}$, by preserving only these highly attributed regions and masking out the rest. The spurious adapter will be trained to predict the label $y$ from this masked input $x_{spurious}$.
*   **For NLP Tasks:** We use gradient-based token importance scores. For a given text $x \in D_{easy}$, we identify tokens that are highly influential but are not causally central to the task (e.g., identity terms in toxicity detection, lexical overlap in natural language inference). The input $x_{spurious}$ is then constructed by masking all but these identified spurious tokens.

This automated procedure provides the necessary data $(x_{spurious}, y)$ to train the spurious adapter to specifically recognize and exploit shortcuts.

#### **3.3. Training via Orthogonal Gradient Projection**
Our training process iteratively updates the task and spurious adapters. The key innovation is how we update the task adapter's parameters, $\phi_{task}$.

Let $\mathcal{L}_{task}(A_{task}(f_\theta(x)), y)$ be the loss for the main task (e.g., cross-entropy) on a mini-batch of original data $(x, y)$. Let $\mathcal{L}_{spurious}(A_{spurious}(f_\theta(x_{spurious})), y)$ be the loss for the spurious adapter, trained on the automatically generated spurious-only data $(x_{spurious}, y)$.

The training proceeds as follows for each mini-batch:

1.  **Update the Spurious Adapter:** We first train the spurious adapter to become an expert at using shortcuts. We compute its gradient and update its parameters:
    $$ g_{spurious} = \nabla_{\phi_{spurious}} \mathcal{L}_{spurious} $$
    $$ \phi_{spurious} \leftarrow \phi_{spurious} - \eta \cdot g_{spurious} $$
    This step ensures the spurious adapter accurately models the shortcut direction.

2.  **Compute Task and Spurious Gradients for the Task Adapter:** We compute two gradients with respect to the *task adapter's parameters*, $\phi_{task}$:
    a. The standard task gradient, which represents the direction of steepest descent for the main task:
    $$ g_{task} = \nabla_{\phi_{task}} \mathcal{L}_{task} $$
    b. The "spurious direction" gradient. This is the gradient that the *spurious loss* would induce on the *task adapter's parameters*. It represents the direction in the parameter space that would leverage shortcuts:
    $$ g_{spurious\_dir} = \nabla_{\phi_{task}} \mathcal{L}_{spurious} $$
    Note that for this computation, we use the same inputs $(x_{spurious}, y)$ as the spurious adapter, but calculate the gradient for $\phi_{task}$.

3.  **Project and Update the Task Adapter:** We project the task gradient $g_{task}$ to be orthogonal to the spurious direction gradient $g_{spurious\_dir}$. The component of $g_{task}$ that lies along the spurious direction is removed. The updated gradient $g'_{task}$ is calculated as:
    $$ g'_{task} = g_{task} - \text{proj}_{g_{spurious\_dir}}(g_{task}) = g_{task} - \frac{g_{task} \cdot g_{spurious\_dir}}{\|g_{spurious\_dir}\|^2 + \epsilon} g_{spurious\_dir} $$
    where $\epsilon$ is a small constant for numerical stability. This modified gradient, $g'_{task}$, is now "blind" to the shortcut information captured by $g_{spurious\_dir}$.

4.  Finally, we update the task adapter's parameters using this purified gradient:
    $$ \phi_{task} \leftarrow \phi_{task} - \eta \cdot g'_{task} $$

This orthogonalization process forces the task adapter to find alternative, non-spurious features to solve the task, thereby promoting the learning of core, invariant representations.

#### **3.4. Experimental Design and Evaluation**
We will conduct a comprehensive set of experiments to validate our method.

*   **Models:** We will use state-of-the-art foundation models, such as CLIP (ViT-B/16) for vision-language tasks and Llama-3-8B for language tasks.
*   **Datasets:** We will use standard benchmarks known for their well-characterized spurious correlations:
    *   **Vision:** **Waterbirds** (birds on land/water backgrounds), **CelebA** (hair color vs. gender), and **ImageNet-9 (IN-9)** subsets. For OOD evaluation, we will use **ImageNet-A** and **ImageNet-R**.
    *   **NLP:** **CivilComments-WILDS** (toxicity vs. identity group mentions) and **MNLI** (genre and hypothesis-only bias).
*   **Baselines:** We will compare SCA-Adapters against a strong suite of methods:
    1.  **Standard Fine-tuning (ERM):** Standard fine-tuning of the full model.
    2.  **Standard PEFT (LoRA):** Vanilla fine-tuning using LoRA, representing the standard parameter-efficient approach.
    3.  **Group-based Full-Model Methods:** **GroupDRO** (Sagawa et al., 2019), using ground-truth group labels as an upper-bound comparison for robustness.
    4.  **Label-Free Full-Model Methods:** Methods like **UnLearning from Experience (ULE)** (Mitchell et al., 2024) or loss-based upweighting (Ghaznavi et al., 2024) that do not require group labels.
    5.  **Multi-modal Methods:** On relevant datasets, we will compare against methods like that of Yang et al. (2023) which leverage multi-modal signals.
*   **Evaluation Metrics:** Our evaluation will be multi-faceted:
    1.  **Robustness:** Worst-Group Accuracy (WGA), calculated on the pre-defined bias-conflicting groups (e.g., landbirds on water). This is our primary metric for success.
    2.  **In-Distribution (ID) Performance:** Average accuracy on the standard test set to ensure our method does not degrade general performance.
    3.  **Efficiency:** We will report the number of trainable parameters, total training time, and peak GPU memory usage to quantify the efficiency gains of our method.
    4.  **Catastrophic Forgetting:** We will assess the model's performance on a small set of upstream tasks (e.g., from the ImageNet-1k validation set for a vision model) to ensure the pre-trained knowledge is preserved.
*   **Ablation Studies:** To understand the contribution of each component of our framework, we will conduct several ablations:
    *   **No Orthogonal Projection:** Train both adapters independently without the gradient projection step to isolate the effect of the orthogonalization.
    *   **Impact of Spurious Feature Identifier:** Compare the performance of our automatic spurious feature identification method against an oracle version that uses ground-truth segmentation masks or manually identified spurious tokens.
    *   **Adapter Architecture and Size:** We will vary the complexity (rank) and placement of the adapters to analyze their impact on performance and efficiency.

### **4. Expected Outcomes & Impact**

We anticipate that this research will yield several key outcomes and have a significant impact on the field of machine learning.

**Expected Outcomes:**
1.  **A Novel Algorithm and Public Codebase:** The primary outcome will be the SCA-Adapter framework, a novel algorithm for efficient and robust fine-tuning. We will provide a high-quality, open-source implementation of our method to facilitate adoption and further research by the community.
2.  **State-of-the-Art Robustness with High Efficiency:** We expect SCA-Adapters to achieve worst-group accuracy comparable to or exceeding computationally expensive full fine-tuning methods, while using only a fraction (<1%) of the trainable parameters and computational cost.
3.  **Empirical Validation and Best Practices:** Our extensive experiments will provide a comprehensive benchmark of SCA-Adapters against existing methods, establishing its viability as a practical solution. The results will also offer insights into best practices for applying PEFT for robustness.
4.  **Deeper Insights into Disentangled Learning:** The analysis of the learned representations and the success of the gradient projection mechanism will contribute to a deeper foundational understanding of how to actively control and steer the learning process in deep neural networks, separating unwanted biases from core knowledge.

**Impact:**
*   **Democratizing Robust AI:** By drastically lowering the computational barrier, SCA-Adapters will enable a wider range of organizations and researchers to build and deploy robust foundation models. This is crucial for fostering innovation and ensuring that the benefits of AI are not limited to entities with massive computational resources.
*   **Enhancing Trustworthiness and Safety:** Our work directly addresses a critical failure mode of modern AI systems. The ability to efficiently mitigate spurious correlations will lead to more reliable, fair, and safe models, increasing trust and enabling their deployment in sensitive applications such as healthcare, autonomous driving, and legal analysis.
*   **Paving the Way for Future Research:** The principle of using targeted gradient manipulation within a PEFT framework opens up new research avenues. Future work could extend this concept to other challenges, such as mitigating fairness biases, enhancing continual learning, or defending against adversarial attacks, all within a parameter-efficient setting. This research will serve as a foundational step towards a new class of robust and adaptable AI systems.