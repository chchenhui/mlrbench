Title: FineActionBench: A Benchmark for Fine-Grained Temporal Video-Language Alignment

1. Introduction  
Background  
Recent advances in video-language modeling have unlocked new capabilities in video search, captioning, and autonomous agent perception. However, most existing benchmarks and applications focus on coarse-grained tasks—e.g., retrieving entire videos given a caption or generating a single descriptive caption per clip—without requiring models to pinpoint exactly when in a video a given textual phrase applies. In contrast, many real-world tasks (robot instruction following, video editing, surveillance alerting) demand precise temporal grounding of language: aligning each short instruction or description to the exact moment or action segment it references. The scarcity of fine-grained temporal benchmarks impedes progress toward models that can truly “understand” video at the level of events and sub-actions.  

Research Objectives  
We propose to fill this gap by creating FineActionBench, a large-scale benchmark that:  
• Curates a diverse set of multi-step, complex activity videos (cooking, DIY assembly, sports drills).  
• Provides densely annotated, temporally localized phrase-to-segment mappings: each short textual phrase is grounded to its precise start and end times in the video.  
• Defines rigorous evaluation metrics—such as phrase-localized Temporal Intersection over Union (T-IoU), recall@K thresholds, and mean alignment error—for quantifying fine-grained temporal understanding.  
• Evaluates state-of-the-art video-language models and establishes strong baselines.  

Significance  
FineActionBench will serve as a crucial driving force for:  
• Precise robot action understanding and planning, where every instruction must be executed at the correct time.  
• Video editing and indexing tools that allow users to jump directly to the moment a described event occurs.  
• Surveillance and safety systems that trigger alarms only when a specific sub-action (e.g., “person picking up an object”) happens.  
• The broader research community, by providing a standardized, open benchmark that pushes beyond coarse video-text matching toward true temporal comprehension.  

2. Methodology  
2.1 Dataset Curation and Annotation  
Video Selection  
– Source raw videos from public platforms (YouTube, instructional sites) under Creative Commons licenses.  
– Select videos depicting complex, multi-step activities including cooking (e.g., “chopping onions,” “stirring sauce”), furniture assembly (“align screw holes,” “tighten bolt”), and sports drills (“dribble between cones,” “shoot at goal”).  
– Restrict length to 30 seconds–5 minutes to balance temporal complexity and annotation cost.  

Annotation Protocol  
– Annotation Interface: Develop a web-based tool where annotators can view video playback, create new phrase annotations, and specify exact start/end timestamps.  
– Annotation Guidelines:  
  1. Decompose the video into semantically meaningful sub-actions (5–20 per video).  
  2. Write concise phrases (3–8 words) describing each sub-action.  
  3. Mark start and end times to the nearest 0.1 second.  
– Quality Control: Each video is annotated by two independent workers. Conflicts (IoU < 0.5 between their segment boundaries) are resolved by expert reviewers.  

Dataset Statistics  
– Target: 2,000 videos, ∼25 sub-action annotations per video → 50,000 phrase-to-segment instances.  
– Split: 1,400 train / 300 validation / 300 test.  

2.2 Evaluation Metrics  
Temporal Intersection over Union (T-IoU)  
For a predicted segment ˆs = [ˆt₁, ˆt₂] and ground truth segment s = [t₁, t₂], define  
$$
\mathrm{T{\text -}IoU}(s, \hat s)=\frac{\min(t_2,\hat t_2)-\max(t_1,\hat t_1)}{\max(t_2,\hat t_2)-\min(t_1,\hat t_1)}
$$  
if the intersection is positive, else 0.  

Recall@K, IoUₜ  
Recall@K, IoUₜ measures the fraction of phrases for which at least one of the top-K predicted segments has $\mathrm{T\text -}IoU\ge t$. We report R@1 and R@5 at thresholds $t\in\{0.3,0.5,0.7\}$.  

Mean Alignment Error (MAE)  
Define the center of each segment $c = (t_1+t_2)/2$. Then  
$$
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^N\bigl|c_i-\hat c_i\bigr|
$$  
captures average temporal localization error in seconds.  

2.3 Baseline Models  
We will implement and evaluate:  
1) Sliding-Window + Embedding Similarity: extract uniform proposals, embed video segment and text via pretrained CLIP-video and text encoders, score by cosine similarity.  
2) Two-Tower Contrastive: dual encoders trained with InfoNCE loss  
$$
L = -\sum_{i=1}^B\log\frac{\exp(s(v_i,p_i)/\tau)}{\sum_{j=1}^B\exp(s(v_i,p_j)/\tau)}
$$  
where $s(\cdot,\cdot)$ is cosine similarity, $B$ batch size, $\tau$ temperature.  
3) Cross-Encoder Localization: use a transformer that concatenates frame features and phrase tokens, outputs frame-wise relevance scores, then apply dynamic programming to find best segment.  

2.4 Proposed Model: FineAlignNet  
Architecture Overview  
– Video Encoder: 3D CNN backbone (e.g., SlowFast) extracts per-frame features $F\in\mathbb{R}^{T\times D}$.  
– Phrase Encoder: Transformer or RoBERTa produces token embeddings $E\in\mathbb{R}^{L\times D}$.  
– Cross-Modal Attention: compute attention between each text token and every frame:  
$$
A = \mathrm{softmax}\Bigl(\frac{F W_f (E W_e)^\top}{\sqrt{D}}\Bigr)\quad\in\mathbb{R}^{T\times L}
$$  
where $W_f, W_e\in\mathbb{R}^{D\times D}$ are learned.  
– Frame Relevance Score: aggregate across tokens to get per-frame score $r_t = \max_{l}A_{t,l}$.  
– Segment Prediction: treat the sequence $\{r_t\}_{t=1}^T$ as a 1D signal, detect the contiguous interval maximizing the sum of scores under a length prior. Solve  
$$
\hat t_1,\hat t_2 = \arg\max_{1\le t_1<t_2\le T}\sum_{t=t_1}^{t_2}r_t - \lambda (t_2-t_1)
$$  
using dynamic programming.  

Training Objectives  
– Alignment Loss: for each positive pair $(v,p)$ and sampled negatives $p'$, minimize contrastive loss on predicted segment features pooled via mask.  
– Regression Loss: if ground truth segment is $[t_1,t_2]$, regress predicted boundaries:  
$$
L_{\text{reg}} = |\,\hat t_1 - t_1| + |\,\hat t_2 - t_2|.
$$  
Total loss: $L_{\text{total}} = L_{\text{contrastive}} + \alpha L_{\text{reg}}$.  

2.5 Experimental Design  
Data Splits & Protocol  
– Train on 1,400 videos, validate on 300, test on 300. Report metrics on test only once.  

Implementation Details  
– Framework: PyTorch, HuggingFace Transformers.  
– Hardware: 8 NVIDIA A100 GPUs.  
– Hyperparameters: batch size 32, learning rate $1e$-4 with cosine decay, $\alpha=0.5$. Train for 20 epochs.  

Ablation Studies  
– Impact of cross-modal attention vs. two-tower.  
– Effect of dynamic programming vs. simple thresholding for segment selection.  
– Sensitivity to $\lambda$ length penalty.  
– Using different video backbones (SlowFast vs. Swin3D).  

Statistical Analysis  
– Report standard deviations over three random seeds.  
– Conduct paired t-tests when comparing FineAlignNet to best baseline.  

3. Expected Outcomes & Impact  
Dataset & Benchmark Release  
We will publicly release FineActionBench—videos, phrase annotations, evaluation scripts, and baseline code—under an open license. This benchmark is expected to become a standard testbed for the community.  

Baseline Performance & Insights  
We anticipate:  
• Sliding-window and two-tower baselines will achieve modest R@1@0.5 (∼20–25%) but struggle at higher IoU.  
• Cross-encoder and FineAlignNet will close the gap (R@1@0.5 ∼45–55%), demonstrating the value of tight multi-modal fusion and boundary regression.  
• Ablations will reveal the critical role of dynamic programming segment selection and contrastive-regression joint training.  

Scientific Impact  
FineActionBench will:  
• Spur development of novel architectures for fine-grained temporal grounding.  
• Provide a clear roadmap for improvements—e.g., better proposal generation, transformer-based video encoders, multimodal self-supervision.  
• Encourage cross-pollination between the video-action-localization and video-language communities.  

Broader Societal Impact  
Fine-grained video-language understanding has immediate applications:  
• Interactive video editing tools that allow “find where I stir the sauce.”  
• Assistive robotics that correctly execute multi-step instructions at the right moment.  
• Intelligent surveillance and safety monitoring that triggers only on precise events.  
By benchmarking and advancing temporal grounding, we lay the groundwork for more reliable, context-aware smart systems.  

In summary, FineActionBench will fill a critical gap in video-language research, fostering the next generation of models capable of understanding not just what happens in a video, but exactly when each described event occurs.