# FineActionBench: A Benchmark for Fine-Grained Temporal Video-Language Alignment

## Introduction

The proliferation of video content across various platforms has spurred significant advances in video-language models. These models seek to bridge the semantic gap between visual temporal information and natural language descriptions. While notable progress has been made in video-language tasks such as retrieval, captioning, and question answering, a critical capability remains underdeveloped: fine-grained temporal alignment between specific textual phrases and precise moments within videos.

Current video-language benchmarks predominantly focus on coarse-grained understanding, where models match entire videos with complete captions or answer questions about general video content. However, many real-world applications demand much more precise temporal understanding. Consider assistive technologies that need to identify exactly when a requested action occurs in a video, content moderation systems that must pinpoint problematic segments, or instructional applications that need to align specific verbal instructions with their visual demonstrations. These scenarios require models to perform fine-grained temporal grounding – associating specific textual phrases with exact temporal segments within videos.

Existing benchmarks have made initial progress in this direction. TemporalBench (Cai et al., 2024) evaluates temporal understanding through question-answering, while VideoComp (Kim et al., 2025) focuses on compositional understanding with temporal disruptions. FIBER (Xu et al., 2024) addresses text-to-video retrieval with spatial and temporal annotations. However, these benchmarks either evaluate temporal understanding indirectly (through QA) or prioritize other aspects of video-language understanding, leaving a gap in comprehensive evaluation of fine-grained temporal alignment capabilities.

The lack of a dedicated benchmark focusing specifically on fine-grained temporal alignment between language and video presents several challenges:

1. It hinders the development of models capable of precise temporal grounding, as there is no standardized way to measure progress in this specific capability.
2. It makes it difficult to compare different approaches systematically, as existing metrics often fail to capture the nuances of temporal precision.
3. Without appropriate evaluation tools, researchers cannot effectively diagnose where and why models fail at fine-grained temporal alignment tasks.

To address these challenges, we propose FineActionBench, a novel benchmark specifically designed to evaluate the fine-grained temporal alignment capabilities of video-language models. FineActionBench focuses on complex, multi-step activities where precise temporal localization is critical for understanding. By providing dense annotations where short textual phrases are precisely mapped to specific temporal segments, along with tailored evaluation metrics, our benchmark will enable researchers to measure and improve the ability of models to ground language in video with temporal precision.

The significance of this research extends beyond academic interest. As video-language models are increasingly deployed in real-world applications requiring precise temporal understanding – from video search and content moderation to assistive technologies and robotics – a benchmark focused specifically on fine-grained temporal alignment will drive progress in this crucial capability, ultimately leading to more useful and effective video-language systems.

## Methodology

### 1. Dataset Creation and Curation

#### 1.1 Video Collection

We will curate a diverse collection of 5,000 videos depicting complex, multi-step activities across five domains:
- Cooking and food preparation (1,000 videos)
- DIY and craft tutorials (1,000 videos)
- Assembly and repair instructions (1,000 videos)
- Sports and physical activities (1,000 videos)
- Daily household activities (1,000 videos)

Videos will be selected based on the following criteria:
- Duration: 2-10 minutes to ensure sufficient complexity while remaining manageable
- Resolution: Minimum 720p to ensure visual clarity
- Content: Contains distinct, identifiable sub-actions that form part of a larger procedure
- Diversity: Variety in settings, actors, camera angles, and lighting conditions
- License: Videos with permissive licenses (Creative Commons or similar) to enable research use

We will source videos from existing datasets (like FineAction by Liu et al., 2021) and supplement with additional content from public platforms with appropriate licensing.

#### 1.2 Annotation Process

We will employ a multi-stage annotation process to create dense, high-quality temporal-textual alignments:

**Stage 1: Temporal Segmentation**
- Expert annotators will identify distinct meaningful segments within each video
- Each segment corresponds to a well-defined sub-action or event
- Annotators will mark precise start and end timestamps for each segment
- For each video, we target an average of 15-25 segments, resulting in ~100,000 total segments

**Stage 2: Textual Description**
- For each identified segment, annotators will provide:
  * A concise phrase describing the specific action/event (5-15 words)
  * Relevant objects and their attributes involved in the action
  * Spatial information when relevant (e.g., "cutting vegetables on the left side of the cutting board")

**Stage 3: Relationship Annotation**
- Annotators will identify and label temporal relationships between segments:
  * Sequential relationships (action A followed by action B)
  * Causal relationships (action A enables action B)
  * Parent-child relationships (action A contains sub-actions B and C)

**Stage 4: Verification and Refinement**
- A separate group of annotators will verify the accuracy of temporal boundaries and textual descriptions
- Annotations with <85% agreement will be reviewed and refined
- Verification process will ensure consistent annotation quality across the dataset

#### 1.3 Dataset Organization

The resulting dataset will be structured as follows:

```
{
  "video_id": "v001",
  "url": "https://example.com/video/v001",
  "domain": "cooking",
  "duration": 325.5,  # in seconds
  "segments": [
    {
      "segment_id": "v001_s001",
      "start_time": 15.2,  # in seconds
      "end_time": 23.8,
      "description": "slicing tomatoes into thin rounds",
      "objects": ["tomato", "knife", "cutting board"],
      "relationships": [
        {"type": "follows", "segment_id": "v001_s000"},
        {"type": "enables", "segment_id": "v001_s002"}
      ]
    },
    // Additional segments...
  ]
}
```

### 2. Benchmark Tasks and Evaluation

#### 2.1 Core Tasks

FineActionBench will include the following tasks to evaluate fine-grained temporal alignment:

**Task 1: Phrase-to-Segment Localization**
- Input: Video and a textual phrase
- Output: Predicted temporal boundaries (start and end times)
- Example: Given "whisking eggs in a metal bowl" and a cooking video, identify the exact segment where this action occurs

**Task 2: Dense Video Captioning with Temporal Grounding**
- Input: Video only
- Output: Set of (description, start time, end time) tuples covering the key actions in the video
- Evaluation: Both caption quality and temporal alignment accuracy

**Task 3: Temporal Relationship Reasoning**
- Input: Video and a query about temporal relationships (e.g., "What happens immediately after the person cuts the vegetables?")
- Output: Textual answer and temporal boundaries of the relevant segments
- Focus: Testing understanding of sequential, causal, and hierarchical relationships

**Task 4: Moment Retrieval with Natural Language Queries**
- Input: Natural language query describing a specific moment or action
- Output: Ranked list of temporal segments across all videos in the test set
- Example: "Person adding salt to boiling water" should retrieve all matching segments

#### 2.2 Evaluation Metrics

We propose the following metrics to evaluate fine-grained temporal alignment capabilities:

**1. Phrase-localized Temporal Intersection over Union (PT-IoU)**

The PT-IoU measures the temporal overlap between predicted and ground truth segments while considering the relevance of the textual description:

$$PT\text{-}IoU(p, g) = IoU(p, g) \cdot S(d_p, d_g)$$

Where:
- $p$ is the predicted segment with start time $p_s$, end time $p_e$, and description $d_p$
- $g$ is the ground truth segment with start time $g_s$, end time $g_e$, and description $d_g$
- $IoU(p, g) = \frac{max(0, min(p_e, g_e) - max(p_s, g_s))}{max(p_e, g_e) - min(p_s, g_s)}$
- $S(d_p, d_g)$ is the semantic similarity between descriptions, calculated using a pretrained text embedding model

**2. Temporally-Weighted Recall (TWR)**

TWR evaluates how well a model captures all ground truth segments, with penalties for missing temporally significant segments:

$$TWR = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \mathbf{1}[PT\text{-}IoU(p_i, g_i) > \theta]$$

Where:
- $N$ is the total number of ground truth segments
- $w_i$ is the temporal weight of segment $i$ (proportional to its duration)
- $\mathbf{1}[\cdot]$ is an indicator function
- $\theta$ is a threshold (typically 0.5)

**3. Temporal Action Precision (TAP)**

TAP measures the precision of predicted segments at different overlap thresholds:

$$TAP@\theta = \frac{\sum_{i=1}^{M} \mathbf{1}[\max_j PT\text{-}IoU(p_i, g_j) > \theta]}{M}$$

Where:
- $M$ is the total number of predicted segments
- $\theta$ is a threshold (commonly evaluated at 0.3, 0.5, and 0.7)

**4. Mean Segment Distance (MSD)**

MSD quantifies the average temporal distance between predicted and ground truth segments:

$$MSD = \frac{1}{N} \sum_{i=1}^{N} \min_j D(g_i, p_j)$$

Where:
- $D(g_i, p_j) = max(|g_{i,s} - p_{j,s}|, |g_{i,e} - p_{j,e}|)$ is the maximum boundary distance
- Lower values indicate better alignment

**5. Relationship Accuracy (RA)**

RA evaluates the model's understanding of temporal relationships between segments:

$$RA = \frac{1}{R} \sum_{i=1}^{R} \mathbf{1}[r_i^{pred} = r_i^{gt}]$$

Where:
- $R$ is the total number of relationships in the ground truth
- $r_i^{pred}$ is the predicted relationship type
- $r_i^{gt}$ is the ground truth relationship type

### 3. Benchmark Platform and Tools

To facilitate widespread use and standardized evaluation, we will develop:

1. A public leaderboard for tracking progress on all benchmark tasks
2. Evaluation scripts implementing all proposed metrics
3. Baseline implementations of current state-of-the-art approaches
4. Data loading and preprocessing utilities for common deep learning frameworks
5. Visualization tools to help researchers understand model performance and failure cases

### 4. Baseline Methods

We will implement and evaluate the following baseline approaches:

**1. Two-Stage Approach**
- First stage: Video segment proposal generation using sliding windows or a trained proposal network
- Second stage: Cross-modal matching between proposals and textual descriptions

**2. End-to-End Video-Language Transformer**
- Vision Transformer backbone for video encoding
- Text Transformer for language encoding
- Cross-attention mechanism for alignment
- Multi-task training across all benchmark tasks

**3. Fine-tuned Large Video-Language Models**
- Adapt existing pretrained large video-language models (e.g., VideoLLAMA, Video-LLaVA)
- Fine-tune on our dataset with task-specific objectives
- Evaluate zero-shot, few-shot, and fully fine-tuned performance

**4. Temporal Graph Reasoning Approach**
- Construct a graph representation of video segments
- Model temporal relationships explicitly as graph edges
- Apply graph neural networks to reason about relationships
- Joint optimization of segment localization and relationship prediction

### 5. Experimental Design

We will conduct comprehensive experiments to evaluate and compare different approaches:

1. **Data Split**: 
   - Training set: 3,500 videos (70%)
   - Validation set: 500 videos (10%)
   - Test set: 1,000 videos (20%)
   - Ensure balanced domain distribution across splits

2. **Ablation Studies**:
   - Impact of temporal resolution (frame rate) on performance
   - Effect of video duration on model accuracy
   - Contribution of relationship annotations to overall performance
   - Importance of multi-task training vs. single-task specialization

3. **Human Performance Baseline**:
   - Recruit human annotators to perform the same tasks
   - Establish human-level performance as an upper bound
   - Analyze the gap between model and human performance

4. **Error Analysis**:
   - Categorize and quantify different types of errors (temporal boundary errors, misclassifications, etc.)
   - Identify challenging cases and common failure modes
   - Analyze performance across different domains and action types

## Expected Outcomes & Impact

### 1. Direct Research Contributions

The successful completion of this research will yield several concrete contributions:

1. **FineActionBench Dataset**: A large-scale, densely annotated dataset of 5,000 videos with approximately 100,000 segment-level annotations, providing unprecedented resources for training and evaluating fine-grained temporal video-language alignment.

2. **Novel Evaluation Metrics**: A suite of specialized metrics (PT-IoU, TWR, TAP, MSD, and RA) designed specifically to evaluate fine-grained temporal alignment capabilities, offering more nuanced and informative assessment than existing metrics.

3. **Benchmark Platform**: A comprehensive evaluation framework including standardized tasks, evaluation protocols, baseline implementations, and a public leaderboard, enabling fair and consistent comparison of different approaches.

4. **Performance Analysis**: Detailed comparative analysis of different modeling approaches, highlighting strengths, weaknesses, and promising research directions for improving fine-grained temporal video-language alignment.

### 2. Broader Research Impact

Beyond these direct contributions, FineActionBench will have broader impacts on video-language research:

1. **Guiding Model Development**: By providing clear evaluation criteria for fine-grained temporal alignment, FineActionBench will guide the development of more temporally-aware video-language models, addressing a critical capability gap in current systems.

2. **Standardizing Evaluation**: The benchmark will establish standardized evaluation protocols for temporal grounding tasks, improving research reproducibility and enabling meaningful comparison across approaches.

3. **Identifying Research Challenges**: Comprehensive performance analysis will highlight specific challenges and limitations in current approaches, stimulating targeted research to address these gaps.

4. **Encouraging Multidisciplinary Collaboration**: The benchmark will bring together researchers from computer vision, natural language processing, and machine learning to tackle the inherently multidisciplinary challenge of fine-grained video-language alignment.

### 3. Practical Applications

Improvements in fine-grained temporal video-language alignment will enable numerous practical applications:

1. **Enhanced Video Search and Retrieval**: Enabling users to find precise moments in videos matching specific textual queries, significantly improving video search utility.

2. **Intelligent Video Editing and Summarization**: Automating the process of identifying and extracting key moments from longer videos based on textual descriptions or queries.

3. **Detailed Content Moderation**: Helping content moderation systems precisely identify problematic segments within videos, improving moderation efficiency and accuracy.

4. **Advanced Instructional Systems**: Enhancing tutorial and educational platforms by allowing precise alignment between verbal instructions and visual demonstrations.

5. **Video-Based Assistive Technologies**: Improving accessibility tools that help users with visual impairments understand video content through precise narration of visual events.

6. **Robotics and Embodied AI**: Enabling robots and embodied agents to better understand and follow sequential instructions by precisely aligning language with observed or demonstrated actions.

### 4. Long-Term Vision

In the longer term, FineActionBench aims to stimulate progress toward video-language models that truly understand the rich temporal dynamics of human activities. Current models often treat videos as collections of loosely connected frames, failing to capture the intricate temporal relationships that define meaningful actions and events. By focusing specifically on fine-grained temporal alignment, our benchmark will push the field toward models that can:

1. Parse continuous video streams into semantically meaningful temporal segments
2. Ground natural language descriptions at precisely the right moments in time
3. Understand the causal and sequential relationships between actions
4. Reason about hierarchical action structures (actions composed of sub-actions)

These capabilities represent crucial steps toward artificial intelligence systems that can interpret and interact with the temporal world in human-like ways, with applications ranging from assistive technologies to autonomous systems.

Through FineActionBench, we aim to close the gap between the current state of video-language models and the fine-grained temporal understanding capabilities required for truly useful and versatile video-language applications.