# Proactive Routing in Mixture-of-Experts for Zero-Shot Task Adaptation

This project implements and evaluates the Proactive Router (PRo-MoE), a novel approach for enabling zero-shot task adaptation in Mixture-of-Experts (MoE) models.

## Experimental Plan

The experiment is designed to test the hypothesis that a Proactive Router can dynamically configure an MoE model's routing policy for unseen tasks, leading to superior zero-shot performance compared to static routing and competitive performance with few-shot fine-tuning methods.

### 1. Models

*   **PRo-MoE (Proposed)**: An MoE model equipped with a Proactive Router that generates task-specific routing parameters from a natural language task description.
*   **Standard MoE (Baseline)**: A standard MoE model with a static gating network.
*   **Dense Model (Baseline)**: A dense transformer with a similar parameter count to the activated parameters of the MoE models.

### 2. Dataset

The experiment will use the **Super-NaturalInstructions** dataset, a large and diverse collection of instructional tasks, perfect for the meta-learning framework. A subset of tasks will be held out for evaluation to test zero-shot generalization.

### 3. Procedure

1.  **Meta-Training**: The models are trained on a diverse set of tasks from the Super-NaturalInstructions dataset. The PRo-MoE model learns to map task descriptions to routing policies.
2.  **Zero-Shot Evaluation**: The trained models are evaluated on a held-out set of unseen tasks.
3.  **Analysis**: Performance is measured using task-specific metrics (e.g., ROUGE for summarization). Expert utilization patterns are analyzed to understand the effect of the Proactive Router.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the experiment:**
    ```bash
    python run_experiment.py
    ```

This will execute the full experimental pipeline, including data preparation, model training, evaluation, and result visualization. The final results and figures will be saved in the `results` directory.
