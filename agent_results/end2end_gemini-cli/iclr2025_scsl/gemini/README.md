# SCA-Adapter Experiment

This project implements and evaluates the Spurious-Correlation-Aware Adapter (SCA-Adapter) method as described in the research proposal.

## Overview

The experiment tests the hypothesis that SCA-Adapters can improve the robustness of foundation models to spurious correlations in a parameter-efficient way. It compares the SCA-Adapter method against three baselines:
1.  **Zero-Shot:** The pre-trained CLIP model without any fine-tuning.
2.  **LoRA:** Standard parameter-efficient fine-tuning using LoRA.
3.  **SCA-Adapter:** The proposed method, which uses orthogonal gradient projection to disentangle task and spurious feature learning.

The experiment is run on a simulated version of the Waterbirds dataset, created from the CUB_200_2011 dataset.

## Requirements

You will need Python 3.8+ and the following packages:
- `torch`
- `torchvision`
- `transformers`
- `datasets`
- `pandas`
- `numpy`
- `matplotlib`
- `tqdm`
- `peft`
- `captum`
- `Pillow`

You can install them using pip:
```bash
pip install torch torchvision transformers datasets pandas numpy matplotlib tqdm peft captum Pillow
```

## How to Run

The entire experiment is automated by the `run_experiment.py` script.

To run the experiment, navigate to the `gemini` directory and execute the script:

```bash
cd gemini
python run_experiment.py
```

The script will perform the following steps:
1.  Download the required model and dataset from Hugging Face.
2.  Run the Zero-Shot, LoRA, and SCA-Adapter experiments in sequence.
3.  Log the entire process to `../results/log.txt`.
4.  Generate comparison plots for key metrics (`wga_comparison.png`, `avg_acc_comparison.png`, `params_comparison.png`) and save them in the `../results` directory.
5.  Generate a summary report `results.md` in the `../results` directory.

## Output Structure

The final results will be organized as follows:

```
.
├── gemini/
│   ├── run_experiment.py
│   └── outputs/
│       └── results.json
└── results/
    ├── log.txt
    ├── results.md
    ├── wga_comparison.png
    ├── avg_acc_comparison.png
    └── params_comparison.png
```
