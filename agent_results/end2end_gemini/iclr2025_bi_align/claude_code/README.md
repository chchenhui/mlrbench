# AI Cognitive Tutor Experiment

This project implements an experimental framework for testing the effectiveness of an AI Cognitive Tutor in improving human understanding of complex AI systems. The experiment is based on the bidirectional human-AI alignment framework, focusing specifically on the "Aligning Humans with AI" perspective.

## Project Overview

The AI Cognitive Tutor is designed to detect when users misunderstand an AI system and provide adaptive explanations to improve their understanding. This experiment simulates an interaction between medical professionals and an AI diagnostic system, comparing a treatment group (with the AI Cognitive Tutor) to a control group (with standard explanations).

## Repository Structure

- `main.py`: Main script for running the experiment
- `config.yaml`: Configuration file for experiment parameters
- `models/`: Module containing implementation of the AI systems
  - `ai_diagnostic.py`: Simulated AI diagnostic system
  - `cognitive_tutor.py`: AI Cognitive Tutor implementation
  - `baselines.py`: Baseline explanation methods for comparison
- `simulation/`: Module for simulating the experiment
  - `participant.py`: Simulated participant behavior
  - `experiment.py`: Experimental workflow
- `evaluation/`: Module for evaluating results
  - `metrics.py`: Statistical analysis of results
- `visualization/`: Module for generating visualizations
  - `visualizer.py`: Visualization of experiment results
- `reports/`: Module for generating reports
  - `report_generator.py`: Generates results.md report

## Requirements

Required Python packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Running the Experiment

To run the experiment with default parameters:

```bash
python main.py
```

Custom configuration:

```bash
python main.py --config custom_config.yaml --results_dir ../results
```

## Configuration Options

The experiment can be configured via the `config.yaml` file:

- **Experiment settings**: Number of trials, participants, etc.
- **AI Diagnostic System**: Accuracy, uncertainty levels, explanation types
- **Participant settings**: Expertise distribution
- **Misunderstanding triggers**: Types of behavior that trigger interventions
- **AI Cognitive Tutor settings**: Tutoring strategies, activation threshold
- **Baseline methods**: Alternative explanation approaches for comparison
- **Evaluation metrics**: Which metrics to calculate

## Results

The experiment generates several outputs:

1. **Raw data** (JSON and CSV files)
2. **Visualizations** (PNG files)
3. **Statistical analysis** (included in results.md)
4. **Summary report** (results.md)

The main results are saved to the specified results directory (`../results` by default).

## Extending the Experiment

To extend or modify this experiment:

1. **New tutoring strategies**: Add them to the AI Cognitive Tutor in `models/cognitive_tutor.py`
2. **Different baselines**: Add new baseline methods in `models/baselines.py`
3. **Custom metrics**: Implement new evaluation metrics in `evaluation/metrics.py`
4. **Additional visualizations**: Add new visualization functions in `visualization/visualizer.py`

## Citation

If you use this code in your research, please cite:

```
@article{aicognitivetutor2025,
  title={Fostering Human Cognitive Alignment in Complex AI Partnerships: An Adaptive AI Tutoring Framework},
  author={AI Cognitive Tutor Team},
  journal={ICLR Workshop on Bidirectional Human-AI Alignment},
  year={2025}
}
```