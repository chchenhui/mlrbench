# Neuro-Symbolic Repair Experiment

This project implements and evaluates the SMT-Repair framework, a method for correcting LLM-generated code using feedback from an SMT solver.

## Overview

The core idea is to leverage the precision of formal verification to guide the self-correction capabilities of Large Language Models (LLMs). The process is as follows:

1.  An LLM generates Python code for a given problem.
2.  A symbolic execution tool (`crosshair`) powered by an SMT solver (Z3) verifies the code against its type hints and inferred contracts (e.g., from docstrings).
3.  If a counterexample (a bug) is found, it's translated into a natural language prompt.
4.  This prompt is fed back to the LLM, which attempts to correct the code.
5.  This loop continues until the code is verified or a maximum number of iterations is reached.

This project compares the effectiveness of this **SMT-Repair** method against two baselines:
-   **Zero-Shot**: The LLM's initial code without any correction.
-   **UT-Repair**: A self-correction loop guided by feedback from failing unit tests.

## How to Run

1.  **Setup the Environment:**
    First, create a virtual environment and install the required packages.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Set API Keys:**
    This experiment uses the OpenAI API. Make sure your API key is available as an environment variable:
    ```bash
    export OPENAI_API_KEY='your_api_key_here'
    ```

3.  **Run the Experiment:**
    The main experiment script will run all methods, save the results, and generate figures. The output, including logs, results, and figures, will be saved in the `results` directory.
    ```bash
    python scripts/run_experiment.py
    ```
