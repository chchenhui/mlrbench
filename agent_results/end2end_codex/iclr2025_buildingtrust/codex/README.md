# Experimental Pipeline for Influence-Driven Selective Unlearning

This folder contains scripts to run a simplified experimental pipeline on the SST2 dataset, demonstrating a baseline retraining method and a gradient-projection unlearning method to remove a canary example.

Requirements
------------
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

Running the experiment
----------------------
Execute the main script:
```bash
python run_experiment.py
```

Outputs
-------
- `results.csv`: Contains accuracy and canary probability for each method.
- `loss_curves.png`: Training and validation loss curves.
- `canary_prob.png`: Bar chart of canary probability by method.
- `log.txt`: Logs of the experiment execution.

The results and figures can be found in the project root after completion.
