# Dataset Directory

This directory is used to store datasets for the Adaptive Code Assistant experiments.

## Dataset Options

The experiment can use one of the following data sources:

1. **Local Dataset**: Place a `coding_tasks.json` file in this directory with the following format:
   ```json
   [
     {
       "task_id": "task_1",
       "context": "def factorial(n):\n    # Calculate the factorial of n\n    ",
       "solution": "def factorial(n):\n    # Calculate the factorial of n\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n",
       "description": "Calculate factorial",
       "tags": ["python", "recursive"]
     },
     ...
   ]
   ```

2. **HuggingFace Dataset**: If no local dataset is found, the code will automatically download and use the OpenAI HumanEval dataset from Hugging Face.

3. **Synthetic Dataset**: If neither of the above options are available, the experiment will generate a synthetic dataset with simple coding tasks. This is primarily for testing the pipeline.

## Generating a Dataset

You can generate a synthetic dataset for testing by running:

```bash
python -c "from utils.data_utils import _generate_synthetic_dataset; dataset = _generate_synthetic_dataset(100); dataset.save_to_disk('./data/synthetic')"
```

## Dataset Format

Each dataset entry should contain the following fields:

- `task_id`: A unique identifier for the task
- `context`: The code context (the prompt given to the model)
- `solution`: The expected or reference solution
- `description` (optional): A description of the task
- `tags` (optional): A list of tags categorizing the task