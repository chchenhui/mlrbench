from datasets import load_dataset

def get_human_eval_subset(n_problems: int = 5):
    """
    Loads the HumanEval dataset and returns a subset of problems.
    
    Args:
        n_problems: The number of problems to select.

    Returns:
        A list of problems.
    """
    dataset = load_dataset("openai_humaneval", split="test")
    
    # Select a subset of problems that are interesting for verification
    # For example, problems with clear mathematical or logical properties.
    # Here we select a few manually.
    # 0: has_close_elements (float comparison)
    # 2: separate_paren_groups (string manipulation)
    # 3: truncate_number (float and integer manipulation)
    # 4: below_zero (list of integers)
    # 5: mean_absolute_deviation (list of floats)
    problem_indices = [0, 2, 3, 4, 5]
    
    if n_problems > len(problem_indices):
        raise ValueError(f"Requested {n_problems} problems, but only {len(problem_indices)} are available in the curated subset.")

    selected_problems = [dataset[i] for i in problem_indices[:n_problems]]
    return selected_problems

if __name__ == '__main__':
    problems = get_human_eval_subset(2)
    print(f"Loaded {len(problems)} problems.")
    for i, problem in enumerate(problems):
        print(f"--- Problem {i} ---")
        print(f"Task ID: {problem['task_id']}")
        print(f"Prompt: {problem['prompt']}")
        print(f"Entry Point: {problem['entry_point']}")
        print(f"Test Cases: {problem['test'][:2]}") # Print first 2 test cases
        print("-" * (len(f"--- Problem {i} ---")))
