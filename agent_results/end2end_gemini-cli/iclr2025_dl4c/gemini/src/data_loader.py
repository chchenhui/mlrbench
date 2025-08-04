
import os
from datasets import load_dataset

def load_data(cache_dir):
    """
    Loads the HumanEval dataset from Hugging Face.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    dataset = load_dataset("openai_humaneval", cache_dir=cache_dir)
    return dataset

def get_human_eval_subset(cache_dir, subset_size=50):
    """
    Loads a subset of the HumanEval dataset.
    """
    dataset = load_data(cache_dir)
    return dataset['test'].select(range(subset_size))

if __name__ == '__main__':
    human_eval_cache_dir = "data/human_eval"
    human_eval_subset = get_human_eval_subset(human_eval_cache_dir)
    print(human_eval_subset)
    print(f"Loaded {len(human_eval_subset)} examples from the HumanEval dataset.")
