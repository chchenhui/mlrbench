import json
import os.path as osp

from mlrbench.utils.utils import *

MODEL_COST_PER_TOKEN = {
    "gpt-4o-2024-08-06": {"in": 0.0000025, "out": 0.00001},
    "gpt-4o": {"in": 0.0000025, "out": 0.00001},
    "gpt-4o-mini": {"in": 0.00000015, "out": 0.0000006},
    "gpt-4o-mini-2024-07-18": {"in": 0.00000015, "out": 0.0000006},
    "gpt-4o-search-preview-2025-03-11": {"in": 0.0000025, "out": 0.00001},
    "o1": {"in": 0.000015, "out": 0.00006},
    "o1-2024-12-17": {"in": 0.000015, "out": 0.00006},
    "o1-mini": {"in": 0.0000011, "out": 0.0000044},
    "o1-mini-2024-09-12": {"in": 0.0000011, "out": 0.0000044},
    "o3-mini": {"in": 0.0000011, "out": 0.0000044},
    "o3-mini-2025-01-31": {"in": 0.0000011, "out": 0.0000044},
    "o4-mini-2025-04-16": {"in": 0.0000011, "out": 0.0000044},
    "gemini-2.5-pro-preview": {"in": 0.00000125, "out": 0.00001},
    "claude-3-7-sonnet-20250219": {"in": 0.000003, "out": 0.000015, "cache_write": 0.00000375, "cache_read": 0.0000003},
}


AGENT_TO_MODEL = {
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.5-pro-preview",
    "o4-mini": "o4-mini-2025-04-16",
    "4o-search": "gpt-4o-search-preview-2025-03-11",
}


def compute_cost(model_name, token_usage, model_cost_per_token) -> float:
    """
    Computes the total cost of using a model given token usage and cost per token
    """
    total_cost = 0
    if len(token_usage.keys()) == 3: # OPENAI 
        in_toks_cost = token_usage["prompt_tokens"] * model_cost_per_token[model_name]["in"]
        out_toks_cost = token_usage["completion_tokens"] * model_cost_per_token[model_name]["out"]
        total_cost = in_toks_cost + out_toks_cost
    elif len(token_usage.keys()) == 4: # ANTHROPIC
        in_toks_cost = token_usage["input_tokens"] * model_cost_per_token[model_name]["in"]
        out_toks_cost = token_usage["output_tokens"] * model_cost_per_token[model_name]["out"]
        cache_write_cost = token_usage["cache_creation_input_tokens"] * model_cost_per_token[model_name]["cache_write"]
        cache_read_cost = token_usage["cache_read_input_tokens"] * model_cost_per_token[model_name]["cache_read"]
        total_cost = in_toks_cost + out_toks_cost + cache_write_cost + cache_read_cost
    return total_cost


def compute_cost_for_pipeline(agent_name):
    model_name = AGENT_TO_MODEL[agent_name]
    lit_model_name = "gpt-4o-search-preview-2025-03-11"
    tasklist = get_tasklist(f"pipeline_{agent_name}")
    total = 0
    for task_name in tasklist:
        task_path = osp.join(f"pipeline_{agent_name}", task_name)
        # Load the token usage
        total_cost = 0
        writing_step = ["idea", "proposal", "paper"]
        for step in writing_step:
            token_usage_file = osp.join(task_path, f"{step}_token_usage.json")
            with open(token_usage_file, "r") as f:
                token_usage = json.load(f)
                cost = compute_cost(model_name, token_usage, MODEL_COST_PER_TOKEN)
                # print(f"Cost for {step} in {task_name}: {cost}")
                total_cost += cost
        lit_usage_file = osp.join(task_path, f"lit_token_usage.json")
        with open(lit_usage_file, "r") as f:
            token_usage = json.load(f)
            cost = compute_cost(lit_model_name, token_usage, MODEL_COST_PER_TOKEN)
            # print(f"Cost for literature review in {task_name}: {cost}")
            total_cost += cost
        coding_file = osp.join(task_path, "claude_output.json")
        with open(coding_file, "r") as f:
            token_usage = json.load(f)
            for item in token_usage:
                if item['role'] == 'system':
                    # print(f"Cost for coding in {task_name}: {item['cost_usd']}")
                    total_cost += item['cost_usd']
        # Compute the cost
        # print(f"Total cost for {task_name}: {total_cost}")
        total += total_cost
    print(f"Average cost per task in pipeline_{agent_name}: {total/len(tasklist)}")


if __name__ == "__main__":
    compute_cost_for_pipeline("o4-mini")
    compute_cost_for_pipeline("gemini")
    compute_cost_for_pipeline("claude")