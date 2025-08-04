
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.trace_generator import trace_execution

def evaluate_model(model, tokenizer, dataset, results_dir="results"):
    """
    Evaluates a model on a given dataset and saves the results.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results = []
    for example in dataset:
        prompt = example['prompt']
        test = example['test']
        
        # Generate code from the model
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Note: This is a simplified generation process. For better results, 
        # more sophisticated generation techniques should be used.
        outputs = model.generate(inputs.input_ids, max_new_tokens=100)
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Execute the generated code
        execution_result = trace_execution(generated_code, test)
        
        results.append({
            "prompt": prompt,
            "generated_code": generated_code,
            "test": test,
            "status": execution_result['status'],
            "trace": execution_result['trace']
        })

    # Save results to a JSON file
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

def plot_results(results, results_dir="results"):
    """
    Generates and saves plots for the evaluation results.
    """
    df = pd.DataFrame(results)
    
    # Plot pass/fail ratio
    pass_fail_counts = df['status'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Pass/Fail Ratio')
    plt.savefig(os.path.join(results_dir, 'pass_fail_ratio.png'))
    plt.close()

if __name__ == '__main__':
    from src.models import get_base_model
    from src.data_loader import get_human_eval_subset

    model, tokenizer = get_base_model()
    
    human_eval_cache_dir = "data/human_eval"
    dataset = get_human_eval_subset(human_eval_cache_dir, subset_size=10)
    
    evaluation_results = evaluate_model(model, tokenizer, dataset)
    plot_results(evaluation_results)
    
    print("Evaluation complete. Results and plots saved in the 'results' directory.")
