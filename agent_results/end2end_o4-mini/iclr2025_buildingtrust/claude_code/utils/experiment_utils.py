"""
Utility functions for experiments.
"""

import os
import json
import numpy as np
import torch
import random
import logging
from tqdm import tqdm
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir, experiment_name=None):
    """
    Set up logging for experiments.
    
    Args:
        log_dir (str): Directory to save logs
        experiment_name (str, optional): Name of the experiment
        
    Returns:
        logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("unlearning_experiment")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
        
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_json(data, file_path):
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        file_path (str): Path to save the data
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Define a custom JSON encoder to handle numpy and torch types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif hasattr(obj, "to_json"):
                return obj.to_json()
            return super(NumpyEncoder, self).default(obj)
    
    # Save the data
    with open(file_path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_json(file_path):
    """
    Load data from JSON.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        data: Loaded data
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data


def create_results_summary(results, output_file, include_details=True):
    """
    Create a summary of results in Markdown format.
    
    Args:
        results (dict): Results dictionary
        output_file (str): Path to save the summary
        include_details (bool): Whether to include detailed results
    """
    # Create the summary
    summary = "# Unlearning Experiment Results Summary\n\n"
    
    # Add date and time
    summary += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add method comparison results
    if "method_comparison" in results:
        summary += "## Method Comparison\n\n"
        summary += "### Performance Metrics\n\n"
        
        # Create table
        summary += "| Method | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |\n"
        summary += "|--------|--------|--------|--------------|----------------|\n"
        
        for method, metrics in results["method_comparison"].items():
            if method == "original_model":
                continue
                
            summary += f"| {method} | {metrics.get('KFR', 'N/A'):.4f} | {metrics.get('KRR', 'N/A'):.4f} | {metrics.get('perplexity', 'N/A'):.4f} | {metrics.get('compute_time', 'N/A'):.2f} |\n"
        
        # Original model reference
        if "original_model" in results["method_comparison"]:
            orig_perplexity = results["method_comparison"]["original_model"]["perplexity"]
            summary += f"\n**Original Model Perplexity:** {orig_perplexity:.4f}\n\n"
    
    # Add sequential unlearning results
    if "sequential_results" in results:
        summary += "## Sequential Unlearning\n\n"
        
        for method, seq_results in results["sequential_results"].items():
            summary += f"### {method}\n\n"
            
            # Create table
            summary += "| Request | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |\n"
            summary += "|---------|--------|--------|--------------|----------------|\n"
            
            for res in seq_results:
                summary += f"| {res.get('request_idx', 'N/A')} | {res.get('KFR', 'N/A'):.4f} | {res.get('KRR', 'N/A'):.4f} | {res.get('perplexity', 'N/A'):.4f} | {res.get('compute_time', 'N/A'):.2f} |\n"
            
            summary += "\n"
    
    # Add deletion size impact results
    if "deletion_size_impact" in results:
        summary += "## Deletion Set Size Impact\n\n"
        
        methods = list(next(iter(results["deletion_size_impact"].values())).keys())
        
        for method in methods:
            summary += f"### {method}\n\n"
            
            # Create table
            summary += "| Size | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |\n"
            summary += "|------|--------|--------|--------------|----------------|\n"
            
            for size, size_results in sorted(results["deletion_size_impact"].items(), key=lambda x: int(x[0])):
                if method in size_results:
                    res = size_results[method]
                    summary += f"| {size} | {res.get('KFR', 'N/A'):.4f} | {res.get('KRR', 'N/A'):.4f} | {res.get('perplexity', 'N/A'):.4f} | {res.get('compute_time', 'N/A'):.2f} |\n"
            
            summary += "\n"
    
    # Add visualization references
    summary += "## Visualizations\n\n"
    
    if "method_comparison" in results:
        summary += "### Performance Comparison\n\n"
        summary += "![Perplexity Comparison](./visualizations/perplexity_comparison.png)\n\n"
        summary += "![Knowledge Retention vs Forgetting](./visualizations/knowledge_retention_vs_forgetting.png)\n\n"
        summary += "![Computational Efficiency](./visualizations/computational_efficiency.png)\n\n"
        summary += "![Metrics Radar](./visualizations/metrics_radar.png)\n\n"
    
    if "sequential_results" in results:
        summary += "### Sequential Unlearning\n\n"
        summary += "![Sequential Unlearning](./visualizations/sequential_unlearning.png)\n\n"
    
    if "deletion_size_impact" in results:
        summary += "### Deletion Set Size Impact\n\n"
        summary += "![Deletion Size Impact (KFR)](./visualizations/deletion_size_impact_KFR.png)\n\n"
        summary += "![Deletion Size Impact (KRR)](./visualizations/deletion_size_impact_KRR.png)\n\n"
        summary += "![Deletion Size Impact (compute_time)](./visualizations/deletion_size_impact_compute_time.png)\n\n"
    
    # Include detailed results if requested
    if include_details:
        summary += "## Detailed Results\n\n"
        summary += "See the [results.json](./results.json) file for detailed results.\n\n"
    
    # Add conclusions
    summary += "## Conclusions\n\n"
    
    # Analysis of Cluster-Driven method (if available)
    if "method_comparison" in results and "cluster_driven" in results["method_comparison"]:
        cd_metrics = results["method_comparison"]["cluster_driven"]
        
        summary += "### Cluster-Driven Certified Unlearning\n\n"
        summary += "The Cluster-Driven Certified Unlearning method demonstrates:\n\n"
        
        # KFR analysis
        kfr = cd_metrics.get('KFR', 0)
        if kfr > 0.8:
            summary += f"- Excellent knowledge forgetting rate (KFR = {kfr:.4f}), indicating highly effective unlearning of targeted information\n"
        elif kfr > 0.5:
            summary += f"- Good knowledge forgetting rate (KFR = {kfr:.4f}), showing effective unlearning of targeted information\n"
        else:
            summary += f"- Moderate knowledge forgetting rate (KFR = {kfr:.4f})\n"
        
        # KRR analysis
        krr = cd_metrics.get('KRR', 0)
        if krr > 0.95:
            summary += f"- Excellent knowledge retention rate (KRR = {krr:.4f}), maintaining almost all utility of the original model\n"
        elif krr > 0.9:
            summary += f"- Very good knowledge retention rate (KRR = {krr:.4f}), preserving most of the model's original capabilities\n"
        else:
            summary += f"- Moderate knowledge retention rate (KRR = {krr:.4f})\n"
        
        # Compute time analysis
        compute_time = cd_metrics.get('compute_time', 0) / 60  # Convert to minutes
        if "relearn" in results["method_comparison"]:
            relearn_time = results["method_comparison"]["relearn"].get('compute_time', 0) / 60
            time_reduction = (relearn_time - compute_time) / relearn_time * 100 if relearn_time > 0 else 0
            
            if time_reduction > 60:
                summary += f"- Significant computational efficiency ({time_reduction:.1f}% reduction in compute time compared to ReLearn)\n"
            elif time_reduction > 30:
                summary += f"- Good computational efficiency ({time_reduction:.1f}% reduction in compute time compared to ReLearn)\n"
            else:
                summary += f"- Moderate computational efficiency ({time_reduction:.1f}% reduction in compute time compared to ReLearn)\n"
        else:
            summary += f"- Compute time of {compute_time:.2f} minutes\n"
        
        # Certification analysis
        if "certified" in cd_metrics:
            if cd_metrics["certified"]:
                summary += f"- Successfully certified unlearning with KL divergence of {cd_metrics.get('kl_divergence', 'N/A'):.6f}\n"
            else:
                summary += "- Did not achieve certification threshold\n"
        
        summary += "\n"
    
    # Comparison with baselines
    if "method_comparison" in results and len(results["method_comparison"]) > 2:  # More than just our method and original
        summary += "### Comparison with Baselines\n\n"
        
        # Get all methods except original model
        methods = [m for m in results["method_comparison"].keys() if m != "original_model"]
        
        # Find best method for each metric
        best_kfr_method = max(methods, key=lambda m: results["method_comparison"][m].get('KFR', 0))
        best_krr_method = max(methods, key=lambda m: results["method_comparison"][m].get('KRR', 0))
        best_perplexity_method = min(methods, key=lambda m: results["method_comparison"][m].get('perplexity', float('inf')))
        best_compute_method = min(methods, key=lambda m: results["method_comparison"][m].get('compute_time', float('inf')))
        
        summary += f"- Best knowledge forgetting rate (KFR): **{best_kfr_method}** ({results['method_comparison'][best_kfr_method].get('KFR', 0):.4f})\n"
        summary += f"- Best knowledge retention rate (KRR): **{best_krr_method}** ({results['method_comparison'][best_krr_method].get('KRR', 0):.4f})\n"
        summary += f"- Best perplexity: **{best_perplexity_method}** ({results['method_comparison'][best_perplexity_method].get('perplexity', 0):.4f})\n"
        summary += f"- Most efficient method: **{best_compute_method}** ({results['method_comparison'][best_compute_method].get('compute_time', 0) / 60:.2f} minutes)\n\n"
        
        # Comparative analysis
        if "cluster_driven" in methods:
            cd_metrics = results["method_comparison"]["cluster_driven"]
            
            # Compare KFR
            cd_kfr = cd_metrics.get('KFR', 0)
            best_kfr = results["method_comparison"][best_kfr_method].get('KFR', 0)
            
            if best_kfr_method == "cluster_driven":
                summary += "- The Cluster-Driven method achieves the best knowledge forgetting rate among all methods\n"
            else:
                kfr_diff = (cd_kfr - best_kfr) / best_kfr * 100
                if kfr_diff > -5:
                    summary += f"- The Cluster-Driven method's knowledge forgetting rate is comparable to the best method ({kfr_diff:.1f}% difference)\n"
                else:
                    summary += f"- The Cluster-Driven method's knowledge forgetting rate is lower than the best method by {-kfr_diff:.1f}%\n"
            
            # Compare KRR
            cd_krr = cd_metrics.get('KRR', 0)
            best_krr = results["method_comparison"][best_krr_method].get('KRR', 0)
            
            if best_krr_method == "cluster_driven":
                summary += "- The Cluster-Driven method achieves the best knowledge retention rate among all methods\n"
            else:
                krr_diff = (cd_krr - best_krr) / best_krr * 100
                if krr_diff > -2:
                    summary += f"- The Cluster-Driven method's knowledge retention rate is comparable to the best method ({krr_diff:.1f}% difference)\n"
                else:
                    summary += f"- The Cluster-Driven method's knowledge retention rate is lower than the best method by {-krr_diff:.1f}%\n"
            
            # Compare efficiency
            cd_time = cd_metrics.get('compute_time', 0)
            best_time = results["method_comparison"][best_compute_method].get('compute_time', 0)
            
            if best_compute_method == "cluster_driven":
                summary += "- The Cluster-Driven method is the most computationally efficient among all methods\n"
            else:
                time_diff = (cd_time - best_time) / best_time * 100
                if time_diff < 20:
                    summary += f"- The Cluster-Driven method's computational efficiency is comparable to the best method ({time_diff:.1f}% difference)\n"
                else:
                    summary += f"- The Cluster-Driven method is {time_diff:.1f}% slower than the most efficient method\n"
            
            # Certification advantage
            if "certified" in cd_metrics and cd_metrics["certified"]:
                summary += "- The Cluster-Driven method provides formal certification of unlearning, which is a unique advantage over other methods\n"
    
    # Add future work suggestions
    summary += "\n### Future Work\n\n"
    summary += "1. **Scalability Testing**: Evaluate the methods on larger language models like GPT-3 or LLaMA to assess scalability.\n"
    summary += "2. **Real-world Data**: Test the unlearning methods on real-world sensitive information deletion requests.\n"
    summary += "3. **Sequential Unlearning Improvements**: Further refine methods for handling continuous unlearning requests without performance degradation.\n"
    summary += "4. **Certification Guarantees**: Strengthen the theoretical guarantees for unlearning certification.\n"
    
    # Save the summary
    with open(output_file, "w") as f:
        f.write(summary)


def get_available_device():
    """
    Get available device (GPU or CPU).
    
    Returns:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        # Get the device with the most free memory
        device_id = 0
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory.append(torch.cuda.memory_allocated(i))
            device_id = free_memory.index(min(free_memory))
        
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    return device