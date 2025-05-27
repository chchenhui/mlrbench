import os
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from datasets import Dataset
from tqdm import tqdm
import logging

from utils.data_utils import (
    DeveloperProfile, 
    CodingTask, 
    simulate_developer_interactions
)

def evaluate_models(
    baseline_model,
    adaptive_model,
    test_data: Dataset,
    num_developers: int = 30,
    num_tasks: int = 12,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Evaluate both baseline and adaptive models on simulated developer interactions.
    
    Args:
        baseline_model: The static baseline model
        adaptive_model: The adaptive model with reinforcement learning
        test_data: Dataset of coding tasks
        num_developers: Number of simulated developers
        num_tasks: Number of tasks per developer
        device: Device to run inference on
    
    Returns:
        Dict containing evaluation metrics for both models
    """
    logger = logging.getLogger('adaptive_code_assistant')
    logger.info(f"Evaluating models with {num_developers} developers on {num_tasks} tasks each")
    
    # Convert test data to CodingTask objects
    tasks = []
    for item in test_data:
        task = CodingTask(
            task_id=item.get('task_id', f"task_{len(tasks)}"),
            context=item['context'],
            solution=item['solution'],
            description=item.get('description', ''),
            tags=item.get('tags', [])
        )
        tasks.append(task)
    
    # Results container
    results = {
        'baseline': {
            'developers': [],
            'acceptance_rate': [],
            'avg_edit_distance': [],
            'avg_reward': [],
            'task_completion_times': [],
            'code_quality_scores': []
        },
        'adaptive': {
            'developers': [],
            'acceptance_rate': [],
            'avg_edit_distance': [],
            'avg_reward': [],
            'task_completion_times': [],
            'code_quality_scores': []
        },
        'summary': {
            'baseline': {},
            'adaptive': {},
            'improvement': {}
        }
    }
    
    # Simulate developers
    for dev_idx in tqdm(range(num_developers), desc="Simulating developers"):
        dev_profile = DeveloperProfile(f"dev_{dev_idx}")
        
        # Select random tasks for this developer
        if len(tasks) <= num_tasks:
            dev_tasks = tasks
        else:
            # Random sample without replacement
            task_indices = np.random.choice(
                len(tasks), 
                size=num_tasks, 
                replace=False
            )
            dev_tasks = [tasks[i] for i in task_indices]
        
        # Evaluate baseline model
        logger.info(f"Evaluating baseline model for developer {dev_idx}")
        baseline_metrics = simulate_developer_interactions(
            model=baseline_model,
            developer_profile=dev_profile,
            tasks=dev_tasks,
            device=device
        )
        
        # Reset developer profile for fair comparison
        dev_profile = DeveloperProfile(f"dev_{dev_idx}")
        
        # Evaluate adaptive model
        logger.info(f"Evaluating adaptive model for developer {dev_idx}")
        adaptive_metrics = simulate_developer_interactions(
            model=adaptive_model,
            developer_profile=dev_profile,
            tasks=dev_tasks,
            device=device
        )
        
        # Store results
        results['baseline']['developers'].append(f"dev_{dev_idx}")
        results['baseline']['acceptance_rate'].append(baseline_metrics['acceptance_rate'])
        results['baseline']['avg_edit_distance'].append(baseline_metrics['avg_edit_distance'])
        results['baseline']['avg_reward'].append(baseline_metrics['avg_reward'])
        results['baseline']['task_completion_times'].extend(baseline_metrics['task_completion_times'])
        results['baseline']['code_quality_scores'].extend(baseline_metrics['code_quality_scores'])
        
        results['adaptive']['developers'].append(f"dev_{dev_idx}")
        results['adaptive']['acceptance_rate'].append(adaptive_metrics['acceptance_rate'])
        results['adaptive']['avg_edit_distance'].append(adaptive_metrics['avg_edit_distance'])
        results['adaptive']['avg_reward'].append(adaptive_metrics['avg_reward'])
        results['adaptive']['task_completion_times'].extend(adaptive_metrics['task_completion_times'])
        results['adaptive']['code_quality_scores'].extend(adaptive_metrics['code_quality_scores'])
    
    # Compute summary statistics
    # For baseline
    results['summary']['baseline'] = {
        'avg_acceptance_rate': np.mean(results['baseline']['acceptance_rate']),
        'std_acceptance_rate': np.std(results['baseline']['acceptance_rate']),
        'avg_edit_distance': np.mean(results['baseline']['avg_edit_distance']),
        'std_edit_distance': np.std(results['baseline']['avg_edit_distance']),
        'avg_reward': np.mean(results['baseline']['avg_reward']),
        'std_reward': np.std(results['baseline']['avg_reward']),
        'avg_task_completion_time': np.mean(results['baseline']['task_completion_times']),
        'std_task_completion_time': np.std(results['baseline']['task_completion_times']),
        'avg_code_quality': np.mean(results['baseline']['code_quality_scores']),
        'std_code_quality': np.std(results['baseline']['code_quality_scores'])
    }
    
    # For adaptive
    results['summary']['adaptive'] = {
        'avg_acceptance_rate': np.mean(results['adaptive']['acceptance_rate']),
        'std_acceptance_rate': np.std(results['adaptive']['acceptance_rate']),
        'avg_edit_distance': np.mean(results['adaptive']['avg_edit_distance']),
        'std_edit_distance': np.std(results['adaptive']['avg_edit_distance']),
        'avg_reward': np.mean(results['adaptive']['avg_reward']),
        'std_reward': np.std(results['adaptive']['avg_reward']),
        'avg_task_completion_time': np.mean(results['adaptive']['task_completion_times']),
        'std_task_completion_time': np.std(results['adaptive']['task_completion_times']),
        'avg_code_quality': np.mean(results['adaptive']['code_quality_scores']),
        'std_code_quality': np.std(results['adaptive']['code_quality_scores'])
    }
    
    # Compute improvements (adaptive over baseline)
    improvement = {
        'acceptance_rate': (
            results['summary']['adaptive']['avg_acceptance_rate'] - 
            results['summary']['baseline']['avg_acceptance_rate']
        ) / results['summary']['baseline']['avg_acceptance_rate'] * 100,
        
        'edit_distance': (
            results['summary']['adaptive']['avg_edit_distance'] - 
            results['summary']['baseline']['avg_edit_distance']
        ) / results['summary']['baseline']['avg_edit_distance'] * 100,
        
        'reward': (
            results['summary']['adaptive']['avg_reward'] - 
            results['summary']['baseline']['avg_reward']
        ) / results['summary']['baseline']['avg_reward'] * 100,
        
        'task_completion_time': (
            results['summary']['baseline']['avg_task_completion_time'] - 
            results['summary']['adaptive']['avg_task_completion_time']
        ) / results['summary']['baseline']['avg_task_completion_time'] * 100,
        
        'code_quality': (
            results['summary']['adaptive']['avg_code_quality'] - 
            results['summary']['baseline']['avg_code_quality']
        ) / results['summary']['baseline']['avg_code_quality'] * 100
    }
    
    results['summary']['improvement'] = improvement
    
    logger.info(f"Evaluation complete. Summary of improvements:")
    logger.info(f"Acceptance rate: {improvement['acceptance_rate']:.2f}%")
    logger.info(f"Edit distance: {improvement['edit_distance']:.2f}%")
    logger.info(f"Reward: {improvement['reward']:.2f}%")
    logger.info(f"Task completion time: {improvement['task_completion_time']:.2f}%")
    logger.info(f"Code quality: {improvement['code_quality']:.2f}%")
    
    return results

def compute_statistical_significance(
    baseline_metrics: List[float],
    adaptive_metrics: List[float]
) -> Dict[str, float]:
    """
    Compute statistical significance tests comparing baseline and adaptive models.
    
    Args:
        baseline_metrics: List of metrics from baseline model
        adaptive_metrics: List of metrics from adaptive model
    
    Returns:
        Dict containing p-values and effect sizes
    """
    from scipy import stats
    import numpy as np
    
    # Check normality of distributions
    _, baseline_norm_p = stats.shapiro(baseline_metrics)
    _, adaptive_norm_p = stats.shapiro(adaptive_metrics)
    
    # Choose appropriate test based on normality
    if baseline_norm_p > 0.05 and adaptive_norm_p > 0.05:
        # Both distributions are approximately normal, use t-test
        t_stat, p_value = stats.ttest_rel(adaptive_metrics, baseline_metrics)
        test_name = "Paired t-test"
        
        # Effect size (Cohen's d for paired samples)
        diff = np.array(adaptive_metrics) - np.array(baseline_metrics)
        d = np.mean(diff) / np.std(diff, ddof=1)
        effect_size = abs(d)
        effect_size_name = "Cohen's d"
    else:
        # At least one distribution is not normal, use Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(adaptive_metrics, baseline_metrics)
        test_name = "Wilcoxon signed-rank test"
        
        # Effect size (r = Z / sqrt(N)) for Wilcoxon
        z = stats.norm.ppf(p_value / 2)  # Two-tailed, so divide p-value by 2
        r = abs(z) / np.sqrt(len(baseline_metrics) * 2)
        effect_size = r
        effect_size_name = "r"
    
    return {
        'test': test_name,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name
    }

def evaluate_code_correctness(
    generated_code: str,
    test_cases: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate the correctness of generated code.
    
    Args:
        generated_code: The code to evaluate
        test_cases: List of test cases, each with inputs and expected outputs
    
    Returns:
        Dict containing evaluation results
    """
    if not test_cases:
        # Return a simple static analysis if no test cases provided
        return evaluate_code_quality(generated_code)
    
    # Results container
    results = {
        'passed': 0,
        'failed': 0,
        'total': len(test_cases),
        'errors': [],
        'pass_rate': 0.0
    }
    
    # Create a namespace to execute the code
    namespace = {}
    
    try:
        # Execute the code to define the function
        exec(generated_code, namespace)
        
        # Extract function name (assumes the code defines exactly one function)
        func_name = None
        for key, value in namespace.items():
            if callable(value) and key != '__builtins__':
                func_name = key
                break
        
        if not func_name:
            results['errors'].append("No function defined in the generated code")
            return results
        
        # Get the function
        func = namespace[func_name]
        
        # Run test cases
        for i, test_case in enumerate(test_cases):
            try:
                # Get inputs and expected output
                inputs = test_case['inputs']
                expected = test_case['expected']
                
                # Call the function
                if isinstance(inputs, list):
                    actual = func(*inputs)
                elif isinstance(inputs, dict):
                    actual = func(**inputs)
                else:
                    actual = func(inputs)
                
                # Check if output matches expected
                if actual == expected:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(
                        f"Test case {i+1} failed: expected {expected}, got {actual}"
                    )
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Test case {i+1} raised exception: {str(e)}")
        
        # Calculate pass rate
        results['pass_rate'] = results['passed'] / results['total'] if results['total'] > 0 else 0
        
    except Exception as e:
        results['errors'].append(f"Failed to execute generated code: {str(e)}")
    
    return results

def evaluate_code_quality(code: str) -> Dict[str, Any]:
    """
    Evaluate code quality using static analysis.
    
    Args:
        code: The code to evaluate
    
    Returns:
        Dict containing quality metrics
    """
    import tokenize
    from io import BytesIO
    
    # Results container
    results = {
        'line_count': 0,
        'cyclomatic_complexity': 0,
        'comment_ratio': 0.0,
        'lint_errors': 0,
        'quality_score': 0.0
    }
    
    try:
        # Count lines
        lines = code.split('\n')
        results['line_count'] = len(lines)
        
        # Tokenize code
        tokens = list(tokenize.tokenize(BytesIO(code.encode('utf-8')).readline))
        
        # Count comments
        comment_lines = sum(1 for token in tokens if token.type == tokenize.COMMENT)
        results['comment_ratio'] = comment_lines / results['line_count'] if results['line_count'] > 0 else 0
        
        # Approximate cyclomatic complexity by counting branches
        branch_keywords = ['if', 'for', 'while', 'and', 'or', 'else', 'elif']
        complexity = 1  # Base complexity
        for line in lines:
            for keyword in branch_keywords:
                if keyword in line and not line.strip().startswith('#'):
                    # Simple heuristic: count keyword if it's not in a comment
                    complexity += 1
        
        results['cyclomatic_complexity'] = complexity
        
        # Simulate lint errors (random for simulation)
        # In a real implementation, use tools like Pylint or Flake8
        lint_errors = int(np.random.gamma(2, 2))  # Gamma distribution for a right-skewed distribution
        results['lint_errors'] = lint_errors
        
        # Compute overall quality score (higher is better)
        # Function of comment ratio (higher is better)
        # and complexity/lines (lower is better)
        # and lint errors (lower is better)
        comment_score = min(results['comment_ratio'] * 10, 3)  # Max 3 points for comments
        complexity_score = max(5 - results['cyclomatic_complexity'] / max(results['line_count'] / 10, 1), 0)  # Max 5 points for low complexity
        lint_score = max(2 - results['lint_errors'] / 2, 0)  # Max 2 points for few lint errors
        
        results['quality_score'] = comment_score + complexity_score + lint_score
    
    except Exception as e:
        # If analysis fails, return default values
        results['error'] = str(e)
    
    return results