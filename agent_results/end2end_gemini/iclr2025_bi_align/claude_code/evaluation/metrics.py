"""
Evaluation Metrics Module

This module implements statistical analysis and evaluation metrics for the experiment.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, List, Any, Optional

def evaluate_experiment(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform statistical analysis on experiment results.
    
    Args:
        results: Results dictionary from the experiment
        config: Configuration dictionary
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating experiment results")
    
    # Convert trial data to DataFrame for easier analysis
    trial_data = pd.DataFrame(results["trial_data"])
    participant_data = pd.DataFrame(results["participant_data"])
    
    # Initialize results dictionary
    evaluation_results = {
        "statistical_tests": {},
        "effect_sizes": {},
        "confidence_intervals": {},
        "regression_analyses": {},
        "subgroup_analyses": {}
    }
    
    # Perform statistical tests for key metrics
    metrics_to_test = [
        {"column": "mental_model_accuracy", "name": "Mental Model Accuracy", "higher_is_better": True},
        {"column": "user_correct", "name": "Diagnostic Accuracy", "higher_is_better": True},
        {"column": "confusion_level", "name": "Confusion Level", "higher_is_better": False},
        {"column": "decision_time", "name": "Decision Time", "higher_is_better": False},
        {"column": "helpfulness", "name": "Intervention Helpfulness", "higher_is_better": True}
    ]
    
    # Create a pivot table for trial-level metrics
    trial_pivot = trial_data.pivot_table(
        index='participant_id', 
        values=['user_correct', 'confusion_level', 'decision_time', 'helpfulness'], 
        aggfunc=np.mean
    ).reset_index()
    
    # Merge with participant data to get group and mental_model_accuracy
    analysis_df = pd.merge(
        trial_pivot,
        participant_data[['participant_id', 'group', 'mental_model_accuracy']],
        on='participant_id'
    )
    
    # Perform t-tests and calculate effect sizes
    for metric in metrics_to_test:
        column = metric["column"]
        name = metric["name"]
        higher_is_better = metric["higher_is_better"]
        
        if column in analysis_df.columns:
            treatment_values = analysis_df[analysis_df['group'] == 'treatment'][column]
            control_values = analysis_df[analysis_df['group'] == 'control'][column]
            
            # Skip if insufficient data
            if len(treatment_values) < 2 or len(control_values) < 2:
                logger.warning(f"Insufficient data for {name} analysis")
                continue
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(treatment_values) ** 2 + np.std(control_values) ** 2) / 2)
            effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
            
            # If higher values are worse, negate the effect size
            if not higher_is_better:
                effect_size = -effect_size
            
            # Calculate 95% confidence interval for the difference
            treatment_mean = np.mean(treatment_values)
            control_mean = np.mean(control_values)
            treatment_sem = stats.sem(treatment_values)
            control_sem = stats.sem(control_values)
            diff_mean = treatment_mean - control_mean
            
            # Accounting for direction
            if not higher_is_better:
                diff_mean = -diff_mean
            
            diff_sem = np.sqrt(treatment_sem**2 + control_sem**2)
            conf_int = stats.t.interval(0.95, len(treatment_values) + len(control_values) - 2, 
                                       loc=diff_mean, scale=diff_sem)
            
            # Store results
            evaluation_results["statistical_tests"][name] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "treatment_mean": float(np.mean(treatment_values)),
                "control_mean": float(np.mean(control_values)),
                "difference": float(np.mean(treatment_values) - np.mean(control_values))
            }
            
            evaluation_results["effect_sizes"][name] = {
                "cohens_d": float(effect_size),
                "interpretation": interpret_cohens_d(effect_size)
            }
            
            evaluation_results["confidence_intervals"][name] = {
                "lower_bound": float(conf_int[0]),
                "upper_bound": float(conf_int[1])
            }
    
    # Analyze performance across different expertise levels
    expertise_levels = config['participants']['expertise_levels']
    expertise_analysis = {}
    
    for level in expertise_levels:
        level_df = analysis_df[analysis_df['participant_id'].str.contains(f"_{level}_")]
        
        if len(level_df) == 0:
            continue
        
        level_treatment = level_df[level_df['group'] == 'treatment']['user_correct']
        level_control = level_df[level_df['group'] == 'control']['user_correct']
        
        # Skip if insufficient data
        if len(level_treatment) < 2 or len(level_control) < 2:
            continue
            
        expertise_analysis[level] = {
            "treatment_accuracy": float(np.mean(level_treatment)),
            "control_accuracy": float(np.mean(level_control)),
            "difference": float(np.mean(level_treatment) - np.mean(level_control)),
            "sample_size": len(level_df)
        }
    
    evaluation_results["subgroup_analyses"]["expertise_levels"] = expertise_analysis
    
    # Analyze performance across different case complexities
    complexity_levels = ['simple', 'medium', 'complex']
    complexity_analysis = {}
    
    for complexity in complexity_levels:
        complexity_df = trial_data[trial_data['complexity'] == complexity]
        
        complexity_treatment = complexity_df[complexity_df['group'] == 'treatment']['user_correct']
        complexity_control = complexity_df[complexity_df['group'] == 'control']['user_correct']
        
        # Skip if insufficient data
        if len(complexity_treatment) == 0 or len(complexity_control) == 0:
            continue
            
        complexity_analysis[complexity] = {
            "treatment_accuracy": float(np.mean(complexity_treatment)),
            "control_accuracy": float(np.mean(complexity_control)),
            "difference": float(np.mean(complexity_treatment) - np.mean(complexity_control)),
            "sample_size": len(complexity_df)
        }
    
    evaluation_results["subgroup_analyses"]["complexity_levels"] = complexity_analysis
    
    # Analyze intervention effectiveness
    intervention_df = trial_data[trial_data['group'] == 'treatment']
    intervention_types = intervention_df['intervention_type'].unique()
    
    intervention_analysis = {}
    for int_type in intervention_types:
        if int_type == 'none':
            continue
            
        type_df = intervention_df[intervention_df['intervention_type'] == int_type]
        
        if len(type_df) == 0:
            continue
            
        helpfulness = type_df['helpfulness'].mean()
        improvement_rate = np.mean(type_df['understanding_improved'])
        
        intervention_analysis[int_type] = {
            "helpfulness": float(helpfulness),
            "improvement_rate": float(improvement_rate),
            "count": len(type_df)
        }
    
    evaluation_results["subgroup_analyses"]["intervention_types"] = intervention_analysis
    
    # Calculate learning trends over time
    if len(trial_data) > 0:
        # Sort by participant and trial ID
        sorted_df = trial_data.sort_values(['participant_id', 'trial_id'])
        
        # Group by participant and calculate rolling accuracy
        treatment_learning = []
        control_learning = []
        
        for group_name, group_df in sorted_df.groupby(['group', 'participant_id']):
            group_type, participant_id = group_name
            
            if len(group_df) < 5:  # Need enough trials to measure learning
                continue
                
            # Calculate cumulative accuracy over time
            accuracy_trend = group_df['user_correct'].cumsum() / (np.arange(len(group_df)) + 1)
            
            # Append to appropriate group
            if group_type == 'treatment':
                treatment_learning.append(accuracy_trend.values)
            else:
                control_learning.append(accuracy_trend.values)
        
        # Calculate average learning curves if we have data
        if treatment_learning and control_learning:
            # Ensure all arrays have same length (use minimum length)
            min_treatment_len = min([len(arr) for arr in treatment_learning])
            min_control_len = min([len(arr) for arr in control_learning])
            
            treatment_learning_padded = [arr[:min_treatment_len] for arr in treatment_learning]
            control_learning_padded = [arr[:min_control_len] for arr in control_learning]
            
            # Calculate mean learning curves
            treatment_curve = np.mean(treatment_learning_padded, axis=0)
            control_curve = np.mean(control_learning_padded, axis=0)
            
            # Store in results
            evaluation_results["learning_trends"] = {
                "trial_points": list(range(1, min(min_treatment_len, min_control_len) + 1)),
                "treatment_curve": treatment_curve.tolist(),
                "control_curve": control_curve.tolist()
            }
    
    logger.info("Evaluation completed")
    return evaluation_results

def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        String interpretation of the effect size
    """
    d = abs(d)
    
    if d < 0.2:
        return "Negligible effect"
    elif d < 0.5:
        return "Small effect"
    elif d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"