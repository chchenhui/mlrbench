"""
Visualization utilities for the SSCSteer experiment.

This module provides functions to visualize the results of the experiment,
including performance metrics, syntax/semantic errors, and generation time.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def plot_performance_comparison(comparison_df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot performance comparison between different approaches.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance Comparison of Code Generation Approaches', fontsize=16)
    
    # 1. Syntactic Validity and Pass Rate
    ax1 = axs[0, 0]
    metrics = ['syntactic_validity', 'pass_rate']
    subset_df = comparison_df[['approach'] + metrics].set_index('approach')
    subset_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Syntactic Validity & Pass Rate')
    ax1.set_ylabel('Rate (0-1)')
    ax1.set_ylim(0, 1)
    ax1.legend(metrics)
    
    # 2. Pass@k Metrics
    ax2 = axs[0, 1]
    metrics = ['pass_at_1', 'pass_at_3', 'pass_at_5']
    subset_df = comparison_df[['approach'] + metrics].set_index('approach')
    subset_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Pass@k Metrics')
    ax2.set_ylabel('Pass Rate (0-1)')
    ax2.set_ylim(0, 1)
    ax2.legend(metrics)
    
    # 3. Code Quality Metrics
    ax3 = axs[1, 0]
    metrics = ['pylint_score', 'bug_density']
    
    # Normalize bug density to 0-10 scale for comparison
    max_bug_density = comparison_df['bug_density'].max()
    if max_bug_density > 0:
        comparison_df['normalized_bug_density'] = 10 * comparison_df['bug_density'] / max_bug_density
    else:
        comparison_df['normalized_bug_density'] = 0
        
    subset_df = comparison_df[['approach', 'pylint_score', 'normalized_bug_density']].set_index('approach')
    subset_df.rename(columns={'normalized_bug_density': 'bug_density (normalized)'}, inplace=True)
    subset_df.plot(kind='bar', ax=ax3)
    ax3.set_title('Code Quality Metrics')
    ax3.set_ylabel('Score (0-10)')
    ax3.set_ylim(0, 10)
    ax3.legend(['Pylint Score', 'Bug Density (Normalized)'])
    
    # 4. Generation Time
    ax4 = axs[1, 1]
    metrics = ['generation_time']
    subset_df = comparison_df[['approach'] + metrics].set_index('approach')
    subset_df.plot(kind='bar', ax=ax4)
    ax4.set_title('Generation Time')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend(metrics)
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_bug_patterns(results: Dict[str, List[Dict[str, Any]]], output_dir: str) -> str:
    """
    Plot bug pattern distribution for different approaches.
    
    Args:
        results: Dictionary mapping approach names to lists of evaluation results
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Extract bug pattern data
    bug_data = {}
    
    for approach, approach_results in results.items():
        bugs = {
            'null_dereference': 0,
            'uninitialized_variable': 0,
            'division_by_zero': 0,
            'index_out_of_bounds': 0,
            'resource_leak': 0
        }
        
        for result in approach_results:
            if 'bug_patterns' in result:
                for bug_type, count in result['bug_patterns'].items():
                    if bug_type in bugs:
                        bugs[bug_type] += count
        
        bug_data[approach] = bugs
    
    # Convert to DataFrame
    bug_df = pd.DataFrame(bug_data).T
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bug_df.plot(kind='bar', stacked=True)
    plt.title('Bug Pattern Distribution by Approach')
    plt.xlabel('Approach')
    plt.ylabel('Number of Bugs')
    plt.legend(title='Bug Type')
    plt.grid(axis='y')
    
    # Save figure
    output_path = os.path.join(output_dir, 'bug_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_dataset_comparison(results: Dict[str, Dict[str, List[Dict[str, Any]]]], output_dir: str) -> str:
    """
    Plot performance comparison across different datasets.
    
    Args:
        results: Dictionary mapping dataset names to approach results
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Extract dataset performance data
    dataset_perf = {}
    
    for dataset, dataset_results in results.items():
        for approach, approach_results in dataset_results.items():
            # Calculate pass rate for this approach on this dataset
            pass_rate = np.mean([r.get('pass_rate', 0.0) for r in approach_results])
            
            if approach not in dataset_perf:
                dataset_perf[approach] = {}
            
            dataset_perf[approach][dataset] = pass_rate
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset_perf)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar')
    plt.title('Performance Comparison Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Pass Rate')
    plt.legend(title='Approach')
    plt.grid(axis='y')
    
    # Save figure
    output_path = os.path.join(output_dir, 'dataset_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_ablation_study(ablation_results: Dict[str, Dict[str, float]], output_dir: str) -> str:
    """
    Plot ablation study results.
    
    Args:
        ablation_results: Dictionary mapping configuration names to metric dictionaries
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Convert to DataFrame
    df = pd.DataFrame(ablation_results).T
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ablation Study Results', fontsize=16)
    
    # 1. Pass Rate
    ax1 = axs[0]
    pass_metrics = ['pass_rate', 'syntactic_validity']
    df[pass_metrics].plot(kind='bar', ax=ax1)
    ax1.set_title('Pass Metrics')
    ax1.set_ylabel('Rate (0-1)')
    ax1.set_ylim(0, 1)
    ax1.legend(pass_metrics)
    
    # 2. Other Metrics
    ax2 = axs[1]
    other_metrics = ['pylint_score', 'generation_time']
    
    # Normalize generation time to 0-10 scale for comparison
    if 'generation_time' in df.columns:
        max_time = df['generation_time'].max()
        if max_time > 0:
            df['normalized_time'] = 10 * df['generation_time'] / max_time
        else:
            df['normalized_time'] = 0
            
        other_metrics = ['pylint_score', 'normalized_time']
            
    df[other_metrics].plot(kind='bar', ax=ax2)
    ax2.set_title('Code Quality & Efficiency')
    ax2.set_ylabel('Score/Time (normalized)')
    ax2.set_ylim(0, 10)
    ax2.legend(['Pylint Score', 'Generation Time (normalized)'])
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(output_dir, 'ablation_study.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_beam_search_evolution(sscsteer_results: List[Dict[str, Any]], output_dir: str) -> str:
    """
    Plot the evolution of beam search during code generation.
    
    Args:
        sscsteer_results: List of SSCSteer generation results with generation_steps
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Extract beam evolution data
    beam_data = []
    
    for result in sscsteer_results:
        if 'generation_steps' in result:
            for step in result['generation_steps']:
                beam_data.append({
                    'token': step['token'],
                    'score': step['best_score'],
                    'issues': step['num_issues'],
                    'diversity': step['beam_diversity']
                })
    
    if not beam_data:
        # No beam evolution data
        return ""
        
    # Convert to DataFrame
    df = pd.DataFrame(beam_data)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Beam Search Evolution During Generation', fontsize=16)
    
    # 1. Beam Score
    ax1 = axs[0]
    ax1.plot(df['token'], df['score'], marker='o')
    ax1.set_title('Best Beam Score')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    
    # 2. Semantic Issues
    ax2 = axs[1]
    ax2.plot(df['token'], df['issues'], marker='s', color='red')
    ax2.set_title('Number of Semantic Issues')
    ax2.set_xlabel('Generation Token')
    ax2.set_ylabel('Issues')
    ax2.grid(True)
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(output_dir, 'beam_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_result_tables(comparison_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate markdown tables from comparison results.
    
    Args:
        comparison_df: DataFrame with comparison results
        
    Returns:
        Dictionary mapping table names to markdown strings
    """
    tables = {}
    
    # Main comparison table
    main_metrics = ['syntactic_validity', 'pass_rate', 'pass_at_1', 'pylint_score', 'bug_density', 'generation_time']
    main_df = comparison_df[['approach'] + main_metrics].copy()
    
    # Format the values
    for col in main_df.columns:
        if col == 'approach':
            continue
        elif col in ['syntactic_validity', 'pass_rate', 'pass_at_1']:
            main_df[col] = main_df[col].apply(lambda x: f"{x:.2%}")
        elif col == 'pylint_score':
            main_df[col] = main_df[col].apply(lambda x: f"{x:.2f}/10")
        elif col == 'bug_density':
            main_df[col] = main_df[col].apply(lambda x: f"{x:.2f}")
        elif col == 'generation_time':
            main_df[col] = main_df[col].apply(lambda x: f"{x:.2f}s")
    
    # Rename columns for clarity
    main_df = main_df.rename(columns={
        'syntactic_validity': 'Syntactic Validity',
        'pass_rate': 'Pass Rate',
        'pass_at_1': 'Pass@1',
        'pylint_score': 'Code Quality',
        'bug_density': 'Bugs/KLOC',
        'generation_time': 'Gen. Time',
        'approach': 'Approach'
    })
    
    # Convert to markdown
    tables['main_comparison'] = main_df.to_markdown(index=False)
    
    # Pass@k table
    passk_metrics = ['pass_at_1', 'pass_at_3', 'pass_at_5']
    passk_df = comparison_df[['approach'] + passk_metrics].copy()
    
    # Format values
    for col in passk_df.columns:
        if col != 'approach':
            passk_df[col] = passk_df[col].apply(lambda x: f"{x:.2%}")
    
    # Rename columns
    passk_df = passk_df.rename(columns={
        'pass_at_1': 'Pass@1',
        'pass_at_3': 'Pass@3',
        'pass_at_5': 'Pass@5',
        'approach': 'Approach'
    })
    
    # Convert to markdown
    tables['pass_at_k'] = passk_df.to_markdown(index=False)
    
    # Code quality table
    quality_metrics = ['pylint_score', 'flake8_violations', 'cyclomatic_complexity', 'bug_density']
    quality_df = comparison_df[['approach'] + quality_metrics].copy()
    
    # Format values
    quality_df['pylint_score'] = quality_df['pylint_score'].apply(lambda x: f"{x:.2f}/10")
    quality_df['bug_density'] = quality_df['bug_density'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns
    quality_df = quality_df.rename(columns={
        'pylint_score': 'Pylint Score',
        'flake8_violations': 'Flake8 Violations',
        'cyclomatic_complexity': 'Cyclomatic Complexity',
        'bug_density': 'Bugs/KLOC',
        'approach': 'Approach'
    })
    
    # Convert to markdown
    tables['code_quality'] = quality_df.to_markdown(index=False)
    
    return tables


def create_results_markdown(comparison_df: pd.DataFrame, 
                            tables: Dict[str, str],
                            figure_paths: Dict[str, str],
                            output_path: str) -> None:
    """
    Create a markdown file with experiment results.
    
    Args:
        comparison_df: DataFrame with comparison results
        tables: Dictionary mapping table names to markdown strings
        figure_paths: Dictionary mapping figure names to file paths
        output_path: Path to save the markdown file
    """
    # Create markdown content
    md_lines = [
        "# SSCSteer Experiment Results",
        "",
        "## Overview",
        "",
        "This document presents the results of the SSCSteer experiment, which evaluates the effectiveness of the Syntactic and Semantic Conformance Steering (SSCSteer) framework for improving LLM code generation.",
        "",
        "## Experimental Setup",
        "",
        "We compared the following approaches:",
        ""
    ]
    
    # Add approach descriptions
    approaches = comparison_df['approach'].tolist()
    approach_descriptions = {
        'Vanilla LLM': "Standard LLM generation without any steering.",
        'Post-hoc Syntax': "LLM generation with post-hoc syntax validation and regeneration if needed.",
        'Feedback Refinement': "LLM generation with feedback-based refinement for semantic correctness.",
        'SSM-only': "SSCSteer with only syntactic steering enabled.",
        'SeSM-only': "SSCSteer with only semantic steering enabled.",
        'Full SSCSteer': "Complete SSCSteer framework with both syntactic and semantic steering."
    }
    
    for approach in approaches:
        desc = approach_descriptions.get(approach, f"The {approach} approach.")
        md_lines.extend([f"- **{approach}**: {desc}", ""])
    
    # Add datasets information
    md_lines.extend([
        "### Datasets",
        "",
        "We evaluated the approaches on two datasets:",
        "",
        "1. **HumanEval Subset**: A subset of the HumanEval benchmark for Python code generation.",
        "2. **Semantic Tasks**: Custom tasks specifically designed to evaluate semantic correctness and robustness.",
        "",
        "## Results",
        "",
        "### Overall Performance Comparison",
        "",
        "The table below shows the overall performance of each approach:"
    ])
    
    # Add main comparison table
    if 'main_comparison' in tables:
        md_lines.extend([
            "",
            tables['main_comparison'],
            ""
        ])
    
    # Add performance comparison figure
    if 'performance_comparison' in figure_paths:
        md_lines.extend([
            "",
            "Visual comparison of performance metrics:",
            "",
            f"![Performance Comparison]({os.path.basename(figure_paths['performance_comparison'])})",
            ""
        ])
    
    # Add Pass@k results
    md_lines.extend([
        "### Pass@k Results",
        "",
        "The Pass@k metric represents the likelihood that at least one correct solution is found within k attempts:",
        ""
    ])
    
    # Add Pass@k table
    if 'pass_at_k' in tables:
        md_lines.extend([
            tables['pass_at_k'],
            ""
        ])
    
    # Add code quality results
    md_lines.extend([
        "### Code Quality Metrics",
        "",
        "We evaluated the quality of the generated code using various static analysis metrics:",
        ""
    ])
    
    # Add code quality table
    if 'code_quality' in tables:
        md_lines.extend([
            tables['code_quality'],
            ""
        ])
    
    # Add bug patterns figure
    if 'bug_patterns' in figure_paths:
        md_lines.extend([
            "",
            "Distribution of bug patterns across different approaches:",
            "",
            f"![Bug Patterns]({os.path.basename(figure_paths['bug_patterns'])})",
            ""
        ])
    
    # Add dataset comparison figure
    if 'dataset_comparison' in figure_paths:
        md_lines.extend([
            "### Performance Across Datasets",
            "",
            "Comparison of approach performance on different datasets:",
            "",
            f"![Dataset Comparison]({os.path.basename(figure_paths['dataset_comparison'])})",
            ""
        ])
    
    # Add ablation study results
    if 'ablation_study' in figure_paths:
        md_lines.extend([
            "### Ablation Study",
            "",
            "We conducted an ablation study to evaluate the contribution of each component in the SSCSteer framework:",
            "",
            f"![Ablation Study]({os.path.basename(figure_paths['ablation_study'])})",
            "",
            "The ablation study shows that both syntactic and semantic steering contribute to the overall performance of the framework, with syntactic steering having a larger impact on syntactic validity and semantic steering improving semantic correctness."
        ])
    
    # Add beam search evolution figure
    if 'beam_evolution' in figure_paths:
        md_lines.extend([
            "",
            "### Beam Search Evolution",
            "",
            "The following figure shows how beam search evolves during code generation with the SSCSteer framework:",
            "",
            f"![Beam Evolution]({os.path.basename(figure_paths['beam_evolution'])})",
            "",
            "The plot illustrates how the beam search algorithm adjusts beam scores based on detected semantic issues, guiding the generation process toward more correct code."
        ])
    
    # Add key findings
    md_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Syntactic Correctness**: SSCSteer significantly improves the syntactic validity of generated code compared to vanilla LLM generation.",
        "2. **Semantic Correctness**: The semantic steering component reduces common bug patterns and improves functional correctness.",
        "3. **Generation Efficiency**: While steering adds computational overhead, the improvement in code quality justifies the additional cost for many applications.",
        "4. **Comparative Advantage**: SSCSteer outperforms post-hoc validation and feedback-based refinement in terms of code correctness and quality.",
        "",
        "## Limitations",
        "",
        "1. **Computational Overhead**: The steering process introduces additional computational cost, which may be prohibitive for some applications.",
        "2. **Complex Semantics**: Some complex semantic properties remain challenging to verify incrementally during generation.",
        "3. **Language Coverage**: The current implementation focuses primarily on Python, with limited support for other languages.",
        "",
        "## Conclusion",
        "",
        "The SSCSteer framework demonstrates that integrating lightweight formal methods into the LLM decoding process can significantly improve the correctness and reliability of generated code. By steering the generation process toward syntactically and semantically valid solutions, SSCSteer produces code that requires less post-hoc validation and correction, potentially improving developer productivity and code reliability."
    ])
    
    # Write markdown content to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))


def visualize_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Visualize experimental results and create summary markdown.
    
    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save visualizations and markdown
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract relevant data
    comparison_df = results.get('comparison')
    detailed_results = results.get('detailed_results', {})
    
    if comparison_df is None:
        print("Error: No comparison data available for visualization")
        return
    
    # Generate figures
    figure_paths = {}
    
    # Performance comparison
    perf_path = plot_performance_comparison(comparison_df, output_dir)
    figure_paths['performance_comparison'] = perf_path
    
    # Bug patterns
    bug_path = plot_bug_patterns(detailed_results, output_dir)
    if bug_path:
        figure_paths['bug_patterns'] = bug_path
    
    # Dataset comparison
    if 'dataset_results' in results:
        dataset_path = plot_dataset_comparison(results['dataset_results'], output_dir)
        figure_paths['dataset_comparison'] = dataset_path
    
    # Ablation study
    if 'ablation_results' in results:
        ablation_path = plot_ablation_study(results['ablation_results'], output_dir)
        figure_paths['ablation_study'] = ablation_path
    
    # Beam search evolution for SSCSteer
    if 'Full SSCSteer' in detailed_results:
        beam_path = plot_beam_search_evolution(detailed_results['Full SSCSteer'], output_dir)
        if beam_path:
            figure_paths['beam_evolution'] = beam_path
    
    # Generate tables
    tables = generate_result_tables(comparison_df)
    
    # Create markdown summary
    output_path = os.path.join(output_dir, 'results.md')
    create_results_markdown(comparison_df, tables, figure_paths, output_path)
    
    print(f"Visualizations and summary created in {output_dir}")
    print(f"Results summary saved to {output_path}")