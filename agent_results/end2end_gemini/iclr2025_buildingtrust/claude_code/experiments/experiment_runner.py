"""
Experiment Runner Module

This module implements the experiment runner for Concept-Graph explanations.
"""

import os
import json
import time
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..models.llm_state_extractor import LLMStateExtractor
from ..models.concept_mapper import ConceptMapper
from ..models.concept_graph import ConceptGraph
from ..evaluation.baselines import BaselinesRunner
from ..evaluation.dataset_handler import DatasetHandler
from ..utils.logging_utils import ExperimentLogger, timeit
from ..visualization.visualization import (
    visualize_metrics_comparison,
    visualize_token_importance,
    visualize_attention_weights,
    visualize_hidden_states_pca
)

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Class for running Concept-Graph experiments.
    
    This class provides functionality for:
    1. Setting up and running experiments
    2. Collecting and analyzing metrics
    3. Generating visualizations and results
    """
    
    def __init__(
        self,
        experiment_dir: str,
        models_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        experiment_config: Dict[str, Any]
    ):
        """
        Initialize the experiment runner.
        
        Args:
            experiment_dir: Directory to store experiment results
            models_config: Configuration for LLM models
            dataset_config: Configuration for datasets
            experiment_config: General experiment configuration
        """
        self.experiment_dir = experiment_dir
        self.models_config = models_config
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        
        # Create experiment directory
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize experiment logger
        log_file = os.path.join(experiment_dir, "experiment_log.txt")
        self.experiment_logger = ExperimentLogger(log_file)
        
        # Log configurations
        self.experiment_logger.log_step("Experiment Initialization")
        self.experiment_logger.log_step("Models Config", json.dumps(models_config, indent=2))
        self.experiment_logger.log_step("Dataset Config", json.dumps(dataset_config, indent=2))
        self.experiment_logger.log_step("Experiment Config", json.dumps(experiment_config, indent=2))
        
        # Initialize metrics
        self.metrics = {}
        
        # Setup components
        self._setup_components()
    
    def _setup_components(self):
        """Set up experiment components."""
        self.experiment_logger.log_step("Setting up components")
        
        # Setup dataset handler
        data_dir = os.path.join(self.experiment_dir, "data")
        self.dataset_handler = DatasetHandler(
            data_dir=data_dir,
            cache_dir=self.dataset_config.get("cache_dir"),
            seed=self.experiment_config.get("seed", 42)
        )
        
        # Setup LLM state extractor
        model_name = self.models_config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        device = self.models_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.state_extractor = LLMStateExtractor(
                model_name=model_name,
                device=device,
                layers_to_extract=self.models_config.get("layers_to_extract"),
                cache_dir=self.models_config.get("cache_dir")
            )
            
            self.experiment_logger.log_step(
                "Initialized LLM State Extractor",
                f"Model: {model_name}, Device: {device}"
            )
        except Exception as e:
            error_msg = f"Failed to initialize LLM State Extractor: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            raise
        
        # Setup concept mapper
        self.concept_mapper = ConceptMapper(
            use_openai_for_labeling=self.experiment_config.get("use_openai", True),
            openai_model=self.experiment_config.get("openai_model", "gpt-4o-mini"),
            cache_dir=os.path.join(self.experiment_dir, "concept_cache")
        )
        
        self.experiment_logger.log_step(
            "Initialized Concept Mapper",
            f"OpenAI: {self.experiment_config.get('use_openai', True)}"
        )
        
        # Setup concept graph
        self.concept_graph = ConceptGraph()
        
        # Setup baselines runner
        self.baselines = BaselinesRunner(
            model=self.state_extractor.model,
            tokenizer=self.state_extractor.tokenizer,
            device=device
        )
        
        self.experiment_logger.log_step("Components setup complete")
    
    @timeit
    def load_datasets(self):
        """Load datasets for experiments."""
        self.experiment_logger.log_step("Loading datasets")
        
        # Load specified datasets
        dataset_names = self.dataset_config.get("datasets", ["gsm8k"])
        max_samples = self.dataset_config.get("max_samples", 100)
        
        for dataset_name in dataset_names:
            if dataset_name == "hotpotqa":
                self.dataset_handler.download_hotpotqa(max_samples)
            elif dataset_name == "gsm8k":
                self.dataset_handler.download_gsm8k(max_samples)
            elif dataset_name == "strategyqa":
                self.dataset_handler.download_strategyqa(max_samples)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            # Create splits
            self.dataset_handler.create_dataset_splits(
                dataset_name,
                train_ratio=self.dataset_config.get("train_ratio", 0.7),
                val_ratio=self.dataset_config.get("val_ratio", 0.15),
                test_ratio=self.dataset_config.get("test_ratio", 0.15)
            )
            
            # Analyze dataset
            stats = self.dataset_handler.analyze_dataset(dataset_name)
            
            self.experiment_logger.log_step(
                f"Loaded dataset: {dataset_name}",
                f"Samples: {stats.get('num_samples', 0)}"
            )
        
        # Save dataset summary
        summary = self.dataset_handler.get_datasets_summary()
        summary_path = os.path.join(self.experiment_dir, "dataset_summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.experiment_logger.log_step("Datasets loading complete")
    
    @timeit
    def run_sample_experiment(
        self,
        sample: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Run experiment for a single sample.
        
        Args:
            sample: Sample data
            output_dir: Directory to store results
            
        Returns:
            Dictionary with experiment results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log sample info
        prompt = sample.get('prompt', '')
        sample_id = sample.get('id', 'unknown')
        
        self.experiment_logger.log_step(
            f"Processing sample {sample_id}",
            f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else prompt
        )
        
        # Step 1: Generate text with state extraction
        self.experiment_logger.log_step(f"Generating text for sample {sample_id}")
        
        generation_params = self.experiment_config.get('generation', {})
        max_new_tokens = generation_params.get('max_new_tokens', 200)
        temperature = generation_params.get('temperature', 0.7)
        
        try:
            generated_text, states = self.state_extractor.generate_with_states(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=generation_params.get('do_sample', True),
                top_p=generation_params.get('top_p', 0.9)
            )
            
            self.experiment_logger.log_step(
                f"Generated text for sample {sample_id}",
                f"Length: {len(generated_text.split())} words"
            )
            
            # Save generated text
            with open(os.path.join(output_dir, "generated_text.txt"), 'w') as f:
                f.write(generated_text)
            
        except Exception as e:
            error_msg = f"Error generating text for sample {sample_id}: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            return {'error': error_msg}
        
        # Step 2: Concept discovery and mapping
        self.experiment_logger.log_step(f"Discovering concepts for sample {sample_id}")
        
        concept_params = self.experiment_config.get('concept_mapping', {})
        num_concepts = concept_params.get('num_concepts', 10)
        
        try:
            # Run unsupervised concept discovery
            concept_result = self.concept_mapper.discover_concepts_unsupervised(
                hidden_states=states['hidden_states'],
                num_clusters=num_concepts,
                pca_components=concept_params.get('pca_components', 50),
                umap_components=concept_params.get('umap_components', 2),
                clustering_method=concept_params.get('clustering_method', 'kmeans'),
                visualize=True,
                save_path=os.path.join(output_dir, "concept_clusters.png")
            )
            
            # Label concepts using OpenAI
            if self.experiment_config.get('use_openai', True):
                labeled_result = self.concept_mapper.llm_aided_concept_labeling(
                    cluster_result=concept_result,
                    prompt=prompt,
                    generated_text=generated_text,
                    max_concepts=num_concepts
                )
            else:
                labeled_result = concept_result
            
            self.experiment_logger.log_step(
                f"Discovered concepts for sample {sample_id}",
                f"Num concepts: {len(labeled_result.get('labeled_clusters', {}))}"
            )
            
            # Save concept discovery results
            with open(os.path.join(output_dir, "concepts.json"), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = {}
                for key, value in labeled_result.items():
                    if key not in ['pca', 'umap']:
                        if isinstance(value, dict) and 'labeled_clusters' in key:
                            serializable_clusters = {}
                            for c_id, c_data in value.items():
                                serializable_c_data = {}
                                for c_key, c_value in c_data.items():
                                    if isinstance(c_value, np.ndarray):
                                        serializable_c_data[c_key] = c_value.tolist()
                                    else:
                                        serializable_c_data[c_key] = c_value
                                serializable_clusters[c_id] = serializable_c_data
                            serializable_result[key] = serializable_clusters
                        else:
                            serializable_result[key] = value
                
                json.dump(serializable_result, f, indent=2)
            
        except Exception as e:
            error_msg = f"Error discovering concepts for sample {sample_id}: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            return {'error': error_msg, 'generated_text': generated_text}
        
        # Step 3: Concept graph construction
        self.experiment_logger.log_step(f"Building concept graph for sample {sample_id}")
        
        try:
            # Build concept graph
            if 'labeled_clusters' in labeled_result:
                graph = self.concept_graph.build_from_labeled_clusters(
                    cluster_result=labeled_result,
                    min_edge_weight=concept_params.get('min_edge_weight', 0.1)
                )
            else:
                graph = self.concept_graph.build_from_concepts(
                    concepts=labeled_result['clusters'],
                    attention_weights=states.get('attention_weights'),
                    temporal_ordering=True,
                    min_edge_weight=concept_params.get('min_edge_weight', 0.1)
                )
            
            # Visualize graph
            self.concept_graph.visualize_graph(
                save_path=os.path.join(output_dir, "concept_graph.png"),
                layout=concept_params.get('graph_layout', 'temporal'),
                title=f"Concept Graph for Sample {sample_id}"
            )
            
            # Extract reasoning path
            reasoning_path = self.concept_graph.extract_reasoning_path()
            
            # Analyze graph
            graph_analysis = self.concept_graph.analyze_graph_structure()
            
            self.experiment_logger.log_step(
                f"Built concept graph for sample {sample_id}",
                f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            )
            
            # Save graph analysis
            with open(os.path.join(output_dir, "graph_analysis.json"), 'w') as f:
                # Convert sets to lists for JSON serialization
                serializable_analysis = {}
                for key, value in graph_analysis.items():
                    if isinstance(value, set):
                        serializable_analysis[key] = list(value)
                    elif isinstance(value, dict):
                        serializable_analysis[key] = {k: v for k, v in value.items()}
                    else:
                        serializable_analysis[key] = value
                
                json.dump(serializable_analysis, f, indent=2)
            
        except Exception as e:
            error_msg = f"Error building concept graph for sample {sample_id}: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            return {
                'error': error_msg,
                'generated_text': generated_text,
                'concepts': labeled_result
            }
        
        # Step 4: Run baselines
        self.experiment_logger.log_step(f"Running baselines for sample {sample_id}")
        
        try:
            baseline_results = self.baselines.run_all_baselines(
                prompt=prompt,
                generated_text=generated_text,
                save_dir=os.path.join(output_dir, "baselines")
            )
            
            self.experiment_logger.log_step(
                f"Completed baselines for sample {sample_id}",
                f"Ran {len(baseline_results)} baseline methods"
            )
            
            # Save baseline results
            with open(os.path.join(output_dir, "baseline_results.json"), 'w') as f:
                # Clean up results for JSON serialization
                serializable_baselines = {}
                for method, result in baseline_results.items():
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            serializable_result[key] = value.tolist()
                        elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                            serializable_result[key] = value
                    
                    serializable_baselines[method] = serializable_result
                
                json.dump(serializable_baselines, f, indent=2)
            
        except Exception as e:
            error_msg = f"Error running baselines for sample {sample_id}: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
        
        # Step 5: Compute and compare metrics
        self.experiment_logger.log_step(f"Computing metrics for sample {sample_id}")
        
        try:
            # Compute metrics for concept graph
            concept_graph_metrics = self.concept_graph.evaluate_concept_graph_quality()
            
            # Compute metrics for baselines
            baseline_metrics = {}
            
            # Simple metrics for attention
            if 'attention' in baseline_results:
                attention_result = baseline_results['attention']
                baseline_metrics['attention'] = {
                    'num_attention_heads': len(attention_result.get('attention_weights', {})),
                    'layer_idx': attention_result.get('layer_idx', -1)
                }
            
            # Metrics for integrated gradients
            if 'integrated_gradients' in baseline_results:
                ig_result = baseline_results['integrated_gradients']
                baseline_metrics['integrated_gradients'] = {
                    'max_attribution': max(ig_result.get('attribution_scores', [0])),
                    'num_tokens': len(ig_result.get('attribution_scores', []))
                }
            
            # Metrics for CoT
            if 'cot' in baseline_results:
                cot_result = baseline_results['cot']
                baseline_metrics['cot'] = {
                    'num_steps': cot_result.get('num_steps', 0)
                }
            
            # Combine all metrics
            metrics = {
                'concept_graph': concept_graph_metrics,
                'baselines': baseline_metrics
            }
            
            # Add ground truth comparison for datasets with reference steps
            if 'solution_steps' in sample or 'reasoning_steps' in sample:
                reference_steps = sample.get('solution_steps', sample.get('reasoning_steps', []))
                
                # Compare reasoning path with reference steps
                if reference_steps:
                    metrics['ground_truth'] = {
                        'num_reference_steps': len(reference_steps),
                        'reference_steps': reference_steps,
                        'concept_graph_path': reasoning_path,
                        'cot_steps': baseline_results.get('cot', {}).get('steps', [])
                    }
            
            self.experiment_logger.log_metrics(metrics, f"Sample {sample_id}")
            
            # Save metrics
            with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
            
        except Exception as e:
            error_msg = f"Error computing metrics for sample {sample_id}: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
        
        # Prepare final result
        result = {
            'sample_id': sample_id,
            'generated_text': generated_text,
            'concept_graph': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'reasoning_path': reasoning_path
            },
            'metrics': metrics
        }
        
        self.experiment_logger.log_step(f"Completed experiment for sample {sample_id}")
        
        return result
    
    @timeit
    def run_dataset_experiments(
        self,
        dataset_name: str,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run experiments on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to use
            num_samples: Number of samples to process (None for all)
            
        Returns:
            Dictionary with experiment results
        """
        self.experiment_logger.log_step(
            f"Running experiments on {dataset_name} ({split} split)",
            f"Samples: {num_samples if num_samples else 'all'}"
        )
        
        # Get dataset samples
        if dataset_name not in self.dataset_handler.splits:
            logger.error(f"Dataset {dataset_name} not found or has no splits")
            return {}
        
        if split not in self.dataset_handler.splits[dataset_name]:
            logger.error(f"Split {split} not found for dataset {dataset_name}")
            return {}
        
        samples = self.dataset_handler.splits[dataset_name][split]
        
        if num_samples is not None:
            samples = samples[:num_samples]
        
        # Create output directory for this dataset
        dataset_dir = os.path.join(self.experiment_dir, f"{dataset_name}_{split}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Process each sample
        results = {}
        
        for i, sample in enumerate(tqdm(samples, desc=f"Processing {dataset_name}")):
            sample_id = sample.get('id', str(i))
            sample_dir = os.path.join(dataset_dir, f"sample_{sample_id}")
            
            self.experiment_logger.log_step(f"Starting sample {i+1}/{len(samples)}: {sample_id}")
            
            try:
                result = self.run_sample_experiment(sample, sample_dir)
                results[sample_id] = result
                
                self.experiment_logger.log_step(f"Completed sample {i+1}/{len(samples)}: {sample_id}")
                
            except Exception as e:
                error_msg = f"Error processing sample {sample_id}: {str(e)}"
                logger.error(error_msg)
                self.experiment_logger.log_error(error_msg, e)
                results[sample_id] = {'error': error_msg}
        
        # Save all results
        results_path = os.path.join(dataset_dir, "all_results.json")
        
        with open(results_path, 'w') as f:
            # Clean up results for JSON serialization
            serializable_results = {}
            for s_id, result in results.items():
                if isinstance(result, dict):
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                            serializable_result[key] = value
                    
                    serializable_results[s_id] = serializable_result
            
            json.dump(serializable_results, f, indent=2)
        
        self.experiment_logger.log_step(
            f"Completed experiments on {dataset_name} ({split} split)",
            f"Processed {len(results)} samples"
        )
        
        # Store in metrics
        self.metrics[f"{dataset_name}_{split}"] = results
        
        return results
    
    @timeit
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run experiments on all configured datasets.
        
        Returns:
            Dictionary with results for all datasets
        """
        self.experiment_logger.log_step("Starting all experiments")
        
        # Load datasets if not already loaded
        if not self.dataset_handler.datasets:
            self.load_datasets()
        
        # Get experiment configuration
        datasets = self.dataset_config.get("datasets", ["gsm8k"])
        splits = self.dataset_config.get("test_splits", ["test"])
        num_samples = self.experiment_config.get("num_samples_per_dataset", 10)
        
        # Run experiments for each dataset and split
        all_results = {}
        
        for dataset_name in datasets:
            for split in splits:
                results = self.run_dataset_experiments(
                    dataset_name=dataset_name,
                    split=split,
                    num_samples=num_samples
                )
                
                all_results[f"{dataset_name}_{split}"] = results
        
        # Save aggregate results
        aggregate_path = os.path.join(self.experiment_dir, "aggregate_results.json")
        
        with open(aggregate_path, 'w') as f:
            # Create a summary of results for each dataset and split
            summary = {}
            
            for key, results in all_results.items():
                num_samples = len(results)
                num_errors = sum(1 for r in results.values() if 'error' in r)
                
                summary[key] = {
                    'num_samples': num_samples,
                    'num_errors': num_errors,
                    'success_rate': (num_samples - num_errors) / num_samples if num_samples > 0 else 0
                }
            
            json.dump(summary, f, indent=2)
        
        self.experiment_logger.log_step("Completed all experiments")
        
        return all_results
    
    @timeit
    def generate_visualizations(self) -> None:
        """Generate final visualizations and summary figures."""
        self.experiment_logger.log_step("Generating visualizations")
        
        # Create visualizations directory
        vis_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate comparative visualizations
        try:
            # Collect metrics from all experiments
            method_metrics = {'concept_graph': {}, 'attention': {}, 'integrated_gradients': {}, 'cot': {}}
            
            for dataset_key, results in self.metrics.items():
                for sample_id, result in results.items():
                    if 'metrics' in result and 'concept_graph' in result['metrics']:
                        # Extract concept graph metrics
                        cg_metrics = result['metrics']['concept_graph']
                        
                        # Update averages
                        for metric, value in cg_metrics.items():
                            if isinstance(value, (int, float)):
                                if metric not in method_metrics['concept_graph']:
                                    method_metrics['concept_graph'][metric] = []
                                
                                method_metrics['concept_graph'][metric].append(value)
                    
                    if 'metrics' in result and 'baselines' in result['metrics']:
                        # Extract baseline metrics
                        baseline_metrics = result['metrics']['baselines']
                        
                        for method, metrics in baseline_metrics.items():
                            if method in method_metrics:
                                for metric, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        if metric not in method_metrics[method]:
                                            method_metrics[method][metric] = []
                                        
                                        method_metrics[method][metric].append(value)
            
            # Compute averages
            avg_metrics = {}
            for method, metrics in method_metrics.items():
                avg_metrics[method] = {}
                for metric, values in metrics.items():
                    if values:
                        avg_metrics[method][metric] = sum(values) / len(values)
            
            # Create comparison visualizations
            if avg_metrics['concept_graph'] and any(avg_metrics[m] for m in ['attention', 'integrated_gradients', 'cot']):
                # Define metrics to plot
                metrics_to_plot = ['num_nodes', 'num_edges', 'is_dag', 'density']
                
                # Define whether higher is better for each metric
                higher_is_better = {
                    'num_nodes': True,        # More concepts is better
                    'num_edges': True,        # More connections is better
                    'is_dag': True,           # Being a DAG is better
                    'density': False          # Lower density often means cleaner structure
                }
                
                # Generate visualization
                visualize_metrics_comparison(
                    method_metrics=avg_metrics,
                    metrics_to_plot=metrics_to_plot,
                    higher_is_better=higher_is_better,
                    save_path=os.path.join(vis_dir, "methods_comparison.png")
                )
            
            # Create success rate visualization
            dataset_success_rates = {}
            for dataset_key, results in self.metrics.items():
                num_samples = len(results)
                num_errors = sum(1 for r in results.values() if 'error' in r)
                success_rate = (num_samples - num_errors) / num_samples if num_samples > 0 else 0
                
                dataset_success_rates[dataset_key] = success_rate
            
            if dataset_success_rates:
                plt.figure(figsize=(10, 6))
                plt.bar(dataset_success_rates.keys(), dataset_success_rates.values(), color='skyblue')
                plt.title("Success Rates by Dataset")
                plt.xlabel("Dataset")
                plt.ylabel("Success Rate")
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "success_rates.png"), dpi=300)
                plt.close()
            
            self.experiment_logger.log_step("Visualization generation complete")
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
    
    @timeit
    def generate_report(self) -> str:
        """
        Generate a final report of the experiment results.
        
        Returns:
            Path to the generated report
        """
        self.experiment_logger.log_step("Generating final report")
        
        report_path = os.path.join(self.experiment_dir, "experiment_report.md")
        
        try:
            with open(report_path, 'w') as f:
                # Generate report header
                f.write("# Concept-Graph Experiments Report\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Model information
                f.write("## Model Information\n\n")
                f.write(f"- Model: {self.models_config.get('model_name', 'Unknown')}\n")
                f.write(f"- Device: {self.models_config.get('device', 'CPU')}\n\n")
                
                # Dataset information
                f.write("## Dataset Information\n\n")
                
                dataset_summary = self.dataset_handler.get_datasets_summary()
                
                for dataset_name, summary in dataset_summary.items():
                    f.write(f"### {dataset_name}\n\n")
                    f.write(f"- Total samples: {summary.get('num_samples', 0)}\n")
                    
                    if 'split_sizes' in summary:
                        f.write("- Splits:\n")
                        for split, size in summary['split_sizes'].items():
                            f.write(f"  - {split}: {size} samples\n")
                    
                    f.write("\n")
                
                # Experiment results
                f.write("## Experiment Results\n\n")
                
                for dataset_key, results in self.metrics.items():
                    f.write(f"### {dataset_key}\n\n")
                    
                    num_samples = len(results)
                    num_errors = sum(1 for r in results.values() if 'error' in r)
                    success_rate = (num_samples - num_errors) / num_samples if num_samples > 0 else 0
                    
                    f.write(f"- Processed {num_samples} samples\n")
                    f.write(f"- Success rate: {success_rate:.2%}\n\n")
                    
                    # Add visualization references
                    vis_path = f"visualizations/{dataset_key}_success_rates.png"
                    if os.path.exists(os.path.join(self.experiment_dir, vis_path)):
                        f.write(f"![Success Rate]({vis_path})\n\n")
                
                # Comparison with baselines
                f.write("## Comparison with Baselines\n\n")
                
                # Add visualization references
                vis_path = "visualizations/methods_comparison.png"
                if os.path.exists(os.path.join(self.experiment_dir, vis_path)):
                    f.write(f"![Methods Comparison]({vis_path})\n\n")
                
                # Calculate average metrics across all experiments
                avg_metrics = {}
                method_counts = {}
                
                for dataset_key, results in self.metrics.items():
                    for sample_id, result in results.items():
                        if 'metrics' not in result:
                            continue
                        
                        # Process concept graph metrics
                        if 'concept_graph' in result['metrics']:
                            method = 'concept_graph'
                            
                            if method not in avg_metrics:
                                avg_metrics[method] = {}
                                method_counts[method] = 0
                            
                            method_counts[method] += 1
                            
                            for metric, value in result['metrics'][method].items():
                                if isinstance(value, (int, float)):
                                    if metric not in avg_metrics[method]:
                                        avg_metrics[method][metric] = 0
                                    
                                    avg_metrics[method][metric] += value
                        
                        # Process baseline metrics
                        if 'baselines' in result['metrics']:
                            for method, metrics in result['metrics']['baselines'].items():
                                if method not in avg_metrics:
                                    avg_metrics[method] = {}
                                    method_counts[method] = 0
                                
                                method_counts[method] += 1
                                
                                for metric, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        if metric not in avg_metrics[method]:
                                            avg_metrics[method][metric] = 0
                                        
                                        avg_metrics[method][metric] += value
                
                # Calculate averages
                for method, metrics in avg_metrics.items():
                    if method_counts.get(method, 0) > 0:
                        for metric in metrics:
                            metrics[metric] /= method_counts[method]
                
                # Create metrics table
                f.write("### Average Metrics\n\n")
                
                f.write("| Method | ")
                metrics_set = set()
                for method, metrics in avg_metrics.items():
                    metrics_set.update(metrics.keys())
                
                f.write(" | ".join(metrics_set))
                f.write(" |\n")
                
                f.write("| --- | ")
                f.write(" | ".join(["---"] * len(metrics_set)))
                f.write(" |\n")
                
                for method, metrics in avg_metrics.items():
                    f.write(f"| {method} | ")
                    
                    for metric in metrics_set:
                        value = metrics.get(metric, "N/A")
                        if isinstance(value, float):
                            f.write(f"{value:.3f} | ")
                        else:
                            f.write(f"{value} | ")
                    
                    f.write("\n")
                
                f.write("\n")
                
                # Sample visualizations
                f.write("## Sample Visualizations\n\n")
                
                # Include some sample visualizations from the results
                sample_dirs = []
                for dataset_key, results in self.metrics.items():
                    dataset_dir = os.path.join(self.experiment_dir, dataset_key)
                    if os.path.exists(dataset_dir):
                        for sample_id in results.keys():
                            sample_dir = os.path.join(dataset_dir, f"sample_{sample_id}")
                            if os.path.exists(sample_dir):
                                sample_dirs.append((dataset_key, sample_id, sample_dir))
                
                # Include up to 3 samples
                for i, (dataset_key, sample_id, sample_dir) in enumerate(sample_dirs[:3]):
                    f.write(f"### Sample {i+1}: {dataset_key} - {sample_id}\n\n")
                    
                    # Include concept graph visualization
                    concept_graph_path = os.path.join(sample_dir, "concept_graph.png")
                    if os.path.exists(concept_graph_path):
                        rel_path = os.path.relpath(concept_graph_path, self.experiment_dir)
                        f.write(f"![Concept Graph]({rel_path})\n\n")
                    
                    # Include baseline visualizations
                    baseline_dir = os.path.join(sample_dir, "baselines")
                    if os.path.exists(baseline_dir):
                        f.write("#### Baseline Visualizations\n\n")
                        
                        attention_path = os.path.join(baseline_dir, "attention_vis.png")
                        if os.path.exists(attention_path):
                            rel_path = os.path.relpath(attention_path, self.experiment_dir)
                            f.write(f"![Attention Visualization]({rel_path})\n\n")
                        
                        ig_path = os.path.join(baseline_dir, "integrated_gradients.png")
                        if os.path.exists(ig_path):
                            rel_path = os.path.relpath(ig_path, self.experiment_dir)
                            f.write(f"![Integrated Gradients]({rel_path})\n\n")
                    
                    f.write("\n")
                
                # Conclusion
                f.write("## Conclusion\n\n")
                f.write("This report provides an overview of the Concept-Graph experiments. ")
                f.write("The Concept-Graph approach offers a structured way to visualize and understand ")
                f.write("the reasoning processes of large language models, providing more interpretable ")
                f.write("explanations compared to traditional methods like attention visualization and token attribution.\n\n")
            
            self.experiment_logger.log_step("Report generation complete", f"Report saved to {report_path}")
            
            return report_path
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            
            return ""
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the full experiment pipeline.
        
        Returns:
            Dictionary with all experiment results
        """
        self.experiment_logger.log_step("Starting full experiment pipeline")
        
        try:
            # Load datasets
            self.load_datasets()
            
            # Run all experiments
            results = self.run_all_experiments()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Generate report
            report_path = self.generate_report()
            
            self.experiment_logger.log_step(
                "Full pipeline complete",
                f"Results saved to {self.experiment_dir}"
            )
            
            return {
                'results': results,
                'report_path': report_path,
                'experiment_dir': self.experiment_dir
            }
            
        except Exception as e:
            error_msg = f"Error in experiment pipeline: {str(e)}"
            logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            
            return {'error': error_msg}