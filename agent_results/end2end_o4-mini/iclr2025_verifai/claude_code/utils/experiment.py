"""
Utilities for running ContractGPT experiments.

This module provides utilities for running experiments with ContractGPT
and baseline methods.
"""

import os
import json
import time
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable

# Add parent directory to path to allow importing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.dsl_parser import parse_dsl
from models.contract_gpt import ContractGPT
from models.baselines import LLMOnly, VeCoGenLike, LLM4CodeLike


class Experiment:
    """Class for running experiments with ContractGPT and baselines."""
    
    def __init__(
        self,
        benchmark_dir: str,
        output_dir: str,
        target_language: str = "python",
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 5,
        temperature: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize an experiment.
        
        Args:
            benchmark_dir: Directory containing benchmark specifications.
            output_dir: Directory to save results.
            target_language: Target programming language.
            model_name: Name of the LLM to use.
            max_iterations: Maximum number of iterations for synthesis.
            temperature: Temperature for LLM generation.
            logger: Logger for recording progress.
        """
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.target_language = target_language
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.logger = logger or logging.getLogger("Experiment")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_benchmarks(self) -> List[Dict[str, Any]]:
        """
        Load benchmarks from the benchmark directory.
        
        Returns:
            List of benchmark dictionaries, each with "name" and "spec" keys.
        """
        benchmarks = []
        
        for filename in os.listdir(self.benchmark_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.benchmark_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    # Validate benchmark data
                    if "name" not in benchmark_data or "spec" not in benchmark_data:
                        self.logger.warning(f"Benchmark {filename} missing required fields")
                        continue
                    
                    benchmarks.append(benchmark_data)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load benchmark {filename}: {e}")
        
        return benchmarks
    
    def run_method(
        self, 
        method_name: str, 
        benchmarks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run a synthesis method on all benchmarks.
        
        Args:
            method_name: Name of the method to run.
            benchmarks: List of benchmark dictionaries.
            
        Returns:
            List of result dictionaries.
        """
        results = []
        
        for benchmark in benchmarks:
            self.logger.info(f"Running {method_name} on {benchmark['name']}")
            
            # Create method instance
            method = self._create_method_instance(method_name)
            
            # Run synthesis
            start_time = time.time()
            success, code, metrics = method.synthesize(benchmark["spec"])
            end_time = time.time()
            
            # Record result
            result = {
                "name": benchmark["name"],
                "method": method_name,
                "success": success,
                "code": code,
                "total_time": end_time - start_time,
                **metrics
            }
            
            results.append(result)
            
            # Save individual result
            self._save_result(result)
        
        return results
    
    def run_all_methods(self, methods: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all specified methods on all benchmarks.
        
        Args:
            methods: List of method names to run.
            
        Returns:
            Dictionary mapping method names to lists of result dictionaries.
        """
        benchmarks = self.load_benchmarks()
        
        self.logger.info(f"Loaded {len(benchmarks)} benchmarks")
        
        all_results = {}
        
        for method_name in methods:
            self.logger.info(f"Running method: {method_name}")
            results = self.run_method(method_name, benchmarks)
            all_results[method_name] = results
        
        # Save combined results
        self._save_all_results(all_results)
        
        return all_results
    
    def _create_method_instance(self, method_name: str) -> Any:
        """
        Create an instance of a synthesis method.
        
        Args:
            method_name: Name of the method.
            
        Returns:
            Instance of the synthesis method.
        """
        if method_name == "ContractGPT":
            return ContractGPT(
                target_language=self.target_language,
                model_name=self.model_name,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
                logger=self.logger
            )
        elif method_name == "LLMOnly":
            return LLMOnly(
                target_language=self.target_language,
                model_name=self.model_name,
                temperature=self.temperature,
                logger=self.logger
            )
        elif method_name == "VeCoGenLike":
            return VeCoGenLike(
                target_language=self.target_language,
                model_name=self.model_name,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
                logger=self.logger
            )
        elif method_name == "LLM4CodeLike":
            return LLM4CodeLike(
                target_language=self.target_language,
                model_name=self.model_name,
                temperature=self.temperature,
                logger=self.logger
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        Save an individual result to the output directory.
        
        Args:
            result: Result dictionary.
        """
        result_dir = os.path.join(self.output_dir, result["method"])
        os.makedirs(result_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(result_dir, f"{result['name']}_metrics.json")
        metrics_data = {k: v for k, v in result.items() if k != "code"}
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save code
        code_file = os.path.join(result_dir, f"{result['name']}_code.{self._get_file_ext()}")
        
        with open(code_file, 'w') as f:
            f.write(result["code"])
    
    def _save_all_results(self, all_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Save combined results to the output directory.
        
        Args:
            all_results: Dictionary mapping method names to lists of result dictionaries.
        """
        combined_file = os.path.join(self.output_dir, "all_results.json")
        
        # Create a simplified version of the results for JSON serialization
        simplified_results = {}
        
        for method_name, results in all_results.items():
            simplified_results[method_name] = []
            
            for result in results:
                simplified_result = {k: v for k, v in result.items() if k != "code"}
                simplified_results[method_name].append(simplified_result)
        
        with open(combined_file, 'w') as f:
            json.dump(simplified_results, f, indent=2)
    
    def _get_file_ext(self) -> str:
        """
        Get the file extension for the target language.
        
        Returns:
            File extension string.
        """
        if self.target_language.lower() == "python":
            return "py"
        elif self.target_language.lower() == "c":
            return "c"
        elif self.target_language.lower() == "rust":
            return "rs"
        else:
            return "txt"


def create_benchmark_json(
    name: str,
    spec: str,
    description: str = "",
    output_dir: str = "benchmarks",
    overwrite: bool = False
) -> str:
    """
    Create a benchmark JSON file.
    
    Args:
        name: Name of the benchmark.
        spec: Specification string in the DSL format.
        description: Description of the benchmark.
        output_dir: Directory to save the benchmark.
        overwrite: Whether to overwrite an existing benchmark.
        
    Returns:
        Path to the created JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"{name}.json")
    
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"Benchmark {name} already exists")
    
    # Try to parse the spec to validate it
    try:
        parse_dsl(spec)
    except Exception as e:
        raise ValueError(f"Invalid specification: {e}")
    
    benchmark_data = {
        "name": name,
        "spec": spec,
        "description": description
    }
    
    with open(file_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    return file_path