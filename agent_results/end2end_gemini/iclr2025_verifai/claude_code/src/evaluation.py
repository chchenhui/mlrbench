"""
Evaluation metrics and testing framework for the SSCSteer experiment.

This module provides functions to evaluate the quality of generated code
and compare different code generation approaches.
"""

import ast
import os
import re
import time
import subprocess
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from pylint import epylint as lint
import flake8.api.legacy as flake8

# Function to check syntactic validity of Python code
def is_syntactically_valid(code: str) -> bool:
    """
    Check if Python code is syntactically valid.
    
    Args:
        code: Python code to check
        
    Returns:
        True if the code is syntactically valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# Function to count syntax errors in Python code
def count_syntax_errors(code: str) -> int:
    """
    Count the number of syntax errors in Python code.
    
    Args:
        code: Python code to check
        
    Returns:
        Number of syntax errors
    """
    try:
        ast.parse(code)
        return 0
    except SyntaxError:
        # Run the code through pylint to count errors
        with open("temp_code.py", "w") as f:
            f.write(code)
        
        (pylint_stdout, _) = lint.py_run("temp_code.py", return_std=True)
        output = pylint_stdout.getvalue()
        
        # Clean up temporary file
        if os.path.exists("temp_code.py"):
            os.remove("temp_code.py")
        
        # Count syntax error messages
        error_count = output.count("syntax-error")
        return max(1, error_count)  # At least 1 error if we got here


# Function to evaluate code against test cases
def evaluate_code_against_tests(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate code against a set of test cases.
    
    Args:
        code: Python code to evaluate
        test_cases: List of test case dictionaries, each with 'input' and 'expected' fields
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "total_tests": len(test_cases),
        "passing_tests": 0,
        "failing_tests": 0,
        "execution_errors": 0,
        "pass_rate": 0.0,
        "test_results": []
    }
    
    # Check if code is syntactically valid
    if not is_syntactically_valid(code):
        # If code isn't valid, all tests fail
        results["failing_tests"] = len(test_cases)
        results["test_results"] = [{"result": "syntax_error"} for _ in test_cases]
        return results
    
    # Write code to temporary file
    with open("temp_code.py", "w") as f:
        f.write(code)
    
    # Execute each test case
    for i, test_case in enumerate(test_cases):
        test_input = test_case["input"]
        expected_output = test_case["expected"]
        
        # Prepare test harness
        test_harness = f"""
import temp_code

try:
    result = {test_input}
    print(repr(result))
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{str(e)}}")
"""
        
        with open("test_harness.py", "w") as f:
            f.write(test_harness)
        
        # Execute test
        try:
            output = subprocess.check_output(
                ["python", "test_harness.py"], 
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5  # Timeout after 5 seconds
            ).strip()
            
            # Check if there was an error
            if output.startswith("ERROR:"):
                results["execution_errors"] += 1
                results["test_results"].append({
                    "test_idx": i,
                    "result": "execution_error",
                    "error": output
                })
            elif output == repr(expected_output):
                # Test passed
                results["passing_tests"] += 1
                results["test_results"].append({
                    "test_idx": i,
                    "result": "pass"
                })
            else:
                # Test failed
                results["failing_tests"] += 1
                results["test_results"].append({
                    "test_idx": i,
                    "result": "fail",
                    "expected": repr(expected_output),
                    "actual": output
                })
        except subprocess.TimeoutExpired:
            # Execution timed out
            results["execution_errors"] += 1
            results["test_results"].append({
                "test_idx": i,
                "result": "timeout"
            })
        except subprocess.CalledProcessError as e:
            # Execution failed
            results["execution_errors"] += 1
            results["test_results"].append({
                "test_idx": i,
                "result": "execution_error",
                "error": e.output if hasattr(e, 'output') else str(e)
            })
    
    # Clean up temporary files
    if os.path.exists("temp_code.py"):
        os.remove("temp_code.py")
    if os.path.exists("test_harness.py"):
        os.remove("test_harness.py")
    
    # Calculate pass rate
    results["pass_rate"] = results["passing_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0.0
    
    return results


# Function to calculate Pass@k metric
def calculate_pass_at_k(results: List[Dict[str, Any]], k: int = 1) -> float:
    """
    Calculate the Pass@k metric for a list of evaluation results.
    
    Args:
        results: List of evaluation result dictionaries, each with a 'pass_rate' field
        k: The k value for Pass@k
        
    Returns:
        Pass@k score
    """
    if not results or k <= 0:
        return 0.0
    
    # Sort results by pass rate, descending
    sorted_results = sorted(results, key=lambda x: x["pass_rate"], reverse=True)
    
    # Count problems with at least one solution that passes all tests
    passing_problems = 0
    current_problem = None
    problem_passing = False
    
    for i, result in enumerate(sorted_results[:k]):
        if result["problem_id"] != current_problem:
            # New problem
            if problem_passing:
                passing_problems += 1
            
            current_problem = result["problem_id"]
            problem_passing = result["pass_rate"] == 1.0
        else:
            # Same problem
            problem_passing = problem_passing or result["pass_rate"] == 1.0
    
    # Don't forget the last problem
    if problem_passing:
        passing_problems += 1
    
    # Calculate Pass@k
    return passing_problems / len(set(r["problem_id"] for r in results))


# Function to calculate code quality metrics using static analyzers
def calculate_code_quality_metrics(code: str) -> Dict[str, Any]:
    """
    Calculate code quality metrics using static analyzers.
    
    Args:
        code: Python code to analyze
        
    Returns:
        Dictionary with code quality metrics
    """
    metrics = {
        "is_valid": is_syntactically_valid(code),
        "pylint_score": 0.0,
        "flake8_violations": 0,
        "cyclomatic_complexity": 0,
        "bug_patterns": {
            "null_dereference": 0,
            "uninitialized_variable": 0,
            "division_by_zero": 0,
            "index_out_of_bounds": 0,
            "resource_leak": 0
        },
        "cognitive_complexity": 0
    }
    
    # If code is not valid, return basic metrics
    if not metrics["is_valid"]:
        return metrics
    
    # Write code to temporary file
    with open("temp_code.py", "w") as f:
        f.write(code)
    
    # Run pylint
    try:
        (pylint_stdout, _) = lint.py_run("temp_code.py --score=y", return_std=True)
        output = pylint_stdout.getvalue()
        
        # Extract score
        score_match = re.search(r"Your code has been rated at ([-\d.]+)/10", output)
        if score_match:
            metrics["pylint_score"] = float(score_match.group(1))
            
        # Count bug patterns
        metrics["bug_patterns"]["null_dereference"] = output.count("NoneType") + output.count("none-value")
        metrics["bug_patterns"]["uninitialized_variable"] = output.count("undefined-variable")
        metrics["bug_patterns"]["division_by_zero"] = output.count("division-by-zero")
        metrics["bug_patterns"]["index_out_of_bounds"] = output.count("index-error")
        metrics["bug_patterns"]["resource_leak"] = output.count("resource-leak")
    except Exception as e:
        print(f"Error running pylint: {e}")
    
    # Run flake8
    try:
        # Assume user has flake8 installed
        flake8_style_guide = flake8.get_style_guide()
        report = flake8_style_guide.check_files(["temp_code.py"])
        metrics["flake8_violations"] = report.total_errors
    except Exception as e:
        print(f"Error running flake8: {e}")
    
    # Calculate cyclomatic complexity
    try:
        # Parse the code
        parsed_code = ast.parse(code)
        
        # Count branches
        branches = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                nonlocal branches
                branches += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                nonlocal branches
                branches += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                nonlocal branches
                branches += 1
                self.generic_visit(node)
                
            def visit_Try(self, node):
                nonlocal branches
                branches += 1 + len(node.handlers)
                self.generic_visit(node)
        
        ComplexityVisitor().visit(parsed_code)
        metrics["cyclomatic_complexity"] = branches + 1  # Base complexity + branches
    except Exception as e:
        print(f"Error calculating cyclomatic complexity: {e}")
    
    # Clean up temporary file
    if os.path.exists("temp_code.py"):
        os.remove("temp_code.py")
    
    return metrics


# Function to compare different approaches
def compare_approaches(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Compare different approaches based on evaluation results.
    
    Args:
        results: Dictionary mapping approach names to lists of evaluation results
        
    Returns:
        DataFrame with comparison results
    """
    comparison = {
        "approach": [],
        "syntactic_validity": [],
        "pass_rate": [],
        "pass_at_1": [],
        "pass_at_3": [],
        "pass_at_5": [],
        "pylint_score": [],
        "flake8_violations": [],
        "cyclomatic_complexity": [],
        "bug_density": [],
        "generation_time": []
    }
    
    for approach, approach_results in results.items():
        comparison["approach"].append(approach)
        
        # Calculate metrics
        syntactic_validity = np.mean([r.get("is_valid", False) for r in approach_results])
        pass_rate = np.mean([r.get("pass_rate", 0.0) for r in approach_results])
        pass_at_1 = calculate_pass_at_k(approach_results, k=1)
        pass_at_3 = calculate_pass_at_k(approach_results, k=3)
        pass_at_5 = calculate_pass_at_k(approach_results, k=5)
        pylint_score = np.mean([r.get("pylint_score", 0.0) for r in approach_results])
        flake8_violations = np.mean([r.get("flake8_violations", 0) for r in approach_results])
        cyclomatic_complexity = np.mean([r.get("cyclomatic_complexity", 0) for r in approach_results])
        
        # Calculate bug density (bugs per 1000 lines of code)
        total_bugs = sum(sum(r.get("bug_patterns", {}).values()) for r in approach_results)
        total_lines = sum(len(r.get("code", "").split("\n")) for r in approach_results)
        bug_density = (total_bugs / total_lines) * 1000 if total_lines > 0 else 0
        
        # Calculate average generation time
        generation_time = np.mean([r.get("metrics", {}).get("generation_time", 0.0) for r in approach_results])
        
        # Add to comparison
        comparison["syntactic_validity"].append(syntactic_validity)
        comparison["pass_rate"].append(pass_rate)
        comparison["pass_at_1"].append(pass_at_1)
        comparison["pass_at_3"].append(pass_at_3)
        comparison["pass_at_5"].append(pass_at_5)
        comparison["pylint_score"].append(pylint_score)
        comparison["flake8_violations"].append(flake8_violations)
        comparison["cyclomatic_complexity"].append(cyclomatic_complexity)
        comparison["bug_density"].append(bug_density)
        comparison["generation_time"].append(generation_time)
    
    # Create DataFrame
    df = pd.DataFrame(comparison)
    
    return df


# Function to evaluate and compare approaches on a dataset
def evaluate_on_dataset(dataset: List[Dict[str, Any]], 
                         approaches: Dict[str, Callable],
                         llm_generator: Callable,
                         num_samples: int = 10) -> Dict[str, Any]:
    """
    Evaluate and compare approaches on a dataset.
    
    Args:
        dataset: List of problem dictionaries, each with 'prompt' and 'test_cases' fields
        approaches: Dictionary mapping approach names to approach functions
        llm_generator: Function to generate tokens using the LLM
        num_samples: Number of problems to sample from the dataset
        
    Returns:
        Dictionary with evaluation results
    """
    # Sample problems
    if num_samples < len(dataset):
        sampled_problems = np.random.choice(dataset, size=num_samples, replace=False)
    else:
        sampled_problems = dataset
    
    # Initialize results
    results = {approach: [] for approach in approaches}
    
    # Evaluate each approach on each problem
    for i, problem in enumerate(sampled_problems):
        print(f"Evaluating problem {i+1}/{len(sampled_problems)}")
        
        problem_id = problem.get("id", f"problem_{i}")
        prompt = problem["prompt"]
        test_cases = problem["test_cases"]
        
        for approach_name, approach_func in approaches.items():
            print(f"  Using approach: {approach_name}")
            
            try:
                # Generate code using the approach
                generation_result = approach_func(prompt, llm_generator)
                code = generation_result["code"]
                
                # Evaluate code against test cases
                evaluation_result = evaluate_code_against_tests(code, test_cases)
                
                # Calculate code quality metrics
                quality_metrics = calculate_code_quality_metrics(code)
                
                # Combine results
                combined_result = {
                    "problem_id": problem_id,
                    "code": code,
                    "pass_rate": evaluation_result["pass_rate"],
                    "is_valid": quality_metrics["is_valid"],
                    "pylint_score": quality_metrics["pylint_score"],
                    "flake8_violations": quality_metrics["flake8_violations"],
                    "cyclomatic_complexity": quality_metrics["cyclomatic_complexity"],
                    "bug_patterns": quality_metrics["bug_patterns"],
                    "metrics": generation_result.get("metrics", {}),
                    "test_results": evaluation_result["test_results"]
                }
                
                # Add to results
                results[approach_name].append(combined_result)
                
            except Exception as e:
                print(f"  Error evaluating {approach_name} on problem {problem_id}: {e}")
                
                # Add error result
                results[approach_name].append({
                    "problem_id": problem_id,
                    "error": str(e),
                    "pass_rate": 0.0,
                    "is_valid": False
                })
    
    # Create comparison DataFrame
    comparison_df = compare_approaches(results)
    
    # Return results
    return {
        "detailed_results": results,
        "comparison": comparison_df
    }