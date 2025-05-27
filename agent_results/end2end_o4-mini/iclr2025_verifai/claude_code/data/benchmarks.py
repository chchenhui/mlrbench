"""
Benchmark suite for ContractGPT.

This module provides a set of algorithmic and systems-level benchmarks
for evaluating ContractGPT and baseline methods.
"""

import os
import json
import sys
from typing import Dict, List, Any

# Add parent directory to path to allow importing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.experiment import create_benchmark_json


def generate_algorithmic_benchmarks(output_dir: str) -> List[str]:
    """
    Generate algorithmic benchmarks.
    
    Args:
        output_dir: Directory to save benchmark files.
        
    Returns:
        List of created benchmark file paths.
    """
    benchmarks = []
    
    # Bubble Sort
    bubble_sort_spec = """
    requires length(arr) == n && n >= 0
    ensures forall i,j. 0 <= i < j < n ==> arr[i] <= arr[j]
    ensures multiset(arr_out) == multiset(arr_in)
    """
    
    benchmarks.append(create_benchmark_json(
        name="bubble_sort",
        spec=bubble_sort_spec,
        description="Bubble sort implementation that sorts an array in ascending order.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    # Binary Search
    binary_search_spec = """
    requires length(arr) == n && n >= 0
    requires forall i,j. 0 <= i < j < n ==> arr[i] <= arr[j]
    ensures (result == -1 ==> forall i. 0 <= i < n ==> arr[i] != target)
    ensures (result != -1 ==> 0 <= result < n && arr[result] == target)
    """
    
    benchmarks.append(create_benchmark_json(
        name="binary_search",
        spec=binary_search_spec,
        description="Binary search implementation that finds the index of a target value in a sorted array.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    # Quick Sort
    quick_sort_spec = """
    requires length(arr) == n && n >= 0
    ensures forall i,j. 0 <= i < j < n ==> arr[i] <= arr[j]
    ensures multiset(arr_out) == multiset(arr_in)
    """
    
    benchmarks.append(create_benchmark_json(
        name="quick_sort",
        spec=quick_sort_spec,
        description="Quick sort implementation that sorts an array in ascending order.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    # Breadth-First Search
    bfs_spec = """
    requires valid_graph(graph)
    requires node_exists(graph, start)
    ensures result == [] || node_exists(graph, result[0])
    ensures forall node in result: reachable(graph, start, node)
    ensures forall i,j. 0 <= i < j < length(result) ==> distance(graph, start, result[i]) <= distance(graph, start, result[j])
    """
    
    benchmarks.append(create_benchmark_json(
        name="breadth_first_search",
        spec=bfs_spec,
        description="Breadth-first search implementation that finds the shortest path in a graph.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    # Dijkstra's Algorithm
    dijkstra_spec = """
    requires valid_weighted_graph(graph)
    requires node_exists(graph, start)
    ensures forall node in result.keys(): node_exists(graph, node) && reachable(graph, start, node)
    ensures result[start] == 0
    ensures forall node in result.keys(): result[node] == shortest_distance(graph, start, node)
    """
    
    benchmarks.append(create_benchmark_json(
        name="dijkstra",
        spec=dijkstra_spec,
        description="Dijkstra's algorithm implementation that finds the shortest paths in a weighted graph.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    return benchmarks


def generate_systems_benchmarks(output_dir: str) -> List[str]:
    """
    Generate systems-level benchmarks.
    
    Args:
        output_dir: Directory to save benchmark files.
        
    Returns:
        List of created benchmark file paths.
    """
    benchmarks = []
    
    # File Buffer
    file_buffer_spec = """
    requires buffer_size > 0
    requires file_path != null && file_path != ""
    ensures result != null
    ensures result.remaining_capacity() == buffer_size
    ensures result.position() == 0
    """
    
    benchmarks.append(create_benchmark_json(
        name="file_buffer",
        spec=file_buffer_spec,
        description="File buffer implementation for reading from a file.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    # Memory Pool Allocator
    memory_pool_spec = """
    requires pool_size > 0
    requires block_size > 0
    ensures result != null
    ensures result.get_pool_size() == pool_size
    ensures result.get_block_size() == block_size
    ensures result.get_free_blocks() == pool_size / block_size
    """
    
    benchmarks.append(create_benchmark_json(
        name="memory_pool_allocator",
        spec=memory_pool_spec,
        description="Memory pool allocator implementation for efficient memory management.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    # HTTP Request Parser
    http_parser_spec = """
    requires request != null && request != ""
    ensures (valid_http_request(request) ==> result != null && result.method != null && result.path != null && result.headers != null)
    ensures (!valid_http_request(request) ==> result == null)
    """
    
    benchmarks.append(create_benchmark_json(
        name="http_request_parser",
        spec=http_parser_spec,
        description="HTTP request parser implementation that parses HTTP request strings.",
        output_dir=output_dir,
        overwrite=True
    ))
    
    return benchmarks


def generate_all_benchmarks(output_dir: str) -> List[str]:
    """
    Generate all benchmarks.
    
    Args:
        output_dir: Directory to save benchmark files.
        
    Returns:
        List of created benchmark file paths.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate benchmarks
    algorithmic_dir = os.path.join(output_dir, "algorithmic")
    systems_dir = os.path.join(output_dir, "systems")
    
    os.makedirs(algorithmic_dir, exist_ok=True)
    os.makedirs(systems_dir, exist_ok=True)
    
    algorithmic_benchmarks = generate_algorithmic_benchmarks(algorithmic_dir)
    systems_benchmarks = generate_systems_benchmarks(systems_dir)
    
    return algorithmic_benchmarks + systems_benchmarks


def load_all_benchmarks(benchmark_dir: str) -> List[Dict[str, Any]]:
    """
    Load all benchmarks from the given directory and its subdirectories.
    
    Args:
        benchmark_dir: Directory containing benchmark files.
        
    Returns:
        List of benchmark dictionaries.
    """
    benchmarks = []
    
    for root, _, files in os.walk(benchmark_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    # Validate benchmark data
                    if "name" not in benchmark_data or "spec" not in benchmark_data:
                        print(f"Warning: Benchmark {file} missing required fields")
                        continue
                    
                    benchmarks.append(benchmark_data)
                    
                except Exception as e:
                    print(f"Failed to load benchmark {file}: {e}")
    
    return benchmarks