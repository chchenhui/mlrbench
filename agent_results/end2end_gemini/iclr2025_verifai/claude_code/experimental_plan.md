# Experimental Plan for SSCSteer Framework

## Overview
This document outlines the experimental plan to evaluate the effectiveness of the Syntactic and Semantic Conformance Steering (SSCSteer) framework for improving LLM code generation. The framework consists of two main components:
1. Syntactic Steering Module (SSM): Leverages context-free grammars to ensure syntactic correctness
2. Semantic Steering Module (SeSM): Uses lightweight static analysis and SMT solver to ensure semantic correctness

## Experimental Design

### Target Programming Languages
- Primary: Python (rich ecosystem, widely used)
- Secondary: Java (statically typed, more explicit semantic checks)
- Low-resource: Nim (relatively niche language with different paradigms)

### Datasets
1. **HumanEval Dataset**: 164 Python programming problems with test cases
2. **MBPP (Mostly Basic Python Problems)**: Collection of Python programming tasks
3. **Custom Semantic Tasks**: Specifically designed tasks with formal specifications
   - Tasks requiring null-checking
   - Tasks with array bound constraints
   - Tasks involving type constraints
   - Tasks requiring resource management (file handling, etc.)

### Models and Baselines
1. **Base Models**:
   - CodeLlama-7B
   - Qwen2-7B-Instruct
   
2. **Baseline Methods**:
   - Vanilla LLM generation without steering
   - Post-hoc syntax validation only (parse the output, regenerate if it fails)
   - Simple feedback-based refinement (providing error messages back to LLM)

3. **Our Methods**:
   - SSM-only: Syntactic steering only
   - SeSM-only: Semantic steering only (with minimal syntactic guarantees)
   - Full SSCSteer: Combined syntactic and semantic steering

### Evaluation Metrics

#### Syntactic Correctness
- Percentage of generated programs that parse successfully
- Average number of syntax errors per solution

#### Semantic Correctness
- Pass@k: Percentage of problems with at least one correct solution in k attempts
- Specification adherence rate for tasks with formal specifications
- Bug density (detected by static analyzers like Pylint, Flake8)
- Frequency of targeted bug patterns (null dereferences, out-of-bounds access)

#### Efficiency
- Generation time per solution
- Computational overhead of steering modules
- Token efficiency (number of tokens to generate working solutions)

#### Code Quality
- CodeBLEU score comparing to reference solutions
- Cyclomatic complexity
- Maintainability metrics

### Experimental Procedure

1. **Setup Phase**:
   - Prepare datasets and evaluation infrastructure
   - Implement SSM using Python's ast module and a CFG parser
   - Implement SeSM using static analysis tools and Z3 SMT solver
   - Integrate with LLM API for controlled generation

2. **Main Experiments**:
   - Generate solutions for all problems using each method and model
   - For each solution, collect metrics on syntactic/semantic correctness
   - Run test cases to verify functional correctness
   - Measure computational overhead and generation time

3. **Ablation Studies**:
   - SSM-only vs. Base LLM
   - SeSM-only vs. Base LLM
   - Different SeSM configurations (static analysis only vs. static analysis + SMT)
   - Impact of beam width on solution quality

## Implementation Plan

### Component 1: Syntactic Steering Module (SSM)
- Implement a Python parser using the `ast` module
- Create token filters based on partial parsing results
- Integrate with LLM generation to mask invalid tokens

### Component 2: Semantic Steering Module (SeSM)
- Implement lightweight static analysis for common bug patterns
- Integrate Z3 solver for formal specification checking
- Create a penalty mechanism for semantically problematic generations

### Component 3: Integration Framework
- Implement beam search with syntactic and semantic steering
- Create a unified generation pipeline
- Design generation monitoring and logging

### Component 4: Evaluation Framework
- Automate test case execution
- Calculate evaluation metrics
- Generate visualization of results

## Expected Results
- Significant improvement in syntactic correctness (>95% parsability)
- Moderate improvement in Pass@1 scores (15-30% relative improvement)
- Reduction in specific bug patterns targeted by SeSM
- Acceptable computational overhead (2-5x generation time)

## Timeline
1. Implementation of core components (SSM, SeSM): 2 days
2. Integration and testing: 1 day
3. Running experiments: 1 day
4. Analysis and visualization: 1 day