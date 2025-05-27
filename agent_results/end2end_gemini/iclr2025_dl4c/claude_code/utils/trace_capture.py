#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execution trace capture module for the IETA framework.
This module provides functionality to execute code snippets and capture detailed execution traces.
"""

import sys
import io
import traceback
import subprocess
import tempfile
import os
import signal
import time
import logging
import docker
from contextlib import redirect_stdout, redirect_stderr
import json

logger = logging.getLogger(__name__)

class ExecutionTraceCapture:
    """Class for executing code and capturing detailed execution traces."""
    
    def __init__(self, max_execution_time=10, use_docker=False, docker_image="python:3.9-slim"):
        """
        Initialize the execution trace capture system.
        
        Args:
            max_execution_time (int): Maximum execution time in seconds
            use_docker (bool): Whether to use Docker for isolated execution
            docker_image (str): Docker image to use for execution
        """
        self.max_execution_time = max_execution_time
        self.use_docker = use_docker
        self.docker_image = docker_image
        
        if use_docker:
            try:
                self.docker_client = docker.from_env()
                logger.info(f"Docker initialized with image: {docker_image}")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker: {e}. Falling back to direct execution.")
                self.use_docker = False
    
    def execute_and_capture(self, code_snippet, test_cases=None):
        """
        Execute a code snippet and capture detailed execution traces.
        
        Args:
            code_snippet (str): The code to execute
            test_cases (list, optional): List of test cases to run
            
        Returns:
            dict: Execution trace including stdout, stderr, error type, stack trace, etc.
        """
        if self.use_docker:
            return self._execute_in_docker(code_snippet, test_cases)
        else:
            return self._execute_directly(code_snippet, test_cases)
    
    def _execute_directly(self, code_snippet, test_cases=None):
        """Execute code directly in the current process with safety measures."""
        # Prepare trace dictionary
        trace = {
            "stdout": "",
            "stderr": "",
            "error_type": None,
            "error_message": None,
            "stack_trace": None,
            "execution_time": 0,
            "timeout": False,
            "compile_error": False,
            "runtime_error": False,
            "variable_states": {},
            "test_results": []
        }
        
        # Combine code and test cases
        if test_cases:
            full_code = code_snippet + "\n\n" + "\n".join(test_cases)
        else:
            full_code = code_snippet
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            # Add tracing code
            trace_code = """
import sys
import traceback
from io import StringIO
import time

# Dictionary to store variable states
_var_states = {}

def _capture_variables(frame, event, arg):
    if event == 'line':
        # Capture local variables
        local_vars = frame.f_locals.copy()
        # Store only serializable variables, avoid capturing functions, classes, etc.
        for name, value in local_vars.items():
            if name.startswith('__') or callable(value):
                continue
            try:
                # Only capture primitives, lists, dicts with simple types
                if isinstance(value, (int, float, str, bool, list, dict, set, tuple, type(None))):
                    _var_states[f"{frame.f_code.co_filename}:{frame.f_lineno}:{name}"] = repr(value)
            except:
                pass
    return _capture_variables

# Set the trace function
sys.settrace(_capture_variables)

# Start time
_start_time = time.time()

try:
    # Main code execution happens here
"""
            
            # Add the code with indentation
            indented_code = "\n".join(f"    {line}" for line in full_code.split("\n"))
            
            # Close the try block and add exception handling
            exception_handling = """
except Exception as e:
    _error_type = type(e).__name__
    _error_message = str(e)
    _stack_trace = traceback.format_exc()
    print(f"Error: {_error_type}: {_error_message}")
    print(_stack_trace)
else:
    _error_type = None
    _error_message = None
    _stack_trace = None
finally:
    # End time
    _execution_time = time.time() - _start_time
    
    # Disable tracing
    sys.settrace(None)
    
    # Print execution metadata as JSON
    import json
    print("\\n==EXECUTION_METADATA==")
    metadata = {
        "error_type": _error_type,
        "error_message": _error_message,
        "stack_trace": _stack_trace,
        "execution_time": _execution_time,
        "variable_states": _var_states
    }
    print(json.dumps(metadata))
"""
            
            # Write the complete instrumented code
            tmp_file.write((trace_code + indented_code + exception_handling).encode('utf-8'))
        
        # Execute the code in a separate process with timeout
        start_time = time.time()
        cmd = [sys.executable, tmp_path]
        
        try:
            # Execute the code
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the process with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.max_execution_time)
                trace["stdout"] = stdout
                trace["stderr"] = stderr
                
                # Parse execution metadata if available
                if "==EXECUTION_METADATA==" in stdout:
                    metadata_str = stdout.split("==EXECUTION_METADATA==")[1].strip()
                    try:
                        metadata = json.loads(metadata_str)
                        trace.update(metadata)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse execution metadata: {e}")
                
                # Check if there was an error
                trace["runtime_error"] = process.returncode != 0 or trace["error_type"] is not None
                
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                stdout, stderr = process.communicate()
                trace["stdout"] = stdout
                trace["stderr"] = stderr
                trace["timeout"] = True
                trace["runtime_error"] = True
                trace["error_type"] = "TimeoutError"
                trace["error_message"] = f"Execution timed out after {self.max_execution_time} seconds"
        
        except Exception as e:
            # Handle any other exceptions
            trace["runtime_error"] = True
            trace["error_type"] = type(e).__name__
            trace["error_message"] = str(e)
            trace["stack_trace"] = traceback.format_exc()
        
        finally:
            # Calculate execution time
            trace["execution_time"] = time.time() - start_time
            
            # Delete the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return trace
    
    def _execute_in_docker(self, code_snippet, test_cases=None):
        """Execute code in a Docker container for isolation."""
        trace = {
            "stdout": "",
            "stderr": "",
            "error_type": None,
            "error_message": None,
            "stack_trace": None,
            "execution_time": 0,
            "timeout": False,
            "compile_error": False,
            "runtime_error": False,
            "variable_states": {},
            "test_results": []
        }
        
        # For this implementation, we'll just return a simulated trace
        # In a real implementation, you would create a Docker container and execute the code
        # with the same instrumentation as in _execute_directly
        
        logger.warning("Docker execution is not fully implemented in this demo version.")
        trace["stdout"] = "Docker execution simulation"
        trace["execution_time"] = 0.1
        
        return trace
    
    def classify_trace(self, trace):
        """
        Classify the execution trace into one of the defined outcome states.
        
        Args:
            trace (dict): The execution trace
            
        Returns:
            str: One of 'S_succ', 'S_err', 'S_comp_err', 'S_timeout', 'S_fail_test'
        """
        if trace["timeout"]:
            return "S_timeout"
        
        if trace["compile_error"]:
            return "S_comp_err"
        
        if trace["runtime_error"]:
            return "S_err"
        
        # Check if all tests passed (if any)
        if trace["test_results"] and not all(result["passed"] for result in trace["test_results"]):
            return "S_fail_test"
        
        # If we got here, execution was successful
        return "S_succ"
    
    def extract_variable_states(self, trace, error_context=True):
        """
        Extract relevant variable states from the trace.
        
        Args:
            trace (dict): The execution trace
            error_context (bool): Whether to focus on variables related to errors
            
        Returns:
            dict: Dictionary of relevant variable states
        """
        variable_states = {}
        
        if not trace["variable_states"]:
            return variable_states
        
        if not error_context or trace["error_type"] is None:
            # Return all variable states
            return trace["variable_states"]
        
        # If there's an error, focus on variables around the error location
        if trace["stack_trace"]:
            # Extract line numbers from the stack trace
            error_lines = []
            for line in trace["stack_trace"].split("\n"):
                if "line" in line and "File" in line:
                    try:
                        line_num = int(line.split("line")[1].split(",")[0].strip())
                        error_lines.append(line_num)
                    except (ValueError, IndexError):
                        pass
            
            # Get variables from error lines and surrounding context (5 lines before and after)
            context_range = 5
            context_lines = set()
            for line in error_lines:
                for i in range(line - context_range, line + context_range + 1):
                    context_lines.add(i)
            
            # Filter variable states to only include those from context lines
            for var_key, var_value in trace["variable_states"].items():
                try:
                    file_line = var_key.split(":")[1]
                    if int(file_line) in context_lines:
                        variable_states[var_key] = var_value
                except (ValueError, IndexError):
                    # Keep variables where we can't parse the line number
                    variable_states[var_key] = var_value
        
        return variable_states