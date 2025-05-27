"""
Baseline Single-Agent model for the Multi-Agent Collaborative Programming (MACP) framework evaluation.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import anthropic

class SingleAgent:
    """
    Baseline Single-Agent model that attempts to solve the entire programming task.
    """
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", logger: Optional[logging.Logger] = None):
        """
        Initialize the single agent.
        
        Args:
            model_name: Name of the LLM to use
            logger: Logger instance for tracking events
        """
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize the anthropic client
        self.client = anthropic.Anthropic()
        self.messages = []
        
    def solve_task(self, task: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a programming task using the single-agent approach.
        
        Args:
            task: The programming task description
            
        Returns:
            Tuple containing the solution code and metadata
        """
        self.logger.info(f"Single-agent solving task: {task['name']}")
        
        # Record start time
        start_time = time.time()
        
        # Construct the prompt
        prompt = self._construct_prompt(task)
        self.logger.info("Prompt constructed for single-agent")
        
        # Get solution from the model
        solution, response_message = self._get_solution(prompt)
        self.logger.info("Received solution from single-agent")
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Collect metadata
        metadata = {
            'execution_time': execution_time,
            'messages_count': 1,  # Only one message for single-agent
            'model_name': self.model_name,
            'task_id': task['id'],
            'task_name': task['name'],
            'task_complexity': task['complexity']
        }
        
        self.logger.info(f"Single-agent completed task in {execution_time:.2f} seconds")
        
        return solution, metadata
    
    def _construct_prompt(self, task: Dict[str, Any]) -> str:
        """
        Construct a prompt for the LLM based on the task.
        
        Args:
            task: The programming task description
            
        Returns:
            The constructed prompt string
        """
        prompt = f"""
You are an expert programmer tasked with implementing a programming solution.

# Task Description
{task['description']}

# Requirements
{', '.join(task['requirements'])}

# Expected Output
{json.dumps(task['expected_output'], indent=2)}

# Test Cases
{json.dumps(task.get('test_cases', []), indent=2)}

# Instructions
1. Analyze the requirements carefully
2. Design a solution that satisfies all requirements
3. Implement the solution in Python
4. Include comments explaining your implementation decisions
5. Ensure your solution passes all test cases

Please provide only the code implementation. Start your implementation with ```python and end with ```.
Make sure your solution is complete, well-structured, and follows best practices.
"""
        return prompt
    
    def _get_solution(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get a solution from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple containing the solution code and the full response message
        """
        try:
            # Send the prompt to the model
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the solution code
            content = response.content[0].text
            
            # Extract code between triple backticks
            code_blocks = content.split("```")
            if len(code_blocks) >= 3:
                # Get the content of the first code block
                solution_code = code_blocks[1]
                # Remove language identifier if present
                if solution_code.startswith("python"):
                    solution_code = solution_code[len("python"):].strip()
                else:
                    solution_code = solution_code.strip()
            else:
                # If no code blocks found, use entire content
                solution_code = content
            
            return solution_code, response.model_dump()
            
        except Exception as e:
            self.logger.error(f"Error getting solution from model: {str(e)}")
            return "", {}