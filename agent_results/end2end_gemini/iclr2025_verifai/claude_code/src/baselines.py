"""
Baseline methods for comparison with the SSCSteer framework.

This module implements several baseline approaches:
1. Vanilla LLM generation without steering
2. Post-hoc syntax validation
3. Simple feedback-based refinement
"""

import ast
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("baselines.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Baselines")


class VanillaLLMGenerator:
    """
    Baseline approach: Generate code using an LLM without any steering.
    """
    
    def __init__(self, max_tokens: int = 1024, verbose: bool = False):
        """
        Initialize the vanilla LLM generator.
        
        Args:
            max_tokens: Maximum tokens to generate
            verbose: Whether to print verbose logs
        """
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Metrics tracking
        self.metrics = {
            "generation_time": 0,
            "total_tokens_generated": 0
        }
        
        logger.info(f"Initialized Vanilla LLM Generator with max tokens {max_tokens}")
    
    def generate_code(self, prompt: str, llm_generator: Callable) -> Dict[str, Any]:
        """
        Generate code using the LLM without any steering.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_generator: Function to generate tokens using the LLM
            
        Returns:
            Dictionary with generation results and metrics
        """
        start_time = time.time()
        
        # Start with empty generated code
        generated_code = ""
        
        # Simple token-by-token generation
        for _ in range(self.max_tokens):
            # Get next token predictions from LLM
            token_probs = llm_generator(prompt + generated_code)
            
            # Select the highest probability token
            next_token = max(token_probs.items(), key=lambda x: x[1])[0]
            
            # Add to generated code
            generated_code += next_token
            
            # Check for termination conditions
            if generated_code.endswith("\n\n") or generated_code.endswith("```"):
                # Simple heuristic: two consecutive newlines often indicate end of generation
                break
        
        # Track metrics
        self.metrics["generation_time"] = time.time() - start_time
        self.metrics["total_tokens_generated"] = len(generated_code.split())
        
        # Prepare result
        result = {
            "code": generated_code,
            "metrics": self.metrics,
            "is_syntactically_valid": self._is_syntactically_valid(generated_code)
        }
        
        logger.info(f"Code generation completed. Generated {result['metrics']['total_tokens_generated']} tokens in {self.metrics['generation_time']:.2f}s")
        logger.info(f"Final code is syntactically valid: {result['is_syntactically_valid']}")
        
        return result
    
    def _is_syntactically_valid(self, code: str) -> bool:
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


class PostHocSyntaxValidator:
    """
    Baseline approach: Generate code with the LLM and validate syntax post-hoc,
    regenerating if the code is invalid.
    """
    
    def __init__(self, 
                 max_tokens: int = 1024, 
                 max_attempts: int = 5,
                 verbose: bool = False):
        """
        Initialize the post-hoc syntax validator.
        
        Args:
            max_tokens: Maximum tokens to generate per attempt
            max_attempts: Maximum number of generation attempts
            verbose: Whether to print verbose logs
        """
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.verbose = verbose
        
        # Metrics tracking
        self.metrics = {
            "generation_time": 0,
            "total_tokens_generated": 0,
            "attempts": 0,
            "valid_generations": 0
        }
        
        logger.info(f"Initialized Post-hoc Syntax Validator with max attempts {max_attempts}")
    
    def generate_code(self, prompt: str, llm_generator: Callable) -> Dict[str, Any]:
        """
        Generate code using the LLM and validate syntax post-hoc.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_generator: Function to generate tokens using the LLM
            
        Returns:
            Dictionary with generation results and metrics
        """
        start_time = time.time()
        
        # Track attempts
        attempts = 0
        valid_generations = 0
        total_tokens_generated = 0
        
        # Generation loop
        for attempt in range(self.max_attempts):
            attempts += 1
            
            if self.verbose:
                logger.info(f"Attempt {attempt+1}/{self.max_attempts}")
            
            # Generate code
            generated_code = ""
            for _ in range(self.max_tokens):
                # Get next token predictions
                token_probs = llm_generator(prompt + generated_code)
                
                # Select highest probability token
                next_token = max(token_probs.items(), key=lambda x: x[1])[0]
                
                # Add to generated code
                generated_code += next_token
                
                # Check for termination
                if generated_code.endswith("\n\n") or generated_code.endswith("```"):
                    break
            
            total_tokens_generated += len(generated_code.split())
            
            # Check if code is syntactically valid
            is_valid = self._is_syntactically_valid(generated_code)
            
            if is_valid:
                valid_generations += 1
                if self.verbose:
                    logger.info("Generated syntactically valid code")
                
                # Break the loop as we found valid code
                break
            
            if self.verbose:
                logger.info("Generated code is not syntactically valid, retrying...")
            
            # If we're on the last attempt, do basic fixes
            if attempt == self.max_attempts - 1:
                generated_code = self._fix_common_syntax_errors(generated_code)
                is_valid = self._is_syntactically_valid(generated_code)
                
                if is_valid:
                    valid_generations += 1
                    if self.verbose:
                        logger.info("Fixed code is syntactically valid")
        
        # Track metrics
        self.metrics["generation_time"] = time.time() - start_time
        self.metrics["total_tokens_generated"] = total_tokens_generated
        self.metrics["attempts"] = attempts
        self.metrics["valid_generations"] = valid_generations
        
        # Prepare result
        result = {
            "code": generated_code,
            "metrics": self.metrics,
            "is_syntactically_valid": is_valid
        }
        
        logger.info(f"Code generation completed. Made {attempts} attempts, found {valid_generations} valid generations")
        logger.info(f"Generated {total_tokens_generated} tokens in {self.metrics['generation_time']:.2f}s")
        logger.info(f"Final code is syntactically valid: {result['is_syntactically_valid']}")
        
        return result
    
    def _is_syntactically_valid(self, code: str) -> bool:
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
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """
        Apply simple fixes for common syntax errors.
        
        Args:
            code: Python code with syntax errors
            
        Returns:
            Fixed Python code
        """
        # 1. Fix missing closing parentheses/brackets/braces
        open_chars = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in open_chars:
                stack.append(char)
            elif char in open_chars.values():
                if stack and open_chars[stack[-1]] == char:
                    stack.pop()
        
        # Add missing closing characters
        fixed_code = code
        while stack:
            fixed_code += open_chars[stack.pop()]
        
        # 2. Fix missing colons after control flow statements
        patterns = [
            (r'(if\s+[^:]+)(\s*)$', r'\1:\2'),
            (r'(elif\s+[^:]+)(\s*)$', r'\1:\2'),
            (r'(else)(\s*)$', r'\1:\2'),
            (r'(for\s+[^:]+)(\s*)$', r'\1:\2'),
            (r'(while\s+[^:]+)(\s*)$', r'\1:\2'),
            (r'(def\s+\w+\([^)]*\))(\s*)$', r'\1:\2'),
            (r'(class\s+\w+)(\s*)$', r'\1:\2')
        ]
        
        for pattern, replacement in patterns:
            fixed_code = re.sub(pattern, replacement, fixed_code)
        
        # 3. Fix indentation errors
        lines = fixed_code.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            if i > 0 and lines[i-1].rstrip().endswith(':'):
                # If previous line ends with colon, this line should be indented
                if not line.startswith(' ') and line.strip():
                    fixed_lines.append('    ' + line)
                    continue
                    
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)


class FeedbackBasedRefinement:
    """
    Baseline approach: Generate code with the LLM, analyze it for issues,
    and provide feedback to the LLM for refinement.
    """
    
    def __init__(self, 
                 max_tokens: int = 1024, 
                 max_iterations: int = 3,
                 verbose: bool = False):
        """
        Initialize the feedback-based refinement generator.
        
        Args:
            max_tokens: Maximum tokens to generate per iteration
            max_iterations: Maximum number of refinement iterations
            verbose: Whether to print verbose logs
        """
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Metrics tracking
        self.metrics = {
            "generation_time": 0,
            "total_tokens_generated": 0,
            "iterations": 0,
            "improvement_ratio": 0  # Ratio of issues fixed across iterations
        }
        
        logger.info(f"Initialized Feedback-based Refinement with max iterations {max_iterations}")
    
    def generate_code(self, prompt: str, llm_generator: Callable) -> Dict[str, Any]:
        """
        Generate code with iterative feedback-based refinement.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_generator: Function to generate tokens using the LLM
            
        Returns:
            Dictionary with generation results and metrics
        """
        start_time = time.time()
        
        # Initial generation
        current_code = self._generate_initial_code(prompt, llm_generator)
        
        # Track iterations and issues
        iterations = 1
        initial_issues = self._analyze_code(current_code)
        current_issues = initial_issues
        
        if self.verbose:
            logger.info(f"Initial generation complete. Found {len(initial_issues)} issues")
        
        # Refinement loop
        for iteration in range(1, self.max_iterations):
            if not current_issues:
                # No issues to fix, terminate early
                if self.verbose:
                    logger.info("No issues to fix, terminating refinement")
                break
                
            # Generate feedback for the LLM
            feedback = self._generate_feedback(current_issues)
            
            if self.verbose:
                logger.info(f"Iteration {iteration+1}/{self.max_iterations}")
                logger.info(f"Feedback: {feedback}")
            
            # Create refinement prompt
            refinement_prompt = f"{prompt}\n\n{current_code}\n\n{feedback}\n\nPlease fix these issues in the code:\n"
            
            # Generate refined code
            refined_code = self._generate_initial_code(refinement_prompt, llm_generator)
            
            # Check if refinement improved the code
            new_issues = self._analyze_code(refined_code)
            
            if len(new_issues) < len(current_issues):
                # Refinement improved the code
                current_code = refined_code
                current_issues = new_issues
                
                if self.verbose:
                    logger.info(f"Refinement improved the code. Issues reduced from {len(current_issues)} to {len(new_issues)}")
            else:
                # Refinement did not improve the code
                if self.verbose:
                    logger.info("Refinement did not improve the code, terminating refinement")
                break
                
            iterations += 1
        
        # Calculate improvement ratio
        if initial_issues:
            self.metrics["improvement_ratio"] = 1.0 - (len(current_issues) / len(initial_issues))
        else:
            self.metrics["improvement_ratio"] = 1.0
            
        # Track metrics
        self.metrics["generation_time"] = time.time() - start_time
        self.metrics["total_tokens_generated"] += len(current_code.split())
        self.metrics["iterations"] = iterations
        
        # Prepare result
        result = {
            "code": current_code,
            "initial_issues": [str(issue) for issue in initial_issues],
            "final_issues": [str(issue) for issue in current_issues],
            "metrics": self.metrics,
            "is_syntactically_valid": self._is_syntactically_valid(current_code)
        }
        
        logger.info(f"Refinement completed. Made {iterations} iterations")
        logger.info(f"Issues reduced from {len(initial_issues)} to {len(current_issues)}")
        logger.info(f"Generated {self.metrics['total_tokens_generated']} tokens in {self.metrics['generation_time']:.2f}s")
        logger.info(f"Final code is syntactically valid: {result['is_syntactically_valid']}")
        
        return result
    
    def _generate_initial_code(self, prompt: str, llm_generator: Callable) -> str:
        """
        Generate initial code using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_generator: Function to generate tokens using the LLM
            
        Returns:
            Generated code string
        """
        # Simple token-by-token generation
        generated_code = ""
        
        for _ in range(self.max_tokens):
            # Get next token predictions
            token_probs = llm_generator(prompt + generated_code)
            
            # Select highest probability token
            next_token = max(token_probs.items(), key=lambda x: x[1])[0]
            
            # Add to generated code
            generated_code += next_token
            
            # Check for termination
            if generated_code.endswith("\n\n") or generated_code.endswith("```"):
                break
        
        # Update token count
        self.metrics["total_tokens_generated"] += len(generated_code.split())
        
        return generated_code
    
    def _analyze_code(self, code: str) -> List[str]:
        """
        Analyze code for issues.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List of issue strings
        """
        issues = []
        
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            
        # Check for common issues using basic patterns
        
        # 1. Missing docstrings
        if "def " in code and '"""' not in code and "'''" not in code:
            issues.append("Missing docstrings in functions")
            
        # 2. Long lines
        for i, line in enumerate(code.split('\n')):
            if len(line) > 100:
                issues.append(f"Line {i+1} is too long ({len(line)} characters)")
                
        # 3. Variables used before assignment
        # This is a very basic check that doesn't handle all cases
        lines = code.split('\n')
        vars_defined = set()
        
        for i, line in enumerate(lines):
            # Check for variable assignments
            if '=' in line and not line.strip().startswith('#') and '==' not in line:
                var_name = line.split('=')[0].strip()
                if var_name.isidentifier():
                    vars_defined.add(var_name)
                    
            # Check for variable uses
            for var in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line):
                if var not in vars_defined and var not in dir(__builtins__) and var not in ["self", "cls"]:
                    issues.append(f"Line {i+1}: Variable '{var}' may be used before assignment")
                    break
        
        # 4. Unused imports
        import_pattern = re.compile(r'import\s+(\w+)')
        from_import_pattern = re.compile(r'from\s+[\w.]+\s+import\s+(.+)')
        
        imports = []
        for match in import_pattern.finditer(code):
            imports.append(match.group(1))
            
        for match in from_import_pattern.finditer(code):
            for imp in match.group(1).split(','):
                imp = imp.strip().split(' as ')[0]
                if imp != '*':
                    imports.append(imp)
        
        for imp in imports:
            if imp not in code.replace(f"import {imp}", "").replace(f"from {imp}", ""):
                issues.append(f"Unused import: {imp}")
                
        return issues
    
    def _generate_feedback(self, issues: List[str]) -> str:
        """
        Generate feedback string based on issues.
        
        Args:
            issues: List of issue strings
            
        Returns:
            Feedback string for the LLM
        """
        if not issues:
            return "The code looks good! No issues found."
            
        feedback = "I found the following issues with the code:\n"
        
        for i, issue in enumerate(issues[:5]):  # Limit to top 5 issues
            feedback += f"{i+1}. {issue}\n"
            
        if len(issues) > 5:
            feedback += f"And {len(issues) - 5} more issues.\n"
            
        feedback += "\nPlease fix these issues while preserving the core functionality."
        
        return feedback
    
    def _is_syntactically_valid(self, code: str) -> bool:
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