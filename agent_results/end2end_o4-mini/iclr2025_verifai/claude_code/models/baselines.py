"""
Baseline methods for comparison with ContractGPT.

This module implements baseline methods for code synthesis to compare with ContractGPT.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from models.dsl_parser import Contract, parse_dsl
from models.llm_wrapper import generate_code
from models.static_analyzer import verify_code


class LLMOnly:
    """
    Baseline: LLM-only code generation without verification loop.
    
    This baseline uses an LLM to generate code from a natural language spec
    without any formal verification or feedback loop.
    """
    
    def __init__(
        self, 
        target_language: str = "python", 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM-only baseline.
        
        Args:
            target_language: Target programming language.
            model_name: Name of the LLM to use.
            temperature: Temperature for LLM generation.
            logger: Logger for recording progress.
        """
        self.target_language = target_language
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logger or logging.getLogger("LLMOnly")
    
    def synthesize(self, spec: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Synthesize code from a specification.
        
        Args:
            spec: Specification string in the DSL format.
            
        Returns:
            A tuple (success, final_code, metrics) where success indicates if the
            code meets the spec, final_code is the synthesized code, and
            metrics contains synthesis metrics.
        """
        # Parse the specification
        contract = parse_dsl(spec)
        
        # Initialize metrics
        metrics = {
            "iterations": 1,
            "verification_time": 0.0,
            "generation_time": 0.0,
            "success": False
        }
        
        # Create the prompt for the LLM
        prompt = self._create_prompt(contract)
        
        # Generate code
        generation_start = time.time()
        candidate_code = generate_code(prompt, self.model_name, self.temperature)
        generation_end = time.time()
        metrics["generation_time"] = generation_end - generation_start
        
        # Verify the code
        verification_start = time.time()
        success, _ = verify_code(candidate_code, contract)
        verification_end = time.time()
        metrics["verification_time"] = verification_end - verification_start
        
        metrics["success"] = success
        
        return success, candidate_code, metrics
    
    def _create_prompt(self, contract: Contract) -> str:
        """
        Create a prompt for the LLM based on the contract.
        
        Args:
            contract: The contract specification.
            
        Returns:
            Prompt string for the LLM.
        """
        # Convert DSL spec to natural language
        nl_spec = self._dsl_to_nl(contract)
        
        prompt_parts = [
            f"Please implement a function in {self.target_language} that satisfies the following requirements:",
            f"",
            f"{nl_spec}",
            f"",
            f"Requirements:",
            f"1. The function should be correct and handle edge cases.",
            f"2. Provide clear comments explaining the implementation.",
        ]
        
        return "\n".join(prompt_parts)
    
    def _dsl_to_nl(self, contract: Contract) -> str:
        """
        Convert a DSL contract to natural language.
        
        Args:
            contract: The contract specification.
            
        Returns:
            Natural language description of the contract.
        """
        nl_parts = []
        
        # Add preconditions
        if contract.preconditions:
            nl_parts.append("Input conditions:")
            for i, pre in enumerate(contract.preconditions):
                nl_parts.append(f"- {pre}")
        
        # Add postconditions
        if contract.postconditions:
            nl_parts.append("\nOutput conditions:")
            for i, post in enumerate(contract.postconditions):
                nl_parts.append(f"- {post}")
        
        return "\n".join(nl_parts)


class VeCoGenLike:
    """
    Baseline: VeCoGen-like approach.
    
    This baseline uses a one-shot approach with formal specifications
    and iterative repair but without natural language feedback.
    """
    
    def __init__(
        self, 
        target_language: str = "python", 
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 5,
        temperature: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the VeCoGen-like baseline.
        
        Args:
            target_language: Target programming language.
            model_name: Name of the LLM to use.
            max_iterations: Maximum number of iterations for synthesis.
            temperature: Temperature for LLM generation.
            logger: Logger for recording progress.
        """
        self.target_language = target_language
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.logger = logger or logging.getLogger("VeCoGenLike")
    
    def synthesize(self, spec: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Synthesize code from a specification.
        
        Args:
            spec: Specification string in the DSL format.
            
        Returns:
            A tuple (success, final_code, metrics) where success is True if
            synthesis was successful, final_code is the synthesized code, and
            metrics contains synthesis metrics.
        """
        # Parse the specification
        contract = parse_dsl(spec)
        
        # Initialize metrics
        metrics = {
            "iterations": 0,
            "verification_time": 0.0,
            "generation_time": 0.0,
            "success": False
        }
        
        # Current candidate code
        candidate_code = None
        
        # Main loop
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteration {iteration}/{self.max_iterations}")
            metrics["iterations"] = iteration
            
            # Create the prompt for the LLM
            prompt = self._create_prompt(contract, candidate_code, iteration)
            
            # Generate code
            generation_start = time.time()
            candidate_code = generate_code(prompt, self.model_name, self.temperature)
            generation_end = time.time()
            metrics["generation_time"] += (generation_end - generation_start)
            
            # Verify the code
            verification_start = time.time()
            success, _ = verify_code(candidate_code, contract)
            verification_end = time.time()
            metrics["verification_time"] += (verification_end - verification_start)
            
            if success:
                # Code meets the specification
                self.logger.info("Synthesis successful!")
                metrics["success"] = True
                break
            
            self.logger.info("Verification failed. Trying again...")
        
        return metrics["success"], candidate_code, metrics
    
    def _create_prompt(self, contract: Contract, previous_code: Optional[str], iteration: int) -> str:
        """
        Create a prompt for the LLM based on the contract and previous code.
        
        Args:
            contract: The contract specification.
            previous_code: Previous candidate code, or None for first iteration.
            iteration: Current iteration number.
            
        Returns:
            Prompt string for the LLM.
        """
        prompt_parts = [
            f"You are a code synthesis system.",
            f"",
            f"Please implement a function in {self.target_language} that satisfies the following formal specification:",
            f"",
            f"{contract.raw_text}",
            f"",
            f"Requirements:",
            f"1. The function must satisfy all preconditions and postconditions.",
            f"2. Include formal annotations or assertions in the code.",
            f"3. Provide clear comments explaining the implementation.",
        ]
        
        if iteration > 1 and previous_code:
            prompt_parts.extend([
                f"",
                f"Your previous implementation contained errors. Here is the previous code:",
                f"",
                f"```{self.target_language}",
                f"{previous_code}",
                f"```",
                f"",
                f"Please fix any errors and provide an improved implementation.",
            ])
        
        return "\n".join(prompt_parts)


class LLM4CodeLike:
    """
    Baseline: LLM4Code-like approach.
    
    This baseline uses an LLM conditioned on formal specifications,
    but in a one-shot manner without a feedback loop.
    """
    
    def __init__(
        self, 
        target_language: str = "python", 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM4Code-like baseline.
        
        Args:
            target_language: Target programming language.
            model_name: Name of the LLM to use.
            temperature: Temperature for LLM generation.
            logger: Logger for recording progress.
        """
        self.target_language = target_language
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logger or logging.getLogger("LLM4CodeLike")
    
    def synthesize(self, spec: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Synthesize code from a specification.
        
        Args:
            spec: Specification string in the DSL format.
            
        Returns:
            A tuple (success, final_code, metrics) where success is True if
            synthesis was successful, final_code is the synthesized code, and
            metrics contains synthesis metrics.
        """
        # Parse the specification
        contract = parse_dsl(spec)
        
        # Initialize metrics
        metrics = {
            "iterations": 1,
            "verification_time": 0.0,
            "generation_time": 0.0,
            "success": False
        }
        
        # Create the prompt for the LLM
        prompt = self._create_prompt(contract)
        
        # Generate code
        generation_start = time.time()
        candidate_code = generate_code(prompt, self.model_name, self.temperature)
        generation_end = time.time()
        metrics["generation_time"] = generation_end - generation_start
        
        # Verify the code
        verification_start = time.time()
        success, _ = verify_code(candidate_code, contract)
        verification_end = time.time()
        metrics["verification_time"] = verification_end - verification_start
        
        metrics["success"] = success
        
        return success, candidate_code, metrics
    
    def _create_prompt(self, contract: Contract) -> str:
        """
        Create a prompt for the LLM based on the contract.
        
        Args:
            contract: The contract specification.
            
        Returns:
            Prompt string for the LLM.
        """
        prompt_parts = [
            f"You are a code synthesis system trained to generate code from formal specifications.",
            f"",
            f"Please implement a function in {self.target_language} that satisfies the following formal specification:",
            f"",
            f"{contract.raw_text}",
            f"",
            f"Requirements:",
            f"1. The function must satisfy all preconditions and postconditions.",
            f"2. Include formal annotations or assertions in the code.",
            f"3. Provide clear comments explaining the implementation.",
            f"4. Make sure edge cases are handled properly.",
            f"5. Pay special attention to the exact requirements specified in the contract.",
        ]
        
        return "\n".join(prompt_parts)