"""
ContractGPT: A Closed-Loop Formal Specification-Guided LLM Code Synthesis Framework.

This module implements the core ContractGPT algorithm that integrates all components
into a single closed-loop "spec-generate-verify-refine" cycle.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from models.dsl_parser import Contract, parse_dsl
from models.llm_wrapper import generate_code
from models.static_analyzer import verify_code
from models.feedback_translator import translate_counterexamples


class ContractGPT:
    """
    The main ContractGPT implementation.
    
    This class implements the "spec-generate-verify-refine" cycle of the
    ContractGPT framework, as described in the paper.
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
        Initialize the ContractGPT framework.
        
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
        self.logger = logger or logging.getLogger("ContractGPT")
    
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
            "success": False,
            "feedback_history": []
        }
        
        # Initialize feedback
        feedback = None
        
        # Current candidate code
        candidate_code = None
        
        # Main loop
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteration {iteration}/{self.max_iterations}")
            metrics["iterations"] = iteration
            
            # Create the prompt for the LLM
            prompt = self._create_prompt(contract, feedback, iteration)
            
            # Generate code
            generation_start = time.time()
            candidate_code = generate_code(prompt, self.model_name, self.temperature)
            generation_end = time.time()
            metrics["generation_time"] += (generation_end - generation_start)
            
            # Verify the code
            verification_start = time.time()
            success, counterexamples = verify_code(candidate_code, contract)
            verification_end = time.time()
            metrics["verification_time"] += (verification_end - verification_start)
            
            if success:
                # Code meets the specification
                self.logger.info("Synthesis successful!")
                metrics["success"] = True
                break
            
            # Convert counterexamples to feedback
            feedback = translate_counterexamples(counterexamples)
            metrics["feedback_history"].append(feedback)
            
            self.logger.info(f"Verification failed. Feedback: {feedback}")
        
        return metrics["success"], candidate_code, metrics
    
    def _create_prompt(self, contract: Contract, feedback: Optional[str], iteration: int) -> str:
        """
        Create a prompt for the LLM based on the contract and feedback.
        
        Args:
            contract: The contract specification.
            feedback: Feedback from previous iteration, or None for first iteration.
            iteration: Current iteration number.
            
        Returns:
            Prompt string for the LLM.
        """
        prompt_parts = [
            f"You are ContractGPT, a system that generates code based on formal specifications.",
            f"",
            f"Please implement a function in {self.target_language} that satisfies the following specification:",
            f"",
            f"{contract.raw_text}",
            f"",
            f"Requirements:",
            f"1. The function must satisfy all preconditions and postconditions.",
            f"2. Include appropriate assertions in the code to enforce the conditions.",
            f"3. Provide clear comments explaining the implementation.",
            f"4. Make sure edge cases are handled properly.",
            f"",
        ]
        
        if iteration > 1 and feedback:
            prompt_parts.extend([
                f"Previous attempt had the following issues that need to be fixed:",
                f"",
                f"{feedback}",
                f"",
                f"Please refine the implementation to address these issues.",
            ])
        
        return "\n".join(prompt_parts)


def synthesize_from_spec(spec: str, language: str = "python", model_name: str = "gpt-4o-mini") -> Tuple[bool, str, Dict[str, Any]]:
    """
    Convenience function to synthesize code from a specification.
    
    Args:
        spec: Specification string in the DSL format.
        language: Target programming language.
        model_name: Name of the LLM to use.
        
    Returns:
        A tuple (success, final_code, metrics) where success is True if
        synthesis was successful, final_code is the synthesized code, and
        metrics contains synthesis metrics.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("ContractGPT")
    
    # Create ContractGPT instance
    cgpt = ContractGPT(
        target_language=language,
        model_name=model_name,
        logger=logger
    )
    
    # Synthesize code
    return cgpt.synthesize(spec)