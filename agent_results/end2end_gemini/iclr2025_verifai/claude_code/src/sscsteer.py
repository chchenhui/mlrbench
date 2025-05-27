"""
Syntactic and Semantic Conformance Steering (SSCSteer) Framework.

This module integrates the Syntactic Steering Module (SSM) and
Semantic Steering Module (SeSM) into a unified framework that interfaces
with LLMs for guided code generation.
"""

import time
import json
import logging
from typing import List, Dict, Set, Tuple, Optional, Union, Any, Callable

from .ssm import SyntacticSteeringModule
from .sesm import SemanticSteeringModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sscsteer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SSCSteer")


class SSCSteer:
    """
    Main SSCSteer framework that combines syntactic and semantic steering
    for improved LLM code generation.
    """
    
    def __init__(self, 
                 language: str = "python", 
                 use_syntactic_steering: bool = True,
                 use_semantic_steering: bool = True,
                 semantic_check_frequency: int = 5,
                 beam_width: int = 3,
                 use_smt: bool = True,
                 max_tokens: int = 1024,
                 verbose: bool = False):
        """
        Initialize the SSCSteer framework.
        
        Args:
            language: Target programming language
            use_syntactic_steering: Whether to use syntactic steering
            use_semantic_steering: Whether to use semantic steering
            semantic_check_frequency: How often to perform semantic checks (in tokens)
            beam_width: Width of beam search
            use_smt: Whether to use SMT solver for formal checks
            max_tokens: Maximum tokens to generate
            verbose: Whether to print verbose logs
        """
        self.language = language
        self.use_syntactic_steering = use_syntactic_steering
        self.use_semantic_steering = use_semantic_steering
        self.semantic_check_frequency = semantic_check_frequency
        self.beam_width = beam_width
        self.use_smt = use_smt
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Initialize steering modules
        if use_syntactic_steering:
            self.ssm = SyntacticSteeringModule(language)
        
        if use_semantic_steering:
            self.sesm = SemanticSteeringModule(language, use_smt)
            
        # Metrics tracking
        self.metrics = {
            "generation_time": 0,
            "total_tokens_generated": 0,
            "syntactic_checks": 0,
            "semantic_checks": 0,
            "syntactic_filter_rate": 0,  # Average % of tokens filtered by SSM
            "semantic_penalties": []  # List of penalties applied by SeSM
        }
        
        logger.info(f"Initialized SSCSteer for {language} with beam width {beam_width}")
        if use_syntactic_steering:
            logger.info("Syntactic steering enabled")
        if use_semantic_steering:
            logger.info(f"Semantic steering enabled (check frequency: {semantic_check_frequency})")
    
    def generate_code(self, 
                      prompt: str, 
                      llm_generator: Callable,
                      formal_specs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate code using an LLM with syntactic and semantic steering.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_generator: Function that takes a prompt and returns a list of (token, probability) tuples
            formal_specs: Optional list of formal specifications for semantic checks
            
        Returns:
            Dictionary with generation results and metrics
        """
        start_time = time.time()
        
        # Start with empty generated code
        generated_code = ""
        tokens_generated = 0
        
        # Keep track of issues encountered
        all_semantic_issues = []
        
        # Initialize beams for beam search
        # Each beam contains: (code_so_far, score, issues)
        beams = [(generated_code, 0.0, [])]
        
        # Track generation steps for visualization/debugging
        generation_steps = []
        
        while tokens_generated < self.max_tokens:
            new_beams = []
            
            # Process each active beam
            for beam_code, beam_score, beam_issues in beams:
                # Get next token predictions from LLM
                token_probs = llm_generator(prompt + beam_code)
                
                # Apply syntactic steering if enabled
                if self.use_syntactic_steering:
                    self.metrics["syntactic_checks"] += 1
                    original_prob_sum = sum(token_probs.values())
                    token_probs = self.ssm.filter_tokens(beam_code, token_probs)
                    filtered_prob_sum = sum(token_probs.values())
                    
                    # Calculate filter rate for metrics
                    if original_prob_sum > 0:
                        filter_rate = 1.0 - (filtered_prob_sum / original_prob_sum)
                        self.metrics["syntactic_filter_rate"] += filter_rate
                
                # Apply semantic steering if enabled and it's time for a check
                if self.use_semantic_steering and tokens_generated % self.semantic_check_frequency == 0:
                    self.metrics["semantic_checks"] += 1
                    
                    # For top candidates, apply semantic checking
                    top_tokens = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    for token, prob in top_tokens:
                        # Check extended code for semantic issues
                        extended_code = beam_code + token
                        issues = self.sesm.analyze_code_snippet(extended_code, formal_specs)
                        
                        # Calculate penalty based on issues
                        penalty = self.sesm.calculate_penalty(issues)
                        self.metrics["semantic_penalties"].append(penalty)
                        
                        # Adjust beam score with log penalty
                        adjusted_score = beam_score
                        if penalty > 0:
                            penalty_factor = max(0.1, 1.0 - penalty)  # Ensure min 0.1 multiplier
                            adjusted_score = beam_score + (token_probs[token] * penalty_factor)
                        else:
                            adjusted_score = beam_score + token_probs[token]
                        
                        # Add to new beam candidates
                        new_beams.append((extended_code, adjusted_score, beam_issues + issues))
                        
                        if self.verbose and penalty > 0:
                            logger.info(f"Semantic penalty {penalty:.3f} applied for token '{token}'")
                            logger.info(f"Issues: {[str(issue) for issue in issues]}")
                else:
                    # Without semantic checking, just add top tokens to beams
                    top_tokens = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    for token, prob in top_tokens:
                        new_beams.append((beam_code + token, beam_score + prob, beam_issues))
            
            # Sort and prune beams
            if new_beams:
                new_beams.sort(key=lambda x: x[1], reverse=True)  # Sort by score
                beams = new_beams[:self.beam_width]  # Keep top beams
                
                # Log current best beam
                if self.verbose:
                    best_beam = beams[0]
                    logger.info(f"Token {tokens_generated}: Best beam score: {best_beam[1]:.4f}")
                    if len(best_beam[0]) > 40:
                        logger.info(f"Code snippet: ...{best_beam[0][-40:]}")
                    else:
                        logger.info(f"Code snippet: {best_beam[0]}")
                        
                # Store step for visualization
                generation_steps.append({
                    "token": tokens_generated,
                    "best_beam": beams[0][0],
                    "best_score": beams[0][1],
                    "num_issues": len(beams[0][2]),
                    "beam_diversity": len(set(b[0][-1] for b in beams))  # Count unique last tokens
                })
                
                tokens_generated += 1
            else:
                # No valid beams, terminate generation
                logger.warning("No valid beams to continue generation")
                break
                
            # Check for termination conditions
            # 1. End of completion (e.g., function definition complete)
            # 2. Reached specific tokens like EOF or end marker
            best_code = beams[0][0]
            if best_code.endswith("\n\n") or best_code.endswith("```"):
                if self.ssm.is_syntactically_valid(best_code):
                    break
        
        # Get the best beam as final output
        best_beam = max(beams, key=lambda x: x[1])
        generated_code, final_score, final_issues = best_beam
        
        # Track metrics
        self.metrics["generation_time"] = time.time() - start_time
        self.metrics["total_tokens_generated"] = tokens_generated
        
        # Normalize some metrics
        if self.metrics["syntactic_checks"] > 0:
            self.metrics["syntactic_filter_rate"] /= self.metrics["syntactic_checks"]
            
        # Prepare result
        result = {
            "code": generated_code,
            "score": final_score,
            "issues": [issue.to_dict() for issue in final_issues],
            "metrics": self.metrics,
            "generation_steps": generation_steps,
            "is_syntactically_valid": self.ssm.is_syntactically_valid(generated_code) 
                                     if self.use_syntactic_steering else None
        }
        
        logger.info(f"Code generation completed. Generated {tokens_generated} tokens in {self.metrics['generation_time']:.2f}s")
        logger.info(f"Final code is syntactically valid: {result['is_syntactically_valid']}")
        logger.info(f"Final code has {len(final_issues)} semantic issues")
        
        return result
    
    def beam_search_decode(self, 
                           prompt: str, 
                           llm_generator: Callable,
                           formal_specs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform beam search decoding with steering.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_generator: Function that takes a prompt and returns next token probabilities
            formal_specs: Optional list of formal specifications for semantic checks
            
        Returns:
            Dictionary with generation results and metrics
        """
        # This is a more explicit implementation of beam search than generate_code
        start_time = time.time()
        
        # Initialize beams: (code_so_far, cumulative_log_prob, issues)
        beams = [("", 0.0, [])]
        
        # Track generation metrics
        metrics = {
            "syntactic_checks": 0,
            "semantic_checks": 0,
            "steps": 0,
            "filtered_tokens_per_step": []
        }
        
        # Generation loop
        for step in range(self.max_tokens):
            metrics["steps"] = step + 1
            new_beams = []
            
            for beam_idx, (beam_text, beam_score, beam_issues) in enumerate(beams):
                # Get next token predictions from LLM
                token_probs = llm_generator(prompt + beam_text)
                
                # Apply syntactic steering
                if self.use_syntactic_steering:
                    metrics["syntactic_checks"] += 1
                    pre_count = len(token_probs)
                    token_probs = self.ssm.filter_tokens(beam_text, token_probs)
                    post_count = len(token_probs)
                    metrics["filtered_tokens_per_step"].append(pre_count - post_count)
                
                # Perform semantic check for this beam if needed
                run_semantic_check = (
                    self.use_semantic_steering and 
                    step % self.semantic_check_frequency == 0
                )
                
                if run_semantic_check:
                    metrics["semantic_checks"] += 1
                    
                    # For efficiency, we'll only check semantic issues for
                    # the most promising token candidates
                    candidates = []
                    for token, prob in sorted(token_probs.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:10]:
                        # Check the extended sequence for semantic issues
                        extended_text = beam_text + token
                        issues = self.sesm.analyze_code_snippet(extended_text, formal_specs)
                        
                        # Apply penalty based on issues
                        penalty = self.sesm.calculate_penalty(issues)
                        adjusted_prob = prob
                        if penalty > 0:
                            # Log-domain penalty
                            adjusted_prob *= (1.0 - penalty)
                            
                        # Add to candidates
                        candidates.append((token, adjusted_prob, 
                                           beam_score - adjusted_prob, 
                                           beam_issues + issues))
                                           
                    # Add candidates to new beams
                    new_beams.extend(candidates)
                else:
                    # Without semantic checking
                    for token, prob in token_probs.items():
                        new_score = beam_score - prob  # Negative log probability
                        new_beams.append((beam_text + token, new_score, beam_issues))
            
            # Keep only the best beams
            if new_beams:
                # Sort by score (lower is better since we use negative log probability)
                new_beams.sort(key=lambda x: x[2])
                beams = new_beams[:self.beam_width]
            else:
                # No valid continuations
                logger.warning(f"No valid continuations at step {step}")
                break
                
            # Check if we should terminate generation
            # (e.g., all beams have reached an end marker)
            done = True
            for beam_text, _, _ in beams:
                if not (beam_text.endswith("\n\n") or beam_text.endswith("```")):
                    done = False
                    break
            if done:
                break
                
        # Find best beam
        best_beam = min(beams, key=lambda x: x[2])  # Lowest negative log probability
        best_text, best_score, best_issues = best_beam
        
        # Compute final metrics
        metrics["generation_time"] = time.time() - start_time
        metrics["tokens_generated"] = len(best_text.split())
        
        # Return results
        result = {
            "code": best_text,
            "score": -best_score,  # Convert back to positive
            "issues": [issue.to_dict() for issue in best_issues],
            "metrics": metrics,
            "is_syntactically_valid": self.ssm.is_syntactically_valid(best_text) 
                                     if self.use_syntactic_steering else None
        }
        
        return result
    
    def post_process_code(self, code: str) -> str:
        """
        Apply post-processing to the generated code.
        
        Args:
            code: Generated code string
            
        Returns:
            Post-processed code
        """
        # Remove any leading/trailing whitespace
        processed_code = code.strip()
        
        # Make sure the code is syntactically valid
        if self.use_syntactic_steering and not self.ssm.is_syntactically_valid(processed_code):
            # Simple heuristic fixes for common issues
            
            # 1. Check for missing closing brackets/parentheses/braces
            open_chars = {'(': ')', '[': ']', '{': '}'}
            stack = []
            
            for char in processed_code:
                if char in open_chars:
                    stack.append(char)
                elif char in open_chars.values():
                    if stack and open_chars[stack[-1]] == char:
                        stack.pop()
            
            # Add missing closing characters
            while stack:
                processed_code += open_chars[stack.pop()]
                
            # 2. Check for incomplete statements (add semicolons for languages that need them)
            if self.language in ["java", "javascript", "c", "cpp"]:
                lines = processed_code.split('\n')
                for i in range(len(lines)):
                    line = lines[i].strip()
                    if line and not line.endswith(';') and not line.endswith('{') and not line.endswith('}'):
                        if not line.endswith('\\'):  # Not a line continuation
                            lines[i] = line + ';'
                            
                processed_code = '\n'.join(lines)
                
        return processed_code


def mock_llm_generator(prompt: str) -> Dict[str, float]:
    """
    A mock LLM generator for testing the SSCSteer framework.
    In a real implementation, this would call the LLM API.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        Dictionary mapping tokens to probabilities
    """
    # This is just a placeholder that returns random tokens
    import random
    
    tokens = [
        "def", "class", "if", "else", "for", "while", "return", "import",
        "print", "x", "y", "z", "=", "+", "-", "*", "/", "(", ")", "[", "]",
        "{", "}", ":", ",", ".", "0", "1", "2", "a", "b", "c", "'", "\"",
        "True", "False", "None", "\n", "    ", " "
    ]
    
    # Generate probabilities for 10 random tokens
    selected_tokens = random.sample(tokens, 10)
    probs = [random.random() for _ in range(10)]
    total = sum(probs)
    normalized_probs = [p/total for p in probs]
    
    return dict(zip(selected_tokens, normalized_probs))