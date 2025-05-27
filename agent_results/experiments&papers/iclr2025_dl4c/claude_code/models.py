"""
Models for the adaptive code assistant experiment.
Includes baseline and adaptive models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import DeveloperProfile
from utils import DEVICE, timer

logger = logging.getLogger("adaptive_code_assistant.models")

class CodeAssistantModel:
    """Base class for all code assistant models."""
    
    def __init__(
        self,
        model_name: str,
        use_small_model: bool = True
    ):
        """
        Initialize the code assistant model.
        
        Args:
            model_name: Name of the model
            use_small_model: Whether to use a small model for faster experimentation
        """
        self.model_name = model_name
        self.use_small_model = use_small_model
        
        # This is a base class, actual model loading is implemented in subclasses
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized base code assistant model: {model_name}")
    
    def complete_code(self, prompt: str, **kwargs) -> str:
        """
        Generate code completion for a given prompt.
        
        Args:
            prompt: Code prompt to complete
            **kwargs: Additional arguments for code generation
            
        Returns:
            Completed code
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def update(self, prompt: str, feedback: Dict[str, Any], task_type: str = None) -> None:
        """
        Update the model based on feedback.
        This method is a no-op for non-adaptive models.
        
        Args:
            prompt: Prompt that was completed
            feedback: Feedback on the completion
            task_type: Type of coding task for task-specific adaptation
        """
        # Base implementation does nothing
        pass

class StaticLLMCodeAssistant(CodeAssistantModel):
    """
    Static LLM code assistant that doesn't adapt to user feedback.
    This is our first baseline model.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        use_small_model: bool = True
    ):
        """
        Initialize the static LLM code assistant.
        
        Args:
            model_name: Name of the model to use
            use_small_model: Whether to use a small model for faster experimentation
        """
        super().__init__(model_name, use_small_model)
        
        if use_small_model:
            # Use a smaller model for experimentation
            self.model_name = "gpt2" if torch.cuda.is_available() else "distilgpt2"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(DEVICE)
            
            if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded model and tokenizer: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Create a dummy model for simulation
            self.model = None
            self.tokenizer = None
            logger.info("Using simulated model responses")
    
    @timer
    def complete_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        task_type: str = None,  # Added task_type parameter
        **kwargs
    ) -> str:
        """
        Generate code completion using the LLM.
        
        Args:
            prompt: Code prompt to complete
            max_length: Maximum length of the generated code
            temperature: Sampling temperature
            task_type: Task type (unused in this class, but required for API compatibility)
            **kwargs: Additional arguments for generation
            
        Returns:
            Completed code
        """
        if self.model is None or self.tokenizer is None:
            # Simulate a response for demo purposes
            return self._simulate_code_completion(prompt)
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Filter kwargs to remove task_type and other non-generate parameters
        generate_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['task_type']}
        
        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs
            )
        
        # Decode the generated text
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated part
        if prompt in completion:
            completion = completion[len(prompt):]
        
        return completion
    
    def _simulate_code_completion(self, prompt: str) -> str:
        """Simulate code completion for demo purposes."""
        # Extract the task from the prompt
        task_keywords = ["calculate", "find", "check", "reverse", "count", "sort", "convert"]
        task_type = "unknown"
        
        for keyword in task_keywords:
            if keyword in prompt.lower():
                task_type = keyword
                break
        
        # Generate a simulated response based on task type
        responses = {
            "calculate": '''def calculate_result(x, y):
    """Calculate the sum and product of two numbers."""
    return {
        'sum': x + y,
        'product': x * y
    }''',
            "find": '''def find_element(arr, target):
    """Find an element in an array and return its index."""
    for i, elem in enumerate(arr):
        if elem == target:
            return i
    return -1''',
            "check": '''def check_condition(value):
    """Check if a value meets certain conditions."""
    if value < 0:
        return "Negative"
    elif value == 0:
        return "Zero"
    else:
        return "Positive" ''',
            "reverse": '''def reverse_string(s):
    """Reverse a string efficiently."""
    return s[::-1]''',
            "count": '''def count_occurrences(text, char):
    """Count occurrences of a character in a text."""
    count = 0
    for c in text:
        if c == char:
            count += 1
    return count''',
            "sort": '''def custom_sort(arr):
    """Sort an array using a custom algorithm."""
    if not arr:
        return []
    
    # Quick sort implementation
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    
    return custom_sort(less) + [pivot] + custom_sort(greater)''',
            "convert": '''def convert_to_binary(decimal):
    """Convert a decimal number to binary representation."""
    if decimal == 0:
        return "0"
    
    binary = ""
    while decimal > 0:
        binary = str(decimal % 2) + binary
        decimal //= 2
        
    return binary'''
        }
        
        # Return a relevant response or a default one
        return responses.get(task_type, '''def process_data(data):
    """Process the input data and return a result."""
    result = []
    for item in data:
        if isinstance(item, (int, float)):
            result.append(item * 2)
        elif isinstance(item, str):
            result.append(item.upper())
        else:
            result.append(item)
    return result''')

class FineTunedLLMCodeAssistant(StaticLLMCodeAssistant):
    """
    Fine-tuned LLM code assistant.
    This is our second baseline model, which has been fine-tuned
    on general coding patterns but doesn't adapt to individual users.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        use_small_model: bool = True
    ):
        """
        Initialize the fine-tuned LLM code assistant.
        
        Args:
            model_name: Name of the model to use
            use_small_model: Whether to use a small model for faster experimentation
        """
        super().__init__(model_name, use_small_model)
        
        # In a real implementation, this would load a fine-tuned model
        # For simulation, we'll just use additional context to simulate improved performance
        self.fine_tuning_context = {
            "python": "Always include type hints and docstrings. Follow PEP8 guidelines.",
            "javascript": "Use modern ES6+ syntax. Always handle errors properly.",
            "java": "Follow Java naming conventions. Use design patterns where appropriate.",
            "c++": "Optimize for performance. Include proper memory management.",
            "rust": "Maximize safety features. Use pattern matching where appropriate."
        }
        
        logger.info(f"Initialized fine-tuned LLM code assistant with simulated fine-tuning")
    
    @timer
    def complete_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        language: str = "python",
        task_type: str = None,  # Added task_type parameter
        **kwargs
    ) -> str:
        """
        Generate code completion using the fine-tuned LLM.
        
        Args:
            prompt: Code prompt to complete
            max_length: Maximum length of the generated code
            temperature: Sampling temperature
            language: Programming language to use for context
            task_type: Task type (passed to parent implementation)
            **kwargs: Additional arguments for generation
            
        Returns:
            Completed code
        """
        # Add fine-tuning context to prompt
        context = self.fine_tuning_context.get(language, "")
        enhanced_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        # Use the parent class implementation
        completion = super().complete_code(
            enhanced_prompt,
            max_length=max_length,
            temperature=temperature,
            task_type=task_type,
            **kwargs
        )
        
        return completion

class RuleBasedPersonalizationAssistant(CodeAssistantModel):
    """
    Rule-based personalization code assistant.
    This is our third baseline model, which uses manually defined rules for personalization.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        use_small_model: bool = True
    ):
        """
        Initialize the rule-based personalization code assistant.
        
        Args:
            model_name: Name of the model to use
            use_small_model: Whether to use a small model for faster experimentation
        """
        super().__init__(model_name, use_small_model)
        
        # Use the static LLM as base
        self.base_model = StaticLLMCodeAssistant(model_name, use_small_model)
        
        # Rules for personalization
        self.rules = {
            "indentation": {
                "spaces": lambda code: code.replace("\t", "    "),
                "tabs": lambda code: code.replace("    ", "\t")
            },
            "docstring_style": {
                "google": self._convert_to_google_style,
                "numpy": self._convert_to_numpy_style,
                "sphinx": self._convert_to_sphinx_style
            },
            "variable_naming": {
                "snake_case": self._convert_to_snake_case,
                "camelCase": self._convert_to_camel_case
            },
            "brace_style": {
                "same_line": lambda code: code.replace(")\n{", ") {"),
                "new_line": lambda code: code.replace(") {", ")\n{")
            }
        }
        
        # Developer preferences for personalization
        self.developer_preferences = {}
        
        logger.info(f"Initialized rule-based personalization assistant")
    
    def set_developer_preferences(self, developer_profile: DeveloperProfile) -> None:
        """
        Set developer preferences for personalization.
        
        Args:
            developer_profile: Developer profile containing preferences
        """
        self.developer_preferences = {
            "indentation": developer_profile.formatting_preferences["indentation"]["style"],
            "docstring_style": developer_profile.formatting_preferences["docstring_style"],
            "variable_naming": developer_profile.formatting_preferences["variable_naming"],
            "brace_style": developer_profile.formatting_preferences["brace_style"]
        }
        
        logger.info(f"Set preferences for developer: {developer_profile.dev_id}")
    
    @timer
    def complete_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate code completion and apply personalization rules.
        
        Args:
            prompt: Code prompt to complete
            max_length: Maximum length of the generated code
            temperature: Sampling temperature
            **kwargs: Additional arguments for generation
            
        Returns:
            Personalized completed code
        """
        # Get completion from base model
        completion = self.base_model.complete_code(
            prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        # Apply personalization rules
        personalized_completion = self._apply_personalization_rules(completion)
        
        return personalized_completion
    
    def _apply_personalization_rules(self, code: str) -> str:
        """Apply personalization rules to the code."""
        personalized_code = code
        
        for rule_type, rule_value in self.developer_preferences.items():
            if rule_type in self.rules and rule_value in self.rules[rule_type]:
                rule_function = self.rules[rule_type][rule_value]
                personalized_code = rule_function(personalized_code)
        
        return personalized_code
    
    def _convert_to_google_style(self, code: str) -> str:
        """Convert docstrings to Google style."""
        # Simple simulation - real implementation would use regex parsing
        return code.replace('"""Parameters:', '"""Args:')
    
    def _convert_to_numpy_style(self, code: str) -> str:
        """Convert docstrings to NumPy style."""
        # Simple simulation
        return code.replace('"""Args:', '"""Parameters\n    ----------')
    
    def _convert_to_sphinx_style(self, code: str) -> str:
        """Convert docstrings to Sphinx style."""
        # Simple simulation
        return code.replace('"""Args:', '""":param')
    
    def _convert_to_snake_case(self, code: str) -> str:
        """Convert camelCase variable names to snake_case."""
        # Simple simulation - real implementation would use regex
        import re
        
        def replace_camel_with_snake(match):
            return f"{match.group(1)}_{match.group(2).lower()}"
        
        # This is a simplified approach - real-world would need more complex parsing
        pattern = r"([a-z])([A-Z])"
        return re.sub(pattern, replace_camel_with_snake, code)
    
    def _convert_to_camel_case(self, code: str) -> str:
        """Convert snake_case variable names to camelCase."""
        # Simple simulation - real implementation would use regex
        import re
        
        def replace_snake_with_camel(match):
            return f"{match.group(1)}{match.group(2).upper()}"
        
        # This is a simplified approach - real-world would need more complex parsing
        pattern = r"([a-z])_([a-z])"
        return re.sub(pattern, replace_snake_with_camel, code)

class OnlineLearningCodeAssistant(CodeAssistantModel):
    """
    Online Learning code assistant that continuously adapts to user feedback.
    This is our first proposed method.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        use_small_model: bool = True,
        learning_rate: float = 0.01
    ):
        """
        Initialize the Online Learning code assistant.
        
        Args:
            model_name: Name of the model to use
            use_small_model: Whether to use a small model for faster experimentation
            learning_rate: Learning rate for online learning
        """
        super().__init__(model_name, use_small_model)
        
        # Use the static LLM as base
        self.base_model = StaticLLMCodeAssistant(model_name, use_small_model)
        
        # Parameters for online learning
        self.learning_rate = learning_rate
        self.preference_weights = {
            "indentation_style": {"spaces": 0.5, "tabs": 0.5},
            "docstring_style": {"google": 0.33, "numpy": 0.33, "sphinx": 0.34},
            "variable_naming": {"snake_case": 0.5, "camelCase": 0.5},
            "brace_style": {"same_line": 0.5, "new_line": 0.5}
        }
        
        # Preference history for updating weights
        self.preference_history = []
        
        # Rules for applying preferences (similar to rule-based model)
        self.rules = {
            "indentation_style": {
                "spaces": lambda code: code.replace("\t", "    "),
                "tabs": lambda code: code.replace("    ", "\t")
            },
            "docstring_style": {
                "google": self._convert_to_google_style,
                "numpy": self._convert_to_numpy_style,
                "sphinx": self._convert_to_sphinx_style
            },
            "variable_naming": {
                "snake_case": self._convert_to_snake_case,
                "camelCase": self._convert_to_camel_case
            },
            "brace_style": {
                "same_line": lambda code: code.replace(")\n{", ") {"),
                "new_line": lambda code: code.replace(") {", ")\n{")
            }
        }
        
        logger.info(f"Initialized Online Learning code assistant with learning rate {learning_rate}")
    
    @timer
    def complete_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate code completion and apply personalization based on learned preferences.
        
        Args:
            prompt: Code prompt to complete
            max_length: Maximum length of the generated code
            temperature: Sampling temperature
            **kwargs: Additional arguments for generation
            
        Returns:
            Personalized completed code
        """
        # Get completion from base model
        completion = self.base_model.complete_code(
            prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        # Apply personalization based on learned preferences
        personalized_completion = self._apply_learned_preferences(completion)
        
        return personalized_completion
    
    def update(self, prompt: str, feedback: Dict[str, Any], task_type: str = None) -> None:
        """
        Update the model based on feedback using online learning.
        
        Args:
            prompt: Prompt that was completed
            feedback: Feedback on the completion
            task_type: Type of coding task (unused in this implementation)
        """
        if not feedback.get("provided", False):
            # No feedback provided, no update needed
            return
        
        # Extract features from feedback
        # In a real implementation, we would use more sophisticated feature extraction
        feedback_text = feedback.get("feedback_text", "")
        satisfaction = feedback.get("satisfaction", 0.5)
        
        # Parse feedback to identify preferences
        preferences = self._extract_preferences_from_feedback(feedback_text)
        
        # Add to history
        self.preference_history.append({
            "preferences": preferences,
            "satisfaction": satisfaction
        })
        
        # Update weights based on feedback
        for pref_type, pref_value in preferences.items():
            if pref_type in self.preference_weights and pref_value in self.preference_weights[pref_type]:
                # Increase weight for preferred option
                current_weight = self.preference_weights[pref_type][pref_value]
                new_weight = current_weight + self.learning_rate * satisfaction
                
                # Normalize weights to sum to 1
                self.preference_weights[pref_type][pref_value] = new_weight
                self._normalize_weights(pref_type)
        
        logger.info(f"Updated model weights based on feedback (satisfaction: {satisfaction:.2f})")
    
    def _extract_preferences_from_feedback(self, feedback_text: str) -> Dict[str, str]:
        """Extract preferences from feedback text."""
        preferences = {}
        
        # Simple keyword matching - real implementation would use NLP
        keywords = {
            "indentation_style": {
                "spaces": ["spaces", "space", "indent with spaces"],
                "tabs": ["tabs", "tab", "indent with tabs"]
            },
            "docstring_style": {
                "google": ["google style", "google docstring"],
                "numpy": ["numpy style", "numpy docstring"],
                "sphinx": ["sphinx style", "sphinx docstring"]
            },
            "variable_naming": {
                "snake_case": ["snake case", "snake_case", "underscore"],
                "camelCase": ["camel case", "camelCase"]
            },
            "brace_style": {
                "same_line": ["same line", "brace on same"],
                "new_line": ["new line", "brace on new"]
            }
        }
        
        feedback_lower = feedback_text.lower()
        
        for pref_type, pref_options in keywords.items():
            for pref_value, pref_keywords in pref_options.items():
                for keyword in pref_keywords:
                    if keyword in feedback_lower:
                        preferences[pref_type] = pref_value
                        break
        
        return preferences
    
    def _normalize_weights(self, pref_type: str) -> None:
        """Normalize weights for a preference type to sum to 1."""
        total = sum(self.preference_weights[pref_type].values())
        for pref_value in self.preference_weights[pref_type]:
            self.preference_weights[pref_type][pref_value] /= total
    
    def _apply_learned_preferences(self, code: str) -> str:
        """Apply learned preferences to the code."""
        personalized_code = code
        
        for pref_type, weights in self.preference_weights.items():
            # Choose preference based on weights
            options = list(weights.keys())
            probs = list(weights.values())
            choice = random.choices(options, weights=probs, k=1)[0]
            
            # Apply the chosen preference
            if pref_type in self.rules and choice in self.rules[pref_type]:
                rule_function = self.rules[pref_type][choice]
                personalized_code = rule_function(personalized_code)
        
        return personalized_code
    
    # Helper methods for applying preferences (same as rule-based model)
    def _convert_to_google_style(self, code: str) -> str:
        """Convert docstrings to Google style."""
        return code.replace('"""Parameters:', '"""Args:')
    
    def _convert_to_numpy_style(self, code: str) -> str:
        """Convert docstrings to NumPy style."""
        return code.replace('"""Args:', '"""Parameters\n    ----------')
    
    def _convert_to_sphinx_style(self, code: str) -> str:
        """Convert docstrings to Sphinx style."""
        return code.replace('"""Args:', '""":param')
    
    def _convert_to_snake_case(self, code: str) -> str:
        """Convert camelCase variable names to snake_case."""
        import re
        
        def replace_camel_with_snake(match):
            return f"{match.group(1)}_{match.group(2).lower()}"
        
        pattern = r"([a-z])([A-Z])"
        return re.sub(pattern, replace_camel_with_snake, code)
    
    def _convert_to_camel_case(self, code: str) -> str:
        """Convert snake_case variable names to camelCase."""
        import re
        
        def replace_snake_with_camel(match):
            return f"{match.group(1)}{match.group(2).upper()}"
        
        pattern = r"([a-z])_([a-z])"
        return re.sub(pattern, replace_snake_with_camel, code)

class MAMLCodeAssistant(CodeAssistantModel):
    """
    MAML-based code assistant that uses Model-Agnostic Meta-Learning for adaptation.
    This is our second proposed method.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        use_small_model: bool = True,
        meta_learning_rate: float = 0.01,
        inner_learning_rate: float = 0.001
    ):
        """
        Initialize the MAML-based code assistant.
        
        Args:
            model_name: Name of the model to use
            use_small_model: Whether to use a small model for faster experimentation
            meta_learning_rate: Learning rate for meta-learning
            inner_learning_rate: Learning rate for inner loop updates
        """
        super().__init__(model_name, use_small_model)
        
        # Use the static LLM as base
        self.base_model = StaticLLMCodeAssistant(model_name, use_small_model)
        
        # Parameters for MAML
        self.meta_learning_rate = meta_learning_rate
        self.inner_learning_rate = inner_learning_rate
        
        # Task-specific adaptation parameters
        self.task_memory = {}  # Maps task types to adapted parameters
        
        # Current task parameters
        self.current_task_params = self._init_task_params()
        
        logger.info(f"Initialized MAML code assistant with meta-learning rate {meta_learning_rate}")
    
    def _init_task_params(self) -> Dict[str, float]:
        """Initialize task parameters."""
        return {
            "formatting_weight": 0.5,
            "verbosity_level": 0.5,
            "comment_density": 0.5,
            "complexity_preference": 0.5
        }
    
    @timer
    def complete_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        task_type: str = None,
        **kwargs
    ) -> str:
        """
        Generate code completion with task-specific adaptation.
        
        Args:
            prompt: Code prompt to complete
            max_length: Maximum length of the generated code
            temperature: Sampling temperature
            task_type: Type of coding task for task-specific adaptation
            **kwargs: Additional arguments for generation
            
        Returns:
            Adapted completed code
        """
        # Load task-specific parameters if available
        if task_type and task_type in self.task_memory:
            self.current_task_params = self.task_memory[task_type]
            logger.info(f"Loaded parameters for task type: {task_type}")
        
        # Modify prompt based on task parameters
        adapted_prompt = self._adapt_prompt(prompt)
        
        # Adjust generation parameters based on task parameters
        adapted_temperature = temperature * (1.0 + 0.5 * (self.current_task_params["complexity_preference"] - 0.5))
        
        # Get completion from base model
        completion = self.base_model.complete_code(
            adapted_prompt,
            max_length=max_length,
            temperature=adapted_temperature,
            **kwargs
        )
        
        # Apply post-processing based on task parameters
        adapted_completion = self._adapt_completion(completion)
        
        return adapted_completion
    
    def update(self, prompt: str, feedback: Dict[str, Any], task_type: str = None) -> None:
        """
        Update the model based on feedback using MAML-inspired approach.
        
        Args:
            prompt: Prompt that was completed
            feedback: Feedback on the completion
            task_type: Type of coding task for task-specific adaptation
        """
        if not feedback.get("provided", False):
            # No feedback provided, no update needed
            return
        
        satisfaction = feedback.get("satisfaction", 0.5)
        
        # Inner loop update: Adjust current task parameters based on feedback
        self._inner_loop_update(feedback, satisfaction)
        
        # If task_type is provided, store the updated parameters
        if task_type:
            self.task_memory[task_type] = self.current_task_params.copy()
            logger.info(f"Updated and stored parameters for task type: {task_type}")
        
        # Meta-learning update: Update meta-parameters
        # In a real MAML implementation, this would involve meta-gradient computation
        # For simulation, we'll just use a simplified approach
        self._meta_update(feedback, satisfaction)
    
    def _adapt_prompt(self, prompt: str) -> str:
        """Adapt the prompt based on task parameters."""
        # Add hints based on verbosity level
        verbosity = self.current_task_params["verbosity_level"]
        comment_density = self.current_task_params["comment_density"]
        
        if verbosity > 0.7:
            prompt += "\n# Please provide detailed explanations in comments"
        elif verbosity < 0.3:
            prompt += "\n# Keep the code concise"
        
        if comment_density > 0.7:
            prompt += "\n# Include comments for all major steps"
        elif comment_density < 0.3:
            prompt += "\n# Minimize comments"
        
        return prompt
    
    def _adapt_completion(self, completion: str) -> str:
        """Adapt the completion based on task parameters."""
        # Adjust formatting
        formatting_weight = self.current_task_params["formatting_weight"]
        
        if formatting_weight > 0.7:
            # Add extra whitespace for readability
            completion = completion.replace("if ", "if  ")
            completion = completion.replace("for ", "for  ")
            completion = completion.replace("def ", "def  ")
            completion = completion.replace("\n", "\n\n")
        elif formatting_weight < 0.3:
            # Make code more compact
            completion = completion.replace("\n\n", "\n")
        
        return completion
    
    def _inner_loop_update(self, feedback: Dict[str, Any], satisfaction: float) -> None:
        """Inner loop update for task-specific parameters."""
        feedback_text = feedback.get("feedback_text", "").lower()
        
        # Extract adjustment signals from feedback
        adjust_formatting = "format" in feedback_text or "indent" in feedback_text
        adjust_verbosity = "verbose" in feedback_text or "explain" in feedback_text
        adjust_comments = "comment" in feedback_text
        adjust_complexity = "simple" in feedback_text or "complex" in feedback_text
        
        # Update parameters based on feedback
        if adjust_formatting:
            direction = 1 if "more" in feedback_text else -1
            self.current_task_params["formatting_weight"] += direction * self.inner_learning_rate
            self.current_task_params["formatting_weight"] = max(0, min(1, self.current_task_params["formatting_weight"]))
        
        if adjust_verbosity:
            direction = 1 if "more" in feedback_text else -1
            self.current_task_params["verbosity_level"] += direction * self.inner_learning_rate
            self.current_task_params["verbosity_level"] = max(0, min(1, self.current_task_params["verbosity_level"]))
        
        if adjust_comments:
            direction = 1 if "more" in feedback_text else -1
            self.current_task_params["comment_density"] += direction * self.inner_learning_rate
            self.current_task_params["comment_density"] = max(0, min(1, self.current_task_params["comment_density"]))
        
        if adjust_complexity:
            direction = 1 if "complex" in feedback_text else -1
            self.current_task_params["complexity_preference"] += direction * self.inner_learning_rate
            self.current_task_params["complexity_preference"] = max(0, min(1, self.current_task_params["complexity_preference"]))
        
        logger.info(f"Inner loop update completed with satisfaction: {satisfaction:.2f}")
    
    def _meta_update(self, feedback: Dict[str, Any], satisfaction: float) -> None:
        """Meta-learning update for meta-parameters."""
        # In a real MAML implementation, this would update meta-parameters
        # For simulation, we'll just adjust the learning rates
        if satisfaction > 0.7:
            # Increase learning rates for faster adaptation
            self.inner_learning_rate *= 1.1
            logger.info(f"Increased inner learning rate to {self.inner_learning_rate:.4f} due to high satisfaction")
        elif satisfaction < 0.3:
            # Decrease learning rates for more stable adaptation
            self.inner_learning_rate *= 0.9
            logger.info(f"Decreased inner learning rate to {self.inner_learning_rate:.4f} due to low satisfaction")

class HybridAdaptiveCodeAssistant(CodeAssistantModel):
    """
    Hybrid Adaptive code assistant that combines online learning with MAML.
    This is our third proposed method.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        use_small_model: bool = True,
        online_learning_rate: float = 0.01,
        meta_learning_rate: float = 0.01,
        inner_learning_rate: float = 0.001
    ):
        """
        Initialize the Hybrid Adaptive code assistant.
        
        Args:
            model_name: Name of the model to use
            use_small_model: Whether to use a small model for faster experimentation
            online_learning_rate: Learning rate for online learning
            meta_learning_rate: Learning rate for meta-learning
            inner_learning_rate: Learning rate for inner loop updates
        """
        super().__init__(model_name, use_small_model)
        
        # Initialize the component models
        self.online_model = OnlineLearningCodeAssistant(
            model_name,
            use_small_model,
            learning_rate=online_learning_rate
        )
        
        self.maml_model = MAMLCodeAssistant(
            model_name,
            use_small_model,
            meta_learning_rate=meta_learning_rate,
            inner_learning_rate=inner_learning_rate
        )
        
        # Weighting factor for combining outputs
        self.blend_factor = 0.5  # Start with equal weighting
        
        # Adaptation rate for blend factor
        self.blend_adaptation_rate = 0.05
        
        logger.info(f"Initialized Hybrid Adaptive code assistant with blend factor {self.blend_factor}")
    
    @timer
    def complete_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        task_type: str = None,
        **kwargs
    ) -> str:
        """
        Generate code completion using hybrid approach.
        
        Args:
            prompt: Code prompt to complete
            max_length: Maximum length of the generated code
            temperature: Sampling temperature
            task_type: Type of coding task for task-specific adaptation
            **kwargs: Additional arguments for generation
            
        Returns:
            Adapted completed code
        """
        # Get completions from both models
        online_completion = self.online_model.complete_code(
            prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        maml_completion = self.maml_model.complete_code(
            prompt,
            max_length=max_length,
            temperature=temperature,
            task_type=task_type,
            **kwargs
        )
        
        # Combine completions
        # In a real implementation, this would use a more sophisticated approach
        # For simulation, we'll use a simple template-based approach
        combined_completion = self._combine_completions(
            online_completion,
            maml_completion,
            self.blend_factor
        )
        
        return combined_completion
    
    def update(self, prompt: str, feedback: Dict[str, Any], task_type: str = None) -> None:
        """
        Update both models based on feedback.
        
        Args:
            prompt: Prompt that was completed
            feedback: Feedback on the completion
            task_type: Type of coding task for task-specific adaptation
        """
        if not feedback.get("provided", False):
            # No feedback provided, no update needed
            return
        
        satisfaction = feedback.get("satisfaction", 0.5)
        
        # Update both models
        self.online_model.update(prompt, feedback, task_type)
        self.maml_model.update(prompt, feedback, task_type)
        
        # Update blend factor based on feedback
        self._update_blend_factor(feedback)
        
        logger.info(f"Updated both models with satisfaction: {satisfaction:.2f}")
    
    def _combine_completions(self, online_completion: str, maml_completion: str, blend_factor: float) -> str:
        """
        Combine completions from both models.
        
        Args:
            online_completion: Completion from online learning model
            maml_completion: Completion from MAML model
            blend_factor: Factor for blending (0.0 = online only, 1.0 = MAML only)
            
        Returns:
            Combined completion
        """
        # For simulation, we'll use a simple approach
        # In a real implementation, this would use a more sophisticated approach
        
        # Split completions into lines
        online_lines = online_completion.split("\n")
        maml_lines = maml_completion.split("\n")
        
        # Determine lines to keep from each model
        n_online = max(1, int(len(online_lines) * (1 - blend_factor)))
        n_maml = max(1, int(len(maml_lines) * blend_factor))
        
        # Combine lines
        combined_lines = []
        
        # Take some lines from online model
        combined_lines.extend(online_lines[:n_online])
        
        # Take some lines from MAML model
        # Avoid duplicating imports and function signatures
        maml_start_idx = 0
        for i, line in enumerate(maml_lines):
            if "def " in line or "class " in line:
                maml_start_idx = i
                break
        
        combined_lines.extend(maml_lines[maml_start_idx:maml_start_idx+n_maml])
        
        # Join lines
        return "\n".join(combined_lines)
    
    def _update_blend_factor(self, feedback: Dict[str, Any]) -> None:
        """
        Update blend factor based on feedback.
        
        Args:
            feedback: Feedback on the completion
        """
        satisfaction = feedback.get("satisfaction", 0.5)
        
        # Adjust blend factor
        if satisfaction < 0.4:
            # If satisfaction is low, shift more towards the other model
            self.blend_factor = 1 - self.blend_factor
            self.blend_factor = min(0.9, max(0.1, self.blend_factor))
        elif satisfaction > 0.6:
            # If satisfaction is high, make a small adjustment towards current model
            adjustment = self.blend_adaptation_rate * satisfaction
            self.blend_factor += adjustment if self.blend_factor > 0.5 else -adjustment
            self.blend_factor = min(0.9, max(0.1, self.blend_factor))
        
        logger.info(f"Updated blend factor to {self.blend_factor:.2f}")

def get_model(model_name: str, use_small_model: bool = True) -> CodeAssistantModel:
    """
    Get a code assistant model by name.
    
    Args:
        model_name: Name of the model to get
        use_small_model: Whether to use a small model for faster experimentation
        
    Returns:
        CodeAssistantModel instance
    """
    models = {
        "static": StaticLLMCodeAssistant(use_small_model=use_small_model),
        "fine_tuned": FineTunedLLMCodeAssistant(use_small_model=use_small_model),
        "rule_based": RuleBasedPersonalizationAssistant(use_small_model=use_small_model),
        "online": OnlineLearningCodeAssistant(use_small_model=use_small_model),
        "maml": MAMLCodeAssistant(use_small_model=use_small_model),
        "hybrid": HybridAdaptiveCodeAssistant(use_small_model=use_small_model)
    }
    
    if model_name not in models:
        logger.warning(f"Model {model_name} not found, defaulting to static")
        return models["static"]
    
    logger.info(f"Created model: {model_name}")
    return models[model_name]