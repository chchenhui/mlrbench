"""
Counterexample to natural language feedback translator for ContractGPT.

This module translates counterexamples from the SMT solver into natural language
feedback that can be used to guide the LLM in refining code.
"""

from typing import Dict, List, Any

from models.static_analyzer import CounterExample


class FeedbackTranslator:
    """
    Translator for converting counterexamples to natural language feedback.
    """
    
    def __init__(self):
        """Initialize the feedback translator."""
        pass
    
    def translate(self, counterexamples: List[CounterExample]) -> str:
        """
        Translate a list of counterexamples to natural language feedback.
        
        Args:
            counterexamples: List of counterexamples from the static analyzer.
            
        Returns:
            Natural language feedback string describing the counterexamples.
        """
        if not counterexamples:
            return "No issues found."
        
        feedback_parts = []
        
        for i, cex in enumerate(counterexamples):
            # Get the input values
            inputs_str = self._format_inputs(cex.input_values)
            
            # Format the feedback based on the verification condition name
            if "assertion" in cex.vc_name:
                feedback = f"Assertion failure with inputs: {inputs_str}."
            elif "precondition" in cex.vc_name:
                feedback = f"Precondition not satisfied with inputs: {inputs_str}."
            elif "postcondition" in cex.vc_name:
                feedback = f"Postcondition not satisfied with inputs: {inputs_str}."
            elif "contract" in cex.vc_name:
                feedback = (
                    f"Contract violation detected with inputs: {inputs_str}. "
                    f"The implementation does not meet the specification."
                )
            else:
                feedback = f"Verification condition '{cex.vc_name}' failed with inputs: {inputs_str}."
            
            feedback_parts.append(f"{i+1}. {feedback}")
        
        return "\n".join(feedback_parts)
    
    def _format_inputs(self, input_values: Dict[str, Any]) -> str:
        """
        Format input values for display in feedback.
        
        Args:
            input_values: Dictionary of variable assignments.
            
        Returns:
            Formatted string of input values.
        """
        parts = []
        for var_name, value in input_values.items():
            # Format the value based on its type
            if hasattr(value, "as_long"):
                # Z3 integer
                formatted_value = str(value.as_long())
            elif hasattr(value, "as_decimal"):
                # Z3 real
                formatted_value = value.as_decimal(10)
            elif hasattr(value, "is_true") and hasattr(value, "is_false"):
                # Z3 boolean
                formatted_value = "true" if value.is_true() else "false"
            else:
                # Other types
                formatted_value = str(value)
            
            parts.append(f"{var_name}={formatted_value}")
        
        return ", ".join(parts)


def translate_counterexamples(counterexamples: List[CounterExample]) -> str:
    """
    Translate counterexamples to natural language feedback.
    
    Args:
        counterexamples: List of counterexamples from the static analyzer.
        
    Returns:
        Natural language feedback string describing the counterexamples.
    """
    translator = FeedbackTranslator()
    return translator.translate(counterexamples)