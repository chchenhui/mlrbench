"""
DSL parser for ContractGPT function contracts.

This module implements a parser for the domain-specific language (DSL) used to specify
function contracts (preconditions and postconditions).
"""

import re
import ast
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class Contract:
    """Represents a function contract with preconditions and postconditions."""
    preconditions: List[str]
    postconditions: List[str]
    raw_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the contract to a dictionary."""
        return {
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "raw_text": self.raw_text
        }
    
    def __str__(self) -> str:
        """String representation of the contract."""
        pre_str = " && ".join(self.preconditions)
        post_str = " && ".join(self.postconditions)
        return f"requires {pre_str}\nensures {post_str}"


class DSLParser:
    """Parser for the ContractGPT DSL."""
    
    def __init__(self):
        """Initialize the DSL parser."""
        # Define regex patterns for parsing
        self.requires_pattern = r"requires\s+(.*?)(?=\s+ensures|\s*$)"
        self.ensures_pattern = r"ensures\s+(.*?)(?=\s+requires|\s*$)"
    
    def parse(self, spec_text: str) -> Contract:
        """
        Parse a specification string into a Contract object.
        
        Args:
            spec_text: String containing the specification in DSL format.
            
        Returns:
            Contract object with parsed preconditions and postconditions.
        """
        # Clean up the text
        cleaned_text = self._preprocess_text(spec_text)
        
        # Extract preconditions
        pre_matches = re.findall(self.requires_pattern, cleaned_text, re.DOTALL)
        preconditions = []
        for match in pre_matches:
            # Split by && if there are multiple conditions
            conditions = [c.strip() for c in match.split("&&")]
            preconditions.extend(conditions)
        
        # Extract postconditions
        post_matches = re.findall(self.ensures_pattern, cleaned_text, re.DOTALL)
        postconditions = []
        for match in post_matches:
            # Split by && if there are multiple conditions
            conditions = [c.strip() for c in match.split("&&")]
            postconditions.extend(conditions)
        
        return Contract(
            preconditions=preconditions,
            postconditions=postconditions,
            raw_text=cleaned_text
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess the text to handle formatting issues."""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        # Add space after keywords if missing
        text = re.sub(r'requires(?!\s)', 'requires ', text)
        text = re.sub(r'ensures(?!\s)', 'ensures ', text)
        return text.strip()


def parse_dsl(spec_text: str) -> Contract:
    """
    Convenience function to parse a DSL specification string.
    
    Args:
        spec_text: String containing the specification in DSL format.
        
    Returns:
        Contract object with parsed preconditions and postconditions.
    """
    parser = DSLParser()
    return parser.parse(spec_text)