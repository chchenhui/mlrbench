"""
Syntactic Steering Module (SSM) for LLM code generation.

This module uses the target language's context-free grammar to ensure
syntactically valid code generation by filtering out invalid tokens.
"""

import ast
import re
import tokenize
from io import StringIO
from typing import List, Dict, Set, Union, Optional, Tuple, Any

class PythonSyntaxAnalyzer:
    """
    A lightweight Python syntax analyzer that can determine valid next tokens 
    based on partial code.
    """
    
    def __init__(self):
        """Initialize the Python syntax analyzer."""
        # Common Python keywords
        self.keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
        }
        
        # Token type patterns for common Python contexts
        self.context_patterns = {
            "block_start": re.compile(r':\s*$'),  # After a colon expecting indented block
            "function_def": re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:?\s*$'),
            "class_def": re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))?\s*:?\s*$'),
            "import_stmt": re.compile(r'(import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)(\s+import)?\s*$'),
            "assignment": re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*$'),
            "function_call": re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*$'),
        }
        
    def is_partial_complete(self, code: str) -> bool:
        """
        Check if the partial code can be parsed without syntax errors.
        
        Args:
            code: Partial Python code string
            
        Returns:
            True if the code can be parsed, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def get_token_context(self, code: str) -> Dict[str, Any]:
        """
        Analyze code to determine the current context for token prediction.
        
        Args:
            code: Partial Python code string
            
        Returns:
            Dictionary with context information
        """
        context = {"type": "unknown", "details": {}}
        
        # Check for specific patterns
        for context_type, pattern in self.context_patterns.items():
            match = pattern.search(code)
            if match:
                context["type"] = context_type
                context["details"] = match.groupdict() if match.groupdict() else {
                    "match": match.group(0),
                    "groups": match.groups()
                }
                break
                
        # Check indentation level
        lines = code.split('\n')
        if lines:
            last_line = lines[-1]
            context["indentation"] = len(last_line) - len(last_line.lstrip())
            
        # Check if we're in a string
        in_string = False
        string_delim = None
        
        # Simple string detection (not perfect but sufficient for most cases)
        for i, char in enumerate(code):
            if char in ('"', "'") and (i == 0 or code[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_delim = char
                elif char == string_delim:
                    in_string = False
                    string_delim = None
        
        context["in_string"] = in_string
        
        # Try to determine if this is part of a specific statement type
        try:
            # Add an arbitrary valid statement to avoid SyntaxError for incomplete code
            temp_code = code + "\npass"
            tree = ast.parse(temp_code)
            if tree.body:
                context["last_node_type"] = type(tree.body[-2]).__name__  # -2 to skip the pass we added
        except SyntaxError:
            pass
            
        return context
    
    def get_valid_next_tokens(self, code: str, all_tokens: List[str]) -> List[str]:
        """
        Determine which tokens are valid to follow the given partial code.
        
        Args:
            code: Partial Python code string
            all_tokens: List of all possible tokens
            
        Returns:
            List of valid next tokens
        """
        context = self.get_token_context(code)
        valid_tokens = []
        
        # Special handling based on context
        if context["in_string"]:
            # In a string, any token is technically valid
            return all_tokens
            
        # Block start (after colon)
        if context["type"] == "block_start":
            # After a block start, we typically expect indentation and a statement
            for token in all_tokens:
                # Whitespace tokens are valid for indentation
                if token.strip() == '' or token.startswith('    '):
                    valid_tokens.append(token)
                # Keywords that typically start blocks
                elif token in {'def', 'class', 'if', 'else', 'elif', 'try', 'except', 
                               'finally', 'for', 'while', 'with', 'pass', 'return',
                               'raise', 'assert', 'import', 'from'}:
                    valid_tokens.append(token)
                    
        # Function definition
        elif context["type"] == "function_def":
            if not code.rstrip().endswith(':'):
                valid_tokens.append(':')
            else:
                # After function definition colon, expect newline + indentation
                valid_tokens.extend(['\n', '\n    '])
                
        # Inside function call parentheses
        elif context["type"] == "function_call":
            # Allow anything that could be a function argument
            for token in all_tokens:
                if token in {')', ',', 'None', 'True', 'False'} or token.isalnum() or token.strip() == '':
                    valid_tokens.append(token)
                    
        # Default fallback: try adding each token and check if it maintains parsability
        else:
            for token in all_tokens:
                # Skip obviously incorrect tokens based on basic rules
                if not self._basic_filter(code, token, context):
                    continue
                    
                # Try to add the token and see if it maintains partial parsability
                test_code = code + token
                
                # For specific token types, add a temporary suffix to make it parsable
                if token.strip() in {'{', '(', '[', 'if', 'def', 'class', 'for', 'while'}:
                    test_code += " pass" if token.strip() in {'if', 'def', 'class', 'for', 'while'} else ")"
                
                try:
                    self._is_potentially_valid(test_code)
                    valid_tokens.append(token)
                except SyntaxError:
                    # Some tokens might be valid even if they create a temporary syntax error
                    # because they're the start of a multi-token construct
                    if token.strip() in {'def', 'class', 'if', 'for', 'while', 'with', 'try', 'except', 'elif', 'else'}:
                        valid_tokens.append(token)
        
        # If no tokens were found valid, return a subset of generally safe tokens
        if not valid_tokens:
            return [t for t in all_tokens if t.strip() in {'pass', '\n', ' '} or t.strip() == '']
            
        return valid_tokens
    
    def _basic_filter(self, code: str, token: str, context: Dict[str, Any]) -> bool:
        """Apply basic filtering rules based on context."""
        # Empty code can take any token
        if not code.strip():
            return True
            
        # Check indentation consistency
        if '\n' in token and context.get("indentation") is not None:
            next_line = token.split('\n')[-1]
            if next_line.strip() and len(next_line) - len(next_line.lstrip()) != context["indentation"]:
                # Indentation mismatch, but allow if it's more (new block)
                if len(next_line) - len(next_line.lstrip()) <= context["indentation"]:
                    return False
                    
        # Check bracket/parenthesis balance
        if token in {')', '}', ']'}:
            opens = {'(': ')', '{': '}', '[': ']'}
            stack = []
            for char in code:
                if char in opens:
                    stack.append(char)
                elif char in opens.values():
                    if not stack or opens[stack.pop()] != char:
                        return False
            if not stack and token in opens.values():
                return False
                
        return True
    
    def _is_potentially_valid(self, code: str) -> bool:
        """
        Check if code could potentially be valid syntax if completed.
        More permissive than is_partial_complete.
        """
        # Try to parse it directly first
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            # Specific error messages that indicate the code is invalid regardless of completion
            fatal_errors = [
                "invalid syntax",
                "unmatched ')'",
                "unmatched '}'",
                "unmatched ']'",
                "unexpected EOF while parsing",
            ]
            
            # If it's a fatal error message, it's not potentially valid
            if any(msg in str(e) for msg in fatal_errors):
                if "unexpected EOF while parsing" in str(e):
                    # EOF error is sometimes recoverable
                    return True
                return False
            
            # Otherwise, it might be valid when completed
            return True


class NimSyntaxAnalyzer:
    """
    A basic syntax analyzer for Nim language (low-resource language example).
    Uses simpler rules than the Python analyzer.
    """
    
    def __init__(self):
        self.keywords = {
            'addr', 'and', 'as', 'asm', 'bind', 'block', 'break', 'case', 'cast',
            'concept', 'const', 'continue', 'converter', 'defer', 'discard',
            'distinct', 'div', 'do', 'elif', 'else', 'end', 'enum', 'except', 'export',
            'finally', 'for', 'from', 'func', 'if', 'import', 'in', 'include',
            'interface', 'is', 'isnot', 'iterator', 'let', 'macro', 'method', 'mixin',
            'mod', 'nil', 'not', 'notin', 'object', 'of', 'or', 'out', 'proc',
            'ptr', 'raise', 'ref', 'return', 'shl', 'shr', 'static', 'template',
            'try', 'tuple', 'type', 'using', 'var', 'when', 'while', 'xor', 'yield'
        }
        
    def get_valid_next_tokens(self, code: str, all_tokens: List[str]) -> List[str]:
        """
        Very basic implementation for Nim that relies on simple rules.
        For a real implementation, this would use a proper Nim parser.
        """
        valid_tokens = []
        
        # Simple checks for obvious patterns
        for token in all_tokens:
            # Check for basic balance of parentheses/brackets
            if token in {')', '}', ']'}:
                opens = {'(': ')', '{': '}', '[': ']'}
                stack = []
                for char in code:
                    if char in opens:
                        stack.append(char)
                    elif char in opens.values():
                        if stack and opens[stack[-1]] == char:
                            stack.pop()
                
                # Only allow closing brackets if there's a matching open bracket
                if not stack or opens[stack[-1]] != token:
                    continue
            
            valid_tokens.append(token)
            
        return valid_tokens


class JavaSyntaxAnalyzer:
    """
    A basic syntax analyzer for Java language.
    """
    
    def __init__(self):
        self.keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
            'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
            'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
            'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
            'package', 'private', 'protected', 'public', 'return', 'short', 'static',
            'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
            'transient', 'try', 'void', 'volatile', 'while'
        }
        
    def get_valid_next_tokens(self, code: str, all_tokens: List[str]) -> List[str]:
        """
        Basic implementation for Java that relies on simple rules.
        For a real implementation, this would use a proper Java parser.
        """
        valid_tokens = []
        
        # Very basic implementation that just checks for bracket balance
        for token in all_tokens:
            # Check for basic balance of parentheses/brackets/braces
            if token in {')', '}', ']'}:
                opens = {'(': ')', '{': '}', '[': ']'}
                stack = []
                for char in code:
                    if char in opens:
                        stack.append(char)
                    elif char in opens.values():
                        if stack and opens[stack[-1]] == char:
                            stack.pop()
                
                # Only allow closing brackets if there's a matching open bracket
                if not stack or opens[stack[-1]] != token:
                    continue
            
            valid_tokens.append(token)
            
        return valid_tokens


class SyntacticSteeringModule:
    """
    Main Syntactic Steering Module that interfaces with LLMs to ensure
    syntactic validity of generated code.
    """
    
    def __init__(self, language: str = "python"):
        """
        Initialize the SSM for a specific language.
        
        Args:
            language: Target programming language ("python", "java", or "nim")
        """
        self.language = language.lower()
        
        # Select the appropriate analyzer based on language
        if self.language == "python":
            self.analyzer = PythonSyntaxAnalyzer()
        elif self.language == "java":
            self.analyzer = JavaSyntaxAnalyzer()
        elif self.language == "nim":
            self.analyzer = NimSyntaxAnalyzer()
        else:
            raise ValueError(f"Unsupported language: {language}")
            
    def filter_tokens(self, partial_code: str, token_probs: Dict[str, float]) -> Dict[str, float]:
        """
        Filter and renormalize token probabilities based on syntactic validity.
        
        Args:
            partial_code: The code generated so far
            token_probs: Dictionary mapping tokens to their probabilities
            
        Returns:
            Filtered and renormalized token probabilities
        """
        # Get list of all tokens with non-zero probability
        all_tokens = list(token_probs.keys())
        
        # Get valid next tokens
        valid_tokens = self.analyzer.get_valid_next_tokens(partial_code, all_tokens)
        
        # Filter probabilities to only include valid tokens
        filtered_probs = {token: prob for token, prob in token_probs.items() 
                         if token in valid_tokens}
        
        # If no valid tokens remain, return the original distribution
        if not filtered_probs:
            return token_probs
        
        # Renormalize probabilities
        total_prob = sum(filtered_probs.values())
        normalized_probs = {token: prob / total_prob for token, prob in filtered_probs.items()}
        
        return normalized_probs
    
    def is_syntactically_valid(self, code: str) -> bool:
        """
        Check if the complete code is syntactically valid.
        
        Args:
            code: Complete code string
            
        Returns:
            True if the code is syntactically valid, False otherwise
        """
        if self.language == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        else:
            # For other languages, we'd need to use external tools
            # This is a simplified implementation
            return True