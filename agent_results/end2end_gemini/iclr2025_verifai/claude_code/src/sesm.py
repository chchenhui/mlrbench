"""
Semantic Steering Module (SeSM) for LLM code generation.

This module implements lightweight static analysis and SMT solving
to identify potential semantic issues in partially generated code.
"""

import ast
import re
import sympy
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from z3 import Solver, Optimize, Int, Real, Bool, And, Or, Not, If, Implies

class SemanticIssue:
    """
    Represents a semantic issue identified in the code.
    """
    
    def __init__(self, issue_type: str, severity: float, message: str, 
                 location: Optional[Tuple[int, int]] = None):
        """
        Initialize a semantic issue.
        
        Args:
            issue_type: Type of the semantic issue (e.g., "null_dereference")
            severity: Severity score between 0.0 and 1.0 (higher = more severe)
            message: Description of the issue
            location: Optional tuple of (line, column) where the issue occurs
        """
        self.issue_type = issue_type
        self.severity = max(0.0, min(1.0, severity))  # Clamp between 0 and 1
        self.message = message
        self.location = location
    
    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.issue_type}] {self.message}{loc} (severity: {self.severity:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue to a dictionary representation."""
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "location": self.location
        }


class PythonStaticAnalyzer:
    """
    Lightweight static analyzer for Python code.
    """
    
    def __init__(self):
        """Initialize the Python static analyzer."""
        # Types of issues to check for
        self.issue_checks = {
            "uninitialized_variable": self._check_uninitialized_variables,
            "null_dereference": self._check_null_dereference,
            "index_out_of_bounds": self._check_index_out_of_bounds,
            "division_by_zero": self._check_division_by_zero,
            "resource_leak": self._check_resource_leak,
            "unused_variable": self._check_unused_variables
        }
        
    def analyze(self, code: str) -> List[SemanticIssue]:
        """
        Analyze Python code for semantic issues.
        
        Args:
            code: Python code string
            
        Returns:
            List of SemanticIssue objects identified in the code
        """
        issues = []
        
        # Try to parse the code using AST
        try:
            tree = ast.parse(code)
            
            # Apply each semantic check
            for check_name, check_func in self.issue_checks.items():
                issues.extend(check_func(tree))
                
        except SyntaxError:
            # If the code can't be parsed, we can't perform proper semantic analysis
            # We could attempt to analyze the code up to the syntax error, but for
            # simplicity, we'll just return an empty list
            pass
            
        return issues
    
    def _check_uninitialized_variables(self, tree: ast.AST) -> List[SemanticIssue]:
        """Check for variables being used before initialization."""
        issues = []
        
        # Track variables that have been defined
        defined_vars = set()
        
        class UninitializedVarVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id not in defined_vars:
                    # Skip builtins and common imports
                    if node.id not in __builtins__ and node.id not in {
                        'np', 'numpy', 'pd', 'pandas', 'plt', 'matplotlib',
                        'os', 'sys', 'math', 're', 'random', 'time', 'datetime'
                    }:
                        issues.append(SemanticIssue(
                            "uninitialized_variable", 
                            0.7, 
                            f"Variable '{node.id}' might be used before initialization",
                            (node.lineno, node.col_offset)
                        ))
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                # Process right side first (in case of a = a + 1)
                self.visit(node.value)
                
                # Then mark variables as defined
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
                    elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined_vars.add(elt.id)
                
            def visit_FunctionDef(self, node):
                # Add parameters to defined variables
                for arg in node.args.args:
                    defined_vars.add(arg.arg)
                    
                # Add function name to defined variables
                defined_vars.add(node.name)
                
                self.generic_visit(node)
        
        visitor = UninitializedVarVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _check_null_dereference(self, tree: ast.AST) -> List[SemanticIssue]:
        """Check for potential None/null dereferences."""
        issues = []
        
        class NullDereferenceVisitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                # Look for patterns like x.y where x might be None
                # This is simplistic - a real analysis would track dataflow
                if isinstance(node.value, ast.Name):
                    # Check if there's a None check before this line
                    None_check_pattern = f"if {node.value.id} is not None" 
                    None_check_pattern2 = f"if {node.value.id} is None"
                    
                    # If no None check was found nearby, flag it
                    # (This is a very basic heuristic - a real analyzer would be more sophisticated)
                    if None_check_pattern not in code[:node.lineno*80] and None_check_pattern2 not in code[:node.lineno*80]:
                        issues.append(SemanticIssue(
                            "null_dereference", 
                            0.5, 
                            f"Potential null dereference on '{node.value.id}.{node.attr}'",
                            (node.lineno, node.col_offset)
                        ))
                        
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # Look for method calls on objects that might be None
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    var_name = node.func.value.id
                    # Similar simple check as above
                    None_check_pattern = f"if {var_name} is not None"
                    None_check_pattern2 = f"if {var_name} is None"
                    
                    if None_check_pattern not in code[:node.lineno*80] and None_check_pattern2 not in code[:node.lineno*80]:
                        issues.append(SemanticIssue(
                            "null_dereference", 
                            0.5, 
                            f"Potential null dereference on '{var_name}.{node.func.attr}()'",
                            (node.lineno, node.col_offset)
                        ))
                        
                self.generic_visit(node)
        
        visitor = NullDereferenceVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _check_index_out_of_bounds(self, tree: ast.AST) -> List[SemanticIssue]:
        """Check for potential index out of bounds errors."""
        issues = []
        
        class IndexBoundsVisitor(ast.NodeVisitor):
            def visit_Subscript(self, node):
                # Check for negative constant indices
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    if node.slice.value < 0:
                        # Could be valid but flag as potential issue if there's no length check
                        issues.append(SemanticIssue(
                            "index_out_of_bounds", 
                            0.3, 
                            f"Negative index {node.slice.value} might cause out-of-bounds access if collection is empty",
                            (node.lineno, node.col_offset)
                        ))
                        
                # Check for variable indices without bounds checking
                elif isinstance(node.slice, ast.Name):
                    idx_var = node.slice.id
                    # Very basic check for length comparison - a real analyzer would be more sophisticated
                    length_check_pattern = f"if {idx_var} < len("
                    if length_check_pattern not in code[:node.lineno*80]:
                        issues.append(SemanticIssue(
                            "index_out_of_bounds", 
                            0.4, 
                            f"Index '{idx_var}' used without bounds checking",
                            (node.lineno, node.col_offset)
                        ))
                        
                self.generic_visit(node)
        
        visitor = IndexBoundsVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _check_division_by_zero(self, tree: ast.AST) -> List[SemanticIssue]:
        """Check for potential division by zero errors."""
        issues = []
        
        class DivisionByZeroVisitor(ast.NodeVisitor):
            def visit_BinOp(self, node):
                # Check for division operation
                if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                    # Check for constant divisor of 0
                    if isinstance(node.right, ast.Constant) and node.right.value == 0:
                        issues.append(SemanticIssue(
                            "division_by_zero", 
                            1.0,  # Highest severity - this is a definite error
                            "Division by zero",
                            (node.lineno, node.col_offset)
                        ))
                    
                    # Check for variable divisor without zero check
                    elif isinstance(node.right, ast.Name):
                        zero_check_pattern = f"if {node.right.id} != 0"
                        zero_check_pattern2 = f"if {node.right.id} == 0"
                        
                        if zero_check_pattern not in code[:node.lineno*80] and zero_check_pattern2 not in code[:node.lineno*80]:
                            issues.append(SemanticIssue(
                                "division_by_zero", 
                                0.6, 
                                f"Potential division by zero with variable '{node.right.id}'",
                                (node.lineno, node.col_offset)
                            ))
                            
                self.generic_visit(node)
        
        visitor = DivisionByZeroVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _check_resource_leak(self, tree: ast.AST) -> List[SemanticIssue]:
        """Check for potential resource leaks (files, connections, etc.)."""
        issues = []
        
        # Set of resource opening functions
        resource_openers = {
            'open', 'socket.socket', 'connection', 'connect',
            'sqlite3.connect', 'psycopg2.connect', 'mysql.connector.connect'
        }
        
        class ResourceLeakVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                # Check if a resource is being assigned
                if isinstance(node.value, ast.Call):
                    func_name = ""
                    if isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                    elif isinstance(node.value.func, ast.Attribute):
                        func_name = f"{node.value.func.value.id}.{node.value.func.attr}" \
                            if isinstance(node.value.func.value, ast.Name) else node.value.func.attr
                    
                    # Check if it's a resource opener
                    if any(opener in func_name for opener in resource_openers):
                        # Check if there's a with statement or close method call
                        if isinstance(node.targets[0], ast.Name):
                            resource_var = node.targets[0].id
                            with_pattern = f"with {resource_var}"
                            close_pattern = f"{resource_var}.close"
                            
                            if with_pattern not in code and close_pattern not in code:
                                issues.append(SemanticIssue(
                                    "resource_leak", 
                                    0.7, 
                                    f"Resource '{resource_var}' opened but might not be closed",
                                    (node.lineno, node.col_offset)
                                ))
                                
                self.generic_visit(node)
        
        visitor = ResourceLeakVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _check_unused_variables(self, tree: ast.AST) -> List[SemanticIssue]:
        """Check for unused variables."""
        issues = []
        
        # First, collect all variable definitions
        defined_vars = {}
        used_vars = set()
        
        class VariableCollector(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    defined_vars[node.id] = node.lineno
                elif isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                # Don't count function parameters as unused
                for arg in node.args.args:
                    used_vars.add(arg.arg)
                self.generic_visit(node)
        
        visitor = VariableCollector()
        visitor.visit(tree)
        
        # Check for variables that are defined but never used
        for var, lineno in defined_vars.items():
            if var not in used_vars and not var.startswith('_'):
                issues.append(SemanticIssue(
                    "unused_variable", 
                    0.2,  # Low severity, just a warning
                    f"Variable '{var}' is defined but never used",
                    (lineno, 0)
                ))
                
        return issues


class SMTSolver:
    """
    Interface to SMT solver for checking formal properties.
    """
    
    def __init__(self):
        """Initialize the SMT solver interface."""
        pass
    
    def check_property(self, code: str, property_spec: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a property holds for the given code.
        
        Args:
            code: Python code to analyze
            property_spec: The property specification to check
            
        Returns:
            Tuple of (property_holds, counterexample)
        """
        # Parse the code and property to build SMT formulas
        # This is a simplified implementation that would be expanded
        # with a real translation from Python to SMT formulas
        
        # Example properties we can check:
        # - Bounds on integer variables
        # - Ranges of values
        # - Simple arithmetic relationships
        
        # For demonstration, we'll implement a very simple check
        if "range_check" in property_spec:
            var_name = property_spec.split("(")[1].split(")")[0]
            min_val, max_val = map(int, property_spec.split("[")[1].split("]")[0].split(","))
            
            # Create a Z3 solver
            solver = Solver()
            
            # Define the variable
            var = Int(var_name)
            
            # Add constraints from the code
            # This would normally involve parsing the code and translating
            # to Z3 constraints. For simplicity, we'll just check the range.
            solver.add(Or(var < min_val, var > max_val))
            
            # Check if the property can be violated
            if solver.check() == sympy.sat:
                model = solver.model()
                counterexample = f"{var_name} = {model[var]}"
                return False, counterexample
            else:
                return True, None
                
        # Default: consider property unverified
        return False, "Property not supported"
    
    def translate_code_to_smt(self, code: str) -> Any:
        """
        Translate Python code to SMT formulas.
        
        Args:
            code: Python code to translate
            
        Returns:
            SMT representation of the code
        """
        # This would be a complex function that translates Python operations
        # to equivalent SMT formulas. For the prototype, we'll implement
        # a very simplified version that handles basic arithmetic.
        
        try:
            tree = ast.parse(code)
            
            # In a real implementation, this would walk the AST and
            # build appropriate SMT formulas for each operation
            
            return "SMT formula"  # Placeholder
            
        except SyntaxError:
            return "Invalid code"


class SemanticSteeringModule:
    """
    Main Semantic Steering Module that evaluates code snippets for
    semantic issues and computes penalties.
    """
    
    def __init__(self, language: str = "python", use_smt: bool = True):
        """
        Initialize the SeSM for a specific language.
        
        Args:
            language: Target programming language
            use_smt: Whether to use SMT solver for formal checks
        """
        self.language = language.lower()
        self.use_smt = use_smt
        self.static_analyzer = None
        self.smt_solver = None
        
        # Initialize appropriate analyzers for the language
        if self.language == "python":
            self.static_analyzer = PythonStaticAnalyzer()
            if use_smt:
                self.smt_solver = SMTSolver()
        else:
            raise ValueError(f"Unsupported language for semantic analysis: {language}")
    
    def analyze_code_snippet(self, code: str, formal_specs: Optional[List[str]] = None) -> List[SemanticIssue]:
        """
        Analyze a code snippet for semantic issues.
        
        Args:
            code: Code snippet to analyze
            formal_specs: Optional list of formal specifications to check
            
        Returns:
            List of SemanticIssue objects
        """
        issues = []
        
        # Run static analysis
        if self.static_analyzer:
            issues.extend(self.static_analyzer.analyze(code))
            
        # Check formal specifications using SMT solver
        if self.smt_solver and formal_specs:
            for spec in formal_specs:
                holds, counterexample = self.smt_solver.check_property(code, spec)
                if not holds:
                    issues.append(SemanticIssue(
                        "specification_violation",
                        0.9,  # High severity for spec violations
                        f"Violation of specification '{spec}'. Counterexample: {counterexample}",
                        None  # Location not determined for SMT checks
                    ))
                    
        return issues
    
    def calculate_penalty(self, issues: List[SemanticIssue]) -> float:
        """
        Calculate a penalty score based on detected semantic issues.
        
        Args:
            issues: List of semantic issues
            
        Returns:
            Penalty score between 0.0 and 1.0 (higher = more severe issues)
        """
        if not issues:
            return 0.0
            
        # Combine issue severities
        # We'll use a weighted average, giving more weight to severe issues
        total_weight = 0.0
        weighted_sum = 0.0
        
        for issue in issues:
            # Square the severity to give more weight to severe issues
            weight = issue.severity ** 2
            weighted_sum += weight * issue.severity
            total_weight += weight
            
        # Normalize to [0, 1] range
        if total_weight == 0:
            return 0.0
            
        penalty = weighted_sum / total_weight
        
        # Apply a scaling function to make moderate issues have more impact
        # This sigmoid-like function keeps the range [0, 1] but makes
        # mid-range issues more significant
        scaled_penalty = 0.5 * (1 + (2 * penalty - 1) / (1 + abs(2 * penalty - 1)))
        
        return scaled_penalty
    
    def get_feedback(self, issues: List[SemanticIssue]) -> str:
        """
        Generate natural language feedback based on detected issues.
        
        Args:
            issues: List of semantic issues
            
        Returns:
            Feedback string explaining the issues
        """
        if not issues:
            return "No semantic issues detected."
            
        # Group issues by type for more coherent feedback
        issues_by_type = {}
        for issue in issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)
            
        # Generate feedback text
        feedback = ["Semantic issues detected:"]
        
        for issue_type, issue_list in issues_by_type.items():
            # Convert to readable name
            readable_type = issue_type.replace("_", " ").title()
            feedback.append(f"\n{readable_type}:")
            
            # Add up to 3 issues of this type
            for i, issue in enumerate(issue_list[:3]):
                feedback.append(f"  - {issue.message}")
                
            # If there are more issues, summarize
            if len(issue_list) > 3:
                feedback.append(f"  - ...and {len(issue_list) - 3} more {readable_type.lower()} issues.")
                
        return "\n".join(feedback)
        
    def steer_token_probabilities(self, code: str, next_token_probs: Dict[str, float], 
                                 formal_specs: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Adjust token probabilities based on semantic analysis of potential continuations.
        
        Args:
            code: Current code snippet
            next_token_probs: Dictionary mapping tokens to their probabilities
            formal_specs: Optional list of formal specifications to check
            
        Returns:
            Adjusted token probabilities
        """
        # For each high-probability token, simulate adding it to the code
        # and check for semantic issues
        adjusted_probs = {}
        
        # Only check the top 10 tokens for efficiency
        sorted_tokens = sorted(next_token_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for token, prob in sorted_tokens:
            # Add the token to the code
            extended_code = code + token
            
            # Check for semantic issues
            issues = self.analyze_code_snippet(extended_code, formal_specs)
            
            # Calculate penalty based on issues
            penalty = self.calculate_penalty(issues)
            
            # Apply penalty to token probability
            adjusted_prob = prob * (1.0 - penalty)
            adjusted_probs[token] = adjusted_prob
            
        # Add remaining tokens with their original probabilities
        for token, prob in next_token_probs.items():
            if token not in adjusted_probs:
                adjusted_probs[token] = prob
                
        # Renormalize probabilities
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            normalized_probs = {token: prob / total_prob for token, prob in adjusted_probs.items()}
            return normalized_probs
            
        # If all tokens got zero probability, return original probabilities
        return next_token_probs