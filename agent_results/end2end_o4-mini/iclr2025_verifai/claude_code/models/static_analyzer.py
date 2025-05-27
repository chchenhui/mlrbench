"""
Static analyzer and Z3 solver integration for ContractGPT.

This module provides functionality to extract verification conditions from
generated code and check them using the Z3 SMT solver.
"""

import re
import ast
import z3
from typing import Dict, List, Tuple, Optional, Union, Any, Set

from models.dsl_parser import Contract


class VerificationCondition:
    """Represents a verification condition to be checked by the SMT solver."""
    
    def __init__(self, name: str, formula: z3.BoolRef, location: str = ""):
        """
        Initialize a verification condition.
        
        Args:
            name: Name/identifier of the verification condition.
            formula: Z3 formula representing the condition.
            location: Source location for reference.
        """
        self.name = name
        self.formula = formula
        self.location = location
    
    def __str__(self) -> str:
        """String representation of the verification condition."""
        return f"{self.name}: {self.formula}"


class CounterExample:
    """Represents a counterexample to a verification condition."""
    
    def __init__(self, vc_name: str, model: z3.ModelRef, input_values: Dict[str, Any]):
        """
        Initialize a counterexample.
        
        Args:
            vc_name: Name of the verification condition that failed.
            model: Z3 model representing the counterexample.
            input_values: Dictionary of variable assignments in the counterexample.
        """
        self.vc_name = vc_name
        self.model = model
        self.input_values = input_values
    
    def __str__(self) -> str:
        """String representation of the counterexample."""
        values = ", ".join([f"{k}={v}" for k, v in self.input_values.items()])
        return f"Counterexample for {self.vc_name}: {values}"


class StaticAnalyzer:
    """
    Static analyzer for verifying code against specifications.
    
    This class extracts verification conditions from code and checks them
    using the Z3 SMT solver.
    """
    
    def __init__(self):
        """Initialize the static analyzer."""
        # Set up Z3 solver
        self.solver = z3.Solver()
        # Maps from variable names to Z3 variables
        self.var_map = {}
    
    def extract_verification_conditions(self, code: str, contract: Contract) -> List[VerificationCondition]:
        """
        Extract verification conditions from code based on a contract.
        
        Args:
            code: The generated code.
            contract: The contract specification.
            
        Returns:
            List of verification conditions.
        """
        # This is a simplified implementation
        # In a complete system, we would parse the AST and extract conditions
        vcs = []
        
        # Extract assertions from the code
        assertions = self._extract_assertions(code)
        
        # Convert preconditions to Z3 formulas
        preconditions = []
        for pre in contract.preconditions:
            z3_formula = self._condition_to_z3(pre)
            if z3_formula is not None:
                preconditions.append(z3_formula)
        
        # Convert postconditions to Z3 formulas
        postconditions = []
        for post in contract.postconditions:
            z3_formula = self._condition_to_z3(post)
            if z3_formula is not None:
                postconditions.append(z3_formula)
        
        # Create verification condition: pre => post
        if preconditions and postconditions:
            pre_formula = z3.And(*preconditions) if len(preconditions) > 1 else preconditions[0]
            post_formula = z3.And(*postconditions) if len(postconditions) > 1 else postconditions[0]
            
            main_vc = VerificationCondition(
                "contract_vc",
                z3.Implies(pre_formula, post_formula)
            )
            vcs.append(main_vc)
        
        # Add VCs for assertions
        for i, assertion in enumerate(assertions):
            z3_formula = self._condition_to_z3(assertion)
            if z3_formula is not None:
                vc = VerificationCondition(
                    f"assertion_{i}",
                    z3_formula
                )
                vcs.append(vc)
        
        return vcs
    
    def _extract_assertions(self, code: str) -> List[str]:
        """
        Extract assertions from code.
        
        Args:
            code: The code to analyze.
            
        Returns:
            List of assertions as strings.
        """
        # Simple regex-based extraction for now
        # In a real implementation, we would use AST parsing
        assertion_pattern = r"assert\s+(.+?)(?:,|\s*$)"
        matches = re.findall(assertion_pattern, code)
        return [m.strip() for m in matches]
    
    def _condition_to_z3(self, condition: str) -> Optional[z3.BoolRef]:
        """
        Convert a condition string to a Z3 formula.
        
        Args:
            condition: Condition string.
            
        Returns:
            Z3 formula or None if conversion fails.
        """
        # This is a simplified implementation
        # In a real system, we would parse the expression properly
        try:
            # Handle some common patterns
            # Replace common operators
            condition = condition.replace("==", " == ")
            condition = condition.replace("!=", " != ")
            condition = condition.replace("<=", " <= ")
            condition = condition.replace(">=", " >= ")
            condition = condition.replace("<", " < ")
            condition = condition.replace(">", " > ")
            
            # Very basic handling for some simple conditions
            if "==" in condition:
                left, right = condition.split("==")
                return self._get_z3_var(left.strip()) == self._get_z3_var(right.strip())
            elif "!=" in condition:
                left, right = condition.split("!=")
                return self._get_z3_var(left.strip()) != self._get_z3_var(right.strip())
            elif "<=" in condition:
                left, right = condition.split("<=")
                return self._get_z3_var(left.strip()) <= self._get_z3_var(right.strip())
            elif ">=" in condition:
                left, right = condition.split(">=")
                return self._get_z3_var(left.strip()) >= self._get_z3_var(right.strip())
            elif "<" in condition:
                left, right = condition.split("<")
                return self._get_z3_var(left.strip()) < self._get_z3_var(right.strip())
            elif ">" in condition:
                left, right = condition.split(">")
                return self._get_z3_var(left.strip()) > self._get_z3_var(right.strip())
            
            # Fall back to treating as a boolean variable
            return self._get_z3_var(condition)
            
        except Exception as e:
            print(f"Failed to convert condition to Z3: {condition}, Error: {e}")
            return None
    
    def _get_z3_var(self, var_name: str) -> Union[z3.ArithRef, z3.BoolRef]:
        """
        Get a Z3 variable for a given name, creating it if it doesn't exist.
        
        Args:
            var_name: Name of the variable.
            
        Returns:
            Z3 variable.
        """
        # Try to parse as a number
        try:
            return z3.IntVal(int(var_name))
        except ValueError:
            try:
                return z3.RealVal(float(var_name))
            except ValueError:
                pass
        
        # Handle as a variable
        if var_name not in self.var_map:
            # Default to Int type
            self.var_map[var_name] = z3.Int(var_name)
        
        return self.var_map[var_name]
    
    def verify(self, vcs: List[VerificationCondition]) -> Tuple[bool, List[CounterExample]]:
        """
        Verify a list of verification conditions.
        
        Args:
            vcs: List of verification conditions to check.
            
        Returns:
            A tuple (success, counterexamples) where success is True if all VCs
            are valid, and counterexamples is a list of counterexamples for
            invalid VCs.
        """
        all_valid = True
        counterexamples = []
        
        for vc in vcs:
            # Reset solver
            self.solver.reset()
            
            # Add negation of VC to find counterexample
            self.solver.add(z3.Not(vc.formula))
            
            # Check satisfiability
            result = self.solver.check()
            
            if result == z3.sat:
                # Found a counterexample
                all_valid = False
                model = self.solver.model()
                
                # Extract input values from model
                input_values = {}
                for var_name, var in self.var_map.items():
                    if model[var] is not None:
                        input_values[var_name] = model[var]
                
                counterexample = CounterExample(vc.name, model, input_values)
                counterexamples.append(counterexample)
        
        return all_valid, counterexamples


def verify_code(code: str, contract: Contract) -> Tuple[bool, List[CounterExample]]:
    """
    Verify generated code against a contract.
    
    Args:
        code: The generated code.
        contract: The contract specification.
        
    Returns:
        A tuple (success, counterexamples) where success is True if the code
        meets the contract, and counterexamples is a list of counterexamples
        for invalid verification conditions.
    """
    analyzer = StaticAnalyzer()
    vcs = analyzer.extract_verification_conditions(code, contract)
    return analyzer.verify(vcs)