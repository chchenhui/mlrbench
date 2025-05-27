"""
Verification module for the VERIL framework.

This module implements the Verification Integration Layer (VIL) component
of the VERIL framework, which integrates multiple verification tools and
standardizes their outputs.
"""

import os
import re
import ast
import sys
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import pylint.lint
import pycodestyle
import mypy.api

from config import ERROR_TAXONOMY
from utils import logger, extract_code_from_response


@dataclass
class VerificationError:
    """Class representing a verification error."""
    error_type: str
    location: Optional[Tuple[int, int]] = None
    message: str = ""
    tool: str = ""
    severity: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "location": self.location,
            "message": self.message,
            "tool": self.tool,
            "severity": self.severity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationError':
        """Create from dictionary."""
        return cls(
            error_type=data["error_type"],
            location=tuple(data["location"]) if data.get("location") else None,
            message=data["message"],
            tool=data["tool"],
            severity=data["severity"],
        )


@dataclass
class VerificationResult:
    """Class representing the result of verification."""
    code: str
    passed: bool = False
    errors: List[VerificationError] = field(default_factory=list)
    execution_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "passed": self.passed,
            "errors": [error.to_dict() for error in self.errors],
            "execution_output": self.execution_output,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create from dictionary."""
        return cls(
            code=data["code"],
            passed=data["passed"],
            errors=[VerificationError.from_dict(error) for error in data["errors"]],
            execution_output=data.get("execution_output"),
        )


class StaticAnalyzer:
    """Static analysis tools for Python code."""
    
    @staticmethod
    def _get_error_type(error_message: str) -> str:
        """Map error message to error type."""
        error_message = error_message.lower()
        
        # Define mapping of error patterns to error types
        error_patterns = {
            "syntax": ["syntaxerror", "invalid syntax", "unexpected", "syntax error"],
            "type": ["typeerror", "type error", "incompatible type", "expected type"],
            "logic": ["logical error", "algorithm error", "incorrect implementation"],
            "semantic": ["nameError", "attributeError", "importError", "valueerror", "assertion"],
            "security": ["security", "vulnerability", "insecure", "injection"],
        }
        
        # Find matching error type
        for error_type, patterns in error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        # Default to semantic error
        return "semantic"
    
    @staticmethod
    def _get_error_severity(error_type: str) -> float:
        """Get severity score for error type."""
        return ERROR_TAXONOMY.get(error_type, {}).get("severity", 0.5)
    
    @staticmethod
    def check_syntax(code: str) -> List[VerificationError]:
        """
        Check code for syntax errors.
        
        Args:
            code: Python code to check
            
        Returns:
            List of verification errors
        """
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Extract error information
            line_num = e.lineno if hasattr(e, "lineno") else 0
            col_num = e.offset if hasattr(e, "offset") else 0
            message = str(e)
            
            # Create verification error
            error_type = StaticAnalyzer._get_error_type(message)
            severity = StaticAnalyzer._get_error_severity(error_type)
            
            error = VerificationError(
                error_type=error_type,
                location=(line_num, col_num),
                message=message,
                tool="syntax_checker",
                severity=severity,
            )
            errors.append(error)
        
        return errors
    
    @staticmethod
    def check_style(code: str) -> List[VerificationError]:
        """
        Check code for style issues using pycodestyle.
        
        Args:
            code: Python code to check
            
        Returns:
            List of verification errors
        """
        errors = []
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Create a custom reporter to capture errors
            class CustomReport:
                def __init__(self):
                    self.errors = []
                
                def error(self, line_number, offset, text, check):
                    self.errors.append((line_number, offset, text, check))
            
            style_report = CustomReport()
            
            # Run pycodestyle
            style = pycodestyle.StyleGuide(quiet=True, reporter=pycodestyle.StandardReport)
            style.check_files([temp_file_path])
            
            # Process style errors
            for line_num, offset, message, check in style_report.errors:
                error_type = "style"
                severity = StaticAnalyzer._get_error_severity(error_type)
                
                error = VerificationError(
                    error_type=error_type,
                    location=(line_num, offset),
                    message=message,
                    tool="pycodestyle",
                    severity=severity,
                )
                errors.append(error)
        
        except Exception as e:
            logger.warning(f"Style check failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return errors
    
    @staticmethod
    def check_types(code: str) -> List[VerificationError]:
        """
        Check code for type errors using mypy.
        
        Args:
            code: Python code to check
            
        Returns:
            List of verification errors
        """
        errors = []
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Run mypy
            mypy_result = mypy.api.run([temp_file_path, "--ignore-missing-imports"])
            error_messages = mypy_result[0].split("\n")
            
            # Process error messages
            for message in error_messages:
                if not message or ":" not in message:
                    continue
                
                # Parse error message format: "file:line: error: message"
                try:
                    parts = message.split(":", 3)
                    if len(parts) < 4:
                        continue
                    
                    line_num = int(parts[1])
                    error_msg = parts[3].strip()
                    
                    error_type = StaticAnalyzer._get_error_type(error_msg)
                    severity = StaticAnalyzer._get_error_severity(error_type)
                    
                    error = VerificationError(
                        error_type="type",
                        location=(line_num, 0),
                        message=error_msg,
                        tool="mypy",
                        severity=severity,
                    )
                    errors.append(error)
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse mypy error: {message}")
        
        except Exception as e:
            logger.warning(f"Type check failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return errors
    
    @staticmethod
    def run_linting(code: str) -> List[VerificationError]:
        """
        Run pylint on the code.
        
        Args:
            code: Python code to check
            
        Returns:
            List of verification errors
        """
        errors = []
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Create a custom reporter to capture errors
            from pylint.reporters.text import TextReporter
            
            class CustomReporter(TextReporter):
                def __init__(self):
                    self.errors = []
                    super().__init__()
                
                def handle_message(self, msg):
                    self.errors.append(msg)
                    super().handle_message(msg)
            
            reporter = CustomReporter()
            
            # Run pylint with custom reporter
            pylint_args = [
                "--disable=all", 
                "--enable=E,F,unreachable,duplicate-key,unnecessary-semicolon,global-variable-not-assigned,unused-variable,unused-argument,unused-import,assignment-from-no-return,function-redefined,undefined-variable",
                temp_file_path
            ]
            pylint.lint.Run(pylint_args, reporter=reporter, exit=False)
            
            # Process lint errors
            for msg in reporter.errors:
                line_num = msg.line
                col_num = msg.column
                message = msg.msg
                
                error_type = "semantic"
                if msg.msg_id.startswith("E"):
                    error_type = "syntax"
                elif msg.msg_id.startswith("F"):
                    error_type = "semantic"
                
                severity = StaticAnalyzer._get_error_severity(error_type)
                
                error = VerificationError(
                    error_type=error_type,
                    location=(line_num, col_num),
                    message=message,
                    tool="pylint",
                    severity=severity,
                )
                errors.append(error)
        
        except Exception as e:
            logger.warning(f"Linting failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return errors
    
    @staticmethod
    def analyze(code: str) -> List[VerificationError]:
        """
        Run all static analysis checks on the code.
        
        Args:
            code: Python code to check
            
        Returns:
            List of verification errors
        """
        errors = []
        
        # Check syntax (this is quick and required for other checks)
        syntax_errors = StaticAnalyzer.check_syntax(code)
        errors.extend(syntax_errors)
        
        # Skip other checks if there are syntax errors
        if not syntax_errors:
            # Check types
            type_errors = StaticAnalyzer.check_types(code)
            errors.extend(type_errors)
            
            # Run linting
            lint_errors = StaticAnalyzer.run_linting(code)
            errors.extend(lint_errors)
            
            # Check style
            style_errors = StaticAnalyzer.check_style(code)
            errors.extend(style_errors)
        
        return errors


class DynamicAnalyzer:
    """Dynamic analysis tools for Python code."""
    
    @staticmethod
    def execute_tests(code: str, test_cases: List[str]) -> Tuple[bool, List[VerificationError], str]:
        """
        Execute test cases for the code.
        
        Args:
            code: Python code to test
            test_cases: List of test case code
            
        Returns:
            Tuple of (passed, errors, output)
        """
        errors = []
        all_passed = True
        full_output = ""
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Process each test case
            for i, test_code in enumerate(test_cases):
                test_num = i + 1
                
                # Create a combined test file
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as test_file:
                    # Write the original code and the test code
                    test_file.write(code + "\n\n")
                    test_file.write(f"# Test case {test_num}\n")
                    test_file.write("try:\n")
                    # Indent the test code
                    indented_test = "\n".join(f"    {line}" for line in test_code.split("\n"))
                    test_file.write(indented_test)
                    test_file.write("\n    print('Test case passed!')\n")
                    test_file.write("except Exception as e:\n")
                    test_file.write("    print(f'Test failed: {str(e)}')\n")
                    test_file.write("    import traceback\n")
                    test_file.write("    traceback.print_exc()\n")
                    test_file_path = test_file.name
                
                try:
                    # Execute the test
                    process = subprocess.run(
                        [sys.executable, test_file_path],
                        capture_output=True,
                        text=True,
                        timeout=10  # Timeout after 10 seconds
                    )
                    
                    output = process.stdout + process.stderr
                    full_output += f"\n--- Test Case {test_num} ---\n{output}\n"
                    
                    # Check if test passed
                    test_passed = "Test case passed!" in output
                    if not test_passed:
                        all_passed = False
                        
                        # Extract error message
                        error_match = re.search(r"Test failed: (.+)", output)
                        error_message = error_match.group(1) if error_match else "Test failed"
                        
                        # Try to extract line number from traceback
                        line_match = re.search(r"line (\d+)", output)
                        line_num = int(line_match.group(1)) if line_match else 0
                        
                        # Create verification error
                        error_type = StaticAnalyzer._get_error_type(error_message)
                        severity = StaticAnalyzer._get_error_severity(error_type)
                        
                        error = VerificationError(
                            error_type=error_type,
                            location=(line_num, 0),
                            message=f"Test case {test_num} failed: {error_message}",
                            tool="test_execution",
                            severity=severity,
                        )
                        errors.append(error)
                
                except subprocess.TimeoutExpired:
                    all_passed = False
                    full_output += f"\n--- Test Case {test_num} ---\nTEST TIMEOUT: Test took too long to execute\n"
                    
                    error = VerificationError(
                        error_type="logic",
                        location=None,
                        message=f"Test case {test_num} timed out after 10 seconds",
                        tool="test_execution",
                        severity=0.8,
                    )
                    errors.append(error)
                
                except Exception as e:
                    all_passed = False
                    full_output += f"\n--- Test Case {test_num} ---\nERROR: {str(e)}\n"
                    
                    error = VerificationError(
                        error_type="semantic",
                        location=None,
                        message=f"Test case {test_num} execution error: {str(e)}",
                        tool="test_execution",
                        severity=0.7,
                    )
                    errors.append(error)
                
                finally:
                    # Clean up test file
                    if os.path.exists(test_file_path):
                        os.remove(test_file_path)
        
        finally:
            # Clean up code file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return all_passed, errors, full_output
    
    @staticmethod
    def analyze(code: str, test_cases: List[str]) -> Tuple[List[VerificationError], str]:
        """
        Run dynamic analysis on the code.
        
        Args:
            code: Python code to check
            test_cases: List of test case code
            
        Returns:
            Tuple of (errors, output)
        """
        all_passed, errors, output = DynamicAnalyzer.execute_tests(code, test_cases)
        return errors, output


class VerificationIntegrationLayer:
    """
    Verification Integration Layer (VIL) component of the VERIL framework.
    
    This component orchestrates the verification process by integrating multiple
    verification tools and standardizing their outputs.
    """
    
    def __init__(self, verification_types: List[str] = ["static", "dynamic"]):
        """
        Initialize the verification layer.
        
        Args:
            verification_types: List of verification types to use
                ("static", "dynamic", "formal")
        """
        self.verification_types = verification_types
        logger.info(f"Initialized Verification Integration Layer with types: {verification_types}")
    
    def verify(self, code: str, test_cases: List[str]) -> VerificationResult:
        """
        Verify the code using all configured verification tools.
        
        Args:
            code: Python code to verify
            test_cases: List of test case code
            
        Returns:
            VerificationResult object
        """
        # Initialize verification result
        result = VerificationResult(code)
        all_errors = []
        execution_output = ""
        
        # Clean up the code if it's from a model response
        clean_code = extract_code_from_response(code)
        
        # Run static analysis
        if "static" in self.verification_types:
            logger.info("Running static analysis...")
            try:
                static_errors = StaticAnalyzer.analyze(clean_code)
                all_errors.extend(static_errors)
                logger.info(f"Static analysis found {len(static_errors)} errors")
            except Exception as e:
                logger.error(f"Static analysis failed: {str(e)}")
                error = VerificationError(
                    error_type="semantic",
                    location=None,
                    message=f"Static analysis error: {str(e)}",
                    tool="static_analyzer",
                    severity=0.5,
                )
                all_errors.append(error)
        
        # Run dynamic analysis
        if "dynamic" in self.verification_types:
            logger.info("Running dynamic analysis...")
            try:
                # Skip dynamic analysis if there are syntax errors
                has_syntax_errors = any(error.error_type == "syntax" for error in all_errors)
                if has_syntax_errors:
                    logger.info("Skipping dynamic analysis due to syntax errors")
                else:
                    dynamic_errors, output = DynamicAnalyzer.analyze(clean_code, test_cases)
                    all_errors.extend(dynamic_errors)
                    execution_output = output
                    logger.info(f"Dynamic analysis found {len(dynamic_errors)} errors")
            except Exception as e:
                logger.error(f"Dynamic analysis failed: {str(e)}")
                error = VerificationError(
                    error_type="semantic",
                    location=None,
                    message=f"Dynamic analysis error: {str(e)}",
                    tool="dynamic_analyzer",
                    severity=0.5,
                )
                all_errors.append(error)
        
        # Set verification result
        result.errors = all_errors
        result.execution_output = execution_output
        result.passed = len(all_errors) == 0
        
        return result


class ErrorToExplanationConverter:
    """
    Error-to-Explanation Converter (E2EC) component of the VERIL framework.
    
    This component transforms verification outcomes into natural language
    explanations and remediation examples.
    """
    
    @staticmethod
    def _get_code_context(code: str, location: Optional[Tuple[int, int]]) -> str:
        """
        Extract code context around error location.
        
        Args:
            code: Python code
            location: Tuple of (line, column)
            
        Returns:
            Code context as string
        """
        if not location:
            return ""
        
        line_num, _ = location
        lines = code.split("\n")
        
        # Ensure line_num is valid
        if line_num <= 0 or line_num > len(lines):
            return ""
        
        # Get context lines
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 2)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">" if i == line_num - 1 else " "
            context_lines.append(f"{prefix} {i+1}: {lines[i]}")
        
        return "\n".join(context_lines)
    
    @staticmethod
    def _get_error_explanation(error: VerificationError, code: str) -> str:
        """
        Generate explanation for an error.
        
        Args:
            error: VerificationError object
            code: Python code
            
        Returns:
            Explanation as string
        """
        # Get code context
        context = ErrorToExplanationConverter._get_code_context(code, error.location)
        
        # Generate explanation based on error type
        if error.error_type == "syntax":
            explanation = f"There is a syntax error in your code. {error.message}"
            if context:
                explanation += f"\n\nCode context:\n{context}"
            
            explanation += "\n\nSyntax errors occur when the code violates Python's grammar rules. Check for missing parentheses, brackets, colons, or incorrect indentation."
        
        elif error.error_type == "type":
            explanation = f"There is a type error in your code. {error.message}"
            if context:
                explanation += f"\n\nCode context:\n{context}"
            
            explanation += "\n\nType errors occur when operations are performed on incompatible types. Check that variables have the expected types and that you're using appropriate operations for those types."
        
        elif error.error_type == "logic":
            explanation = f"There is a logical error in your code. {error.message}"
            if context:
                explanation += f"\n\nCode context:\n{context}"
            
            explanation += "\n\nLogical errors occur when the code's algorithm is incorrectly implemented. The code may run without errors but produce incorrect results. Check your algorithm implementation carefully."
        
        elif error.error_type == "semantic":
            explanation = f"There is a semantic error in your code. {error.message}"
            if context:
                explanation += f"\n\nCode context:\n{context}"
            
            explanation += "\n\nSemantic errors occur when the code doesn't behave as expected. Check for incorrect variable usage, off-by-one errors, or missing edge case handling."
        
        elif error.error_type == "security":
            explanation = f"There is a security vulnerability in your code. {error.message}"
            if context:
                explanation += f"\n\nCode context:\n{context}"
            
            explanation += "\n\nSecurity vulnerabilities can lead to unintended behavior or exploitation. Check for unsafe input handling, improper access control, or insecure API usage."
        
        else:
            explanation = f"There is an error in your code. {error.message}"
            if context:
                explanation += f"\n\nCode context:\n{context}"
        
        return explanation
    
    @staticmethod
    def generate_explanation(result: VerificationResult) -> Tuple[str, Optional[str]]:
        """
        Generate explanation and remediation for verification result.
        
        Args:
            result: VerificationResult object
            
        Returns:
            Tuple of (explanation, remediation_example)
        """
        if result.passed:
            return "The code passed all verification checks. No errors were found.", None
        
        # Group errors by type
        errors_by_type = {}
        for error in result.errors:
            if error.error_type not in errors_by_type:
                errors_by_type[error.error_type] = []
            errors_by_type[error.error_type].append(error)
        
        # Generate explanation
        explanation = "The code failed verification checks. The following errors were found:\n\n"
        
        for error_type, errors in errors_by_type.items():
            explanation += f"## {error_type.capitalize()} Errors ({len(errors)})\n\n"
            
            for i, error in enumerate(errors[:3]):  # Limit to top 3 errors per type
                error_explanation = ErrorToExplanationConverter._get_error_explanation(error, result.code)
                explanation += f"### Error {i+1}\n{error_explanation}\n\n"
            
            if len(errors) > 3:
                explanation += f"... and {len(errors) - 3} more {error_type} errors\n\n"
        
        # Add execution output if available
        if result.execution_output:
            explanation += "## Test Execution Output\n\n"
            explanation += result.execution_output
        
        # Remediation example would be generated here, but for simplicity, we'll return None
        remediation_example = None
        
        return explanation, remediation_example