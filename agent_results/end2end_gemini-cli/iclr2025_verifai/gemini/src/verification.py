import subprocess
import tempfile
import os
import re

def run_unit_tests(code, test_code):
    """
    Runs unit tests against the given code.
    Returns (passed, feedback).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + test_code)
        filepath = f.name

    try:
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "All unit tests passed."
        else:
            # Provide the failing test case as feedback
            feedback = f"The code failed a unit test.\nError:\n{result.stderr}"
            return False, feedback
    except subprocess.TimeoutExpired:
        return False, "The code timed out during unit test execution."
    finally:
        os.remove(filepath)


def run_smt_verification(code, entry_point):
    """
    Runs SMT-based verification using crosshair.
    Returns (passed, feedback).
    """
    # Crosshair works by analyzing a file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # Crosshair needs type hints to be effective.
        # We assume the LLM provides them based on the prompt.
        f.write("import crosshair\n\n")
        f.write(code)
        filepath = f.name

    try:
        # We tell crosshair to check the specific function.
        # We use a short timeout to avoid long verification times.
        command = [
            "crosshair", "check", f"{filepath}:{entry_point}",
            "--analysis_timeout=15"
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20
        )

        output = result.stdout + result.stderr
        
        if "No errors found" in output:
            return True, "SMT verification passed."
        else:
            # This is the core C2P (Counterexample-to-Prompt) logic
            # We parse crosshair's output to create a user-friendly prompt.
            match = re.search(r"falsified a post-condition!\n\nCounterexample: (.*?)\n", output, re.DOTALL)
            if match:
                counterexample_str = match.group(1).strip()
                # Clean up the counterexample string for the prompt
                counterexample_str = re.sub(r'\s+', ' ', counterexample_str)
                feedback = f"The code failed SMT verification. A counterexample was found:\n{counterexample_str}"
                return False, feedback
            else:
                # Fallback for other errors
                return False, f"SMT verification failed with an unknown error.\nOutput:\n{output}"

    except subprocess.TimeoutExpired:
        return False, "The SMT verification process timed out."
    except Exception as e:
        return False, f"An exception occurred during SMT verification: {e}"
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    # Test UT-Repair
    test_code_pass = "def add(a, b): return a + b"
    test_cases_pass = "assert add(1, 2) == 3\nassert add(-1, 1) == 0"
    passed, feedback = run_unit_tests(test_code_pass, test_cases_pass)
    print(f"UT Test (Pass): {passed}, Feedback: {feedback}")

    test_code_fail = "def add(a, b): return a * b" # Incorrect logic
    passed, feedback = run_unit_tests(test_code_fail, test_cases_pass)
    print(f"UT Test (Fail): {passed}, Feedback: {feedback[:100]}...") # Truncate feedback

    # Test SMT-Repair
    # Crosshair works best with contracts in docstrings.
    # The HumanEval prompts often have these implicitly.
    smt_code_pass = '''
def sort_list(arr: list[int]) -> list[int]:
    """
    post: __return__ == sorted(arr)
    """
    return sorted(arr)
'''
    passed, feedback = run_smt_verification(smt_code_pass, "sort_list")
    print(f"SMT Test (Pass): {passed}, Feedback: {feedback}")

    smt_code_fail = '''
def sort_list(arr: list[int]) -> list[int]:
    """
    post: __return__ == sorted(arr)
    """
    arr.reverse() # Incorrect logic
    return arr
'''
    passed, feedback = run_smt_verification(smt_code_fail, "sort_list")
    print(f"SMT Test (Fail): {passed}, Feedback: {feedback}")
