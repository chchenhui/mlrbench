
import sys
import json
from io import StringIO

def trace_execution(code_string, test_case_input):
    """
    Executes a code string with a given input and captures its execution trace.
    """
    trace = []
    
    def tracer(frame, event, arg):
        if event == 'line':
            trace.append({
                'line_number': frame.f_lineno,
                'event_type': event,
                'variable_states': {k: repr(v) for k, v in frame.f_locals.items()}
            })
        return tracer

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Prepare the execution environment
        local_vars = {}
        
        # The code string is expected to be a function definition.
        # We execute it to define the function in our environment.
        exec(code_string, local_vars)
        
        # The test case input is expected to be a string that calls the function.
        # For example, "my_function(1, 2)"
        
        # Set the tracer
        sys.settrace(tracer)
        
        # Execute the test case
        result = eval(test_case_input, local_vars)
        
        # Stop the tracer
        sys.settrace(None)
        
        return {
            "trace": trace,
            "output": redirected_output.getvalue(),
            "result": result,
            "status": "PASS"
        }
    except Exception as e:
        sys.settrace(None)
        trace.append({
            'event_type': 'exception',
            'error_info': {
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        })
        return {
            "trace": trace,
            "output": redirected_output.getvalue(),
            "result": None,
            "status": "FAIL"
        }
    finally:
        sys.stdout = old_stdout


if __name__ == '__main__':
    sample_code = """
def add(a, b):
    c = a + b
    return c
"""
    test_input = "add(2, 3)"
    
    execution_result = trace_execution(sample_code, test_input)
    print(json.dumps(execution_result, indent=2))

    sample_code_fail = """
def divide(a, b):
    return a / b
"""
    test_input_fail = "divide(10, 0)"
    execution_result_fail = trace_execution(sample_code_fail, test_input_fail)
    print(json.dumps(execution_result_fail, indent=2))
