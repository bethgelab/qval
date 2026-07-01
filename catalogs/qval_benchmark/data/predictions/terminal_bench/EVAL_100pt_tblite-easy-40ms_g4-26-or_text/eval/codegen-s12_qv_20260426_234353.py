import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The estimation is based on detecting success or failure patterns in the command output.
    """
    # Normalize the next state and action for robust pattern matching
    ns = next_state.lower()
    a = action.lower()
    
    # 1. Detect failure patterns in the command output.
    # Many terminal errors (like 'command not found', 'permission denied', or Python tracebacks)
    # are strong indicators that the current action did not progress toward the goal.
    # We check these first because an error message might contain words like 'success' 
    # (e.g., "Error: Success was not achieved").
    failure_pattern = r'(error|failed|not found|denied|invalid|syntax|traceback|exception|no such|command not found|permission denied)'
    if re.search(failure_pattern, ns):
        return 0.0
        
    # 2. Detect success patterns in the command output.
    # If the output explicitly states 'success', 'passed', or 'done', 
    # it is a very strong signal that the task or a verification step has been completed.
    success_pattern = r'(success|passed|completed|done|verified|all tests passed)'
    if re.search(success_pattern, ns):
        return 1.0
        
    # 3. Analyze the nature of the action taken.
    # Test or verification actions (e.g., 'python test.py', 'submit') are crucial.
    # If they don't trigger the explicit error or success patterns above, 
    # we treat the outcome as neutral (uncertain).
    is_test_action = any(word in a for word in ['test', 'verify', 'submit', 'check', 'run'])
    if is_test_action:
        return 0.5
        
    # 4. Default for intermediate successful steps.
    # If no error was detected and it wasn't a terminal test command, the action 
    # likely completed successfully (e.g., a 'mkdir' or 'cd' command).
    # We return a value slightly above 0.5 to represent that the agent is still 
    # in a viable, functioning state within the environment.
    return 0.6