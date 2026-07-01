import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The estimate is based on identifying progress markers, error patterns, and command categories.
    """
    # Normalize action for analysis
    clean_action = action.strip()
    if clean_action.startswith("sudo "):
        clean_action = clean_action[5:].strip()

    # 1. Error Detection
    # These patterns typically indicate a failure or a dead end.
    error_indicators = [
        r"command not found",
        r"Permission denied",
        r"No such file or directory",
        r"SyntaxError",
        r"ModuleNotFoundError",
        r"ImportError",
        r"failed with exit code",
        r"Invalid argument",
        r"cannot open",
        r"segmentation fault",
        r"Traceback \(most recent call last\)",
        r"error: ",
        r"fatal error"
    ]
    
    # We isolate the output part of next_state to see if the action caused an error.
    action_pos = next_state.find(action)
    output_part = next_state[action_pos:] if action_pos != -1 else next_state
    
    for pattern in error_indicators:
        if re.search(pattern, output_part, re.IGNORECASE):
            return 0.05 # Very low value for errors

    # 2. Success Detection
    # These patterns indicate that a goal or a sub-goal has been reached.
    success_indicators = [
        r"successfully",
        r"all tests passed",
        r"completed",
        r"correctly",
        r"done",
        r"exit code 0",
        r"Verification successful",
        r"passed",
        r"Correct"
    ]
    
    is_success = False
    for pattern in success_indicators:
        if re.search(pattern, output_part, re.IGNORECASE):
            is_success = True
            break

    # 3. Action Categorization
    finalization_cmds = ["submit", "pytest", "verify", "test", "make test", "python test"]
    productive_cmds = ["vim", "vi", "nano", "sed", "awk", "echo", "printf", "chmod", "chown", "pip", "apt", "git", "mkdir", "rm", "cp", "mv", "touch"]
    exploratory_cmds = ["ls", "pwd", "cat", "grep", "find", "du", "df", "whoami", "uname", "head", "tail", "env", "dir"]

    # High Q-value for finalization commands that lead to success
    if any(clean_action.startswith(cmd) for cmd in finalization_cmds):
        if is_success:
            return 0.98  # Highly likely the episode is solved
        return 0.4       # If not success but no error, might still be progress

    # High Q-value for any action leading to a clear success marker
    if is_success:
        return 0.85

    # Medium Q-value for productive changes that didn't error
    if any(clean_action.startswith(cmd) for cmd in productive_cmds):
        return 0.5

    # Lower Q-value for purely exploratory actions
    if any(clean_action.startswith(cmd) for cmd in exploratory_cmds):
        return 0.3

    # 4. Repetition Penalty
    # Check if the agent is repeating the same action multiple times, suggesting it is stuck.
    if state.count(action) > 3:
        return 0.1

    # Default base value for ambiguous but non-erroring transitions
    return 0.2