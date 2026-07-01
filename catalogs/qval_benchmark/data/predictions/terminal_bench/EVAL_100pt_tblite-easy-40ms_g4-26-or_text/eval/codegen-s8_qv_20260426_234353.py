import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a terminal task based on the command issued and its output.
    
    The function analyzes the most recent command's output (the difference between the 
    current state and the next state) to detect immediate successes or failures, 
    then uses command-type heuristics to estimate potential progress.
    """
    # Isolate the most recent command's output and the command itself.
    # This prevents old errors or successes from previous steps from affecting the estimate.
    diff_len = len(next_state) - len(state)
    output_segment = next_state[len(state):] if diff_len > 0 else next_state
    
    # 1. Immediate Failure Detection
    # If the output contains critical error signals, the Q-value is 0.0.
    error_patterns = [
        r"error", r"fail", r"not found", r"denied", r"syntaxerror",
        r"traceback", r"command not found", r"no such file",
        r"invalid", r"permission error"
    ]
    error_regex = r"(?i)(" + "|".join(error_patterns) + r")"
    if re.search(error_regex, output_segment):
        return 0.0

    # 2. Immediate Success Detection
    # If the output contains success signals, the Q-value is 1.0.
    # We use word boundaries (\b) for 'ok' to avoid matching words like 'book'.
    success_patterns = [
        r"\bsuccess\b", r"\bpass\b", r"\bcorrect\b", r"\bdone\b",
        r"\bfound\b", r"\bmatch\b", r"\bvalid\b", r"\bok\b",
        r"\bcompleted\b", r"\bfinished\b"
    ]
    success_regex = r"(?i)(" + "|".join(success_patterns) + r")"
    if re.search(success_regex, output_segment):
        return 1.0

    # 3. Command-Type Heuristics
    # If no clear success/failure is detected, we estimate value based on the action type.
    cmd_text = action.strip().lower()
    parts = cmd_text.split()
    if not parts:
        return 0.1
    
    first_cmd = parts[0]

    # Investigative: Commands used for information gathering.
    investigative = {
        'ls', 'cat', 'grep', 'find', 'pwd', 'cd', 'which', 'whereis', 
        'du', 'df', 'ps', 'top', 'head', 'tail', 'more', 'less', 'file', 'stat'
    }
    
    # Modification: Commands that change files or environment settings.
    modification = {
        'mv', 'cp', 'sed', 'echo', 'vi', 'vim', 'nano', 'write', 'touch', 
        'mkdir', 'rm', 'chmod', 'chown', 'apt', 'pip', 'install', 'wget', 'curl'
    }
    
    # Execution: Commands that run scripts, binaries, or interpreters.
    execution = {
        'python', 'bash', 'sh', 'gcc', 'make', 'sudo', 'python3', 'python2', 'perl', 'ruby'
    }

    if first_cmd in investigative:
        # Information gathering is necessary but doesn't directly solve the task.
        return 0.2
    if first_cmd in modification:
        # Modifying the environment is a proactive step towards a solution.
        return 0.4
    if first_cmd in execution or first_cmd.startswith('./'):
        # Running a script or checker is a high-impact action.
        return 0.5
            
    # Default baseline for unknown or low-impact commands.
    return 0.1