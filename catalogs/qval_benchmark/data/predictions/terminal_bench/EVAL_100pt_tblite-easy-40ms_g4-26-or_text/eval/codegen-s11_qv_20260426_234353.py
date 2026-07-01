import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    Returns a float between 0.0 and 1.0.
    """
    # 1. Error Detection
    # If the command execution resulted in common terminal error messages, the Q-value is low.
    error_patterns = [
        r"command not found",
        r"No such file or directory",
        r"Permission denied",
        r"syntax error",
        r"fatal error",
        r"\berror\b",
        r"\bfailed\b",
        r"\bnot found\b",
        r"\binvalid\b"
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0

    # 2. Success Detection
    # If the output explicitly indicates success (e.g., task completion or verification), the Q-value is high.
    success_patterns = [
        r"\bsuccess\b",
        r"\bpassed\b",
        r"\bdone\b",
        r"\bcompleted\b"
    ]
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0

    # 3. Progress Detection
    # If the state has changed (i.e., next_state is different from state), it indicates the action had an effect.
    if next_state != state:
        # Extract the command name to check if it's an information-gathering command.
        cmd_parts = action.strip().split()
        cmd_name = cmd_parts[0].lower() if cmd_parts else ""
        
        # Commands that retrieve information are often high-value steps in investigation-based tasks.
        info_commands = {
            'ls', 'cat', 'grep', 'find', 'pwd', 'echo', 'type', 
            'which', 'printenv', 'nm', 'strings', 'dir', 'lsb_release'
        }
        
        if cmd_name in info_commands:
            return 0.8
        
        # General progress (e.g., file creation, directory movement, etc.)
        return 0.6
    
    # 4. Stagnation
    # If no text was added and there is no error, the action likely had no effect or resulted in a no-op.
    return 0.1