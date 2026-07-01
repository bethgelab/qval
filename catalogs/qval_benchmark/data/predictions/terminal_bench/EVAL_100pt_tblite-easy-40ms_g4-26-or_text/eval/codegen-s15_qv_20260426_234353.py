import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state-action-next_state transition in a terminal environment.
    The estimation is based on identifying command success, error presence, and command type.
    """
    # Normalize input
    action_clean = action.strip().lower()
    if not action_clean:
        return 0.0

    # Get the most recent part of the output to avoid being misled by old terminal history
    lines = next_state.strip().splitlines()
    recent_output = "\n".join(lines[-15:])

    # 1. Error Detection: If the command produced an error, the Q-value is low.
    # These patterns cover common shell and language-specific error indicators.
    error_patterns = [
        r"command not found",
        r"permission denied",
        r"no such file",
        r"syntaxerror",
        r"traceback",
        r"error:",
        r"failed",
        r"not found",
        r"invalid argument",
        r"exception"
    ]

    for pattern in error_patterns:
        if re.search(pattern, recent_output, re.IGNORECASE):
            return 0.0

    # 2. Success Detection: If the output indicates the task is complete, return max value.
    # TerminalBench tasks often end with specific success strings.
    success_patterns = [
        r"success",
        r"passed",
        r"completed",
        r"done",
        r"verified"
    ]

    for pattern in success_patterns:
        if re.search(pattern, recent_output, re.IGNORECASE):
            return 1.0

    # 3. Command Analysis: Determine if the action was productive or neutral.
    command = action_clean.split()[0]

    # Productive commands are those used to explore, read, or process data.
    productive_commands = {
        "ls", "cat", "grep", "python", "python3", "pip", "pip3", "gcc", 
        "make", "openssl", "curl", "wget", "find", "head", "tail", 
        "diff", "sed", "awk", "git", "docker", "apt", "sudo", "nano", 
        "vim", "nmap", "lsblk", "df", "du", "stat", "chmod", "chown"
    }

    # Management commands are used for structural changes (files/dirs).
    management_commands = {
        "cd", "mkdir", "cp", "mv", "rm", "touch", "ln", "mkdir", "rmdir"
    }

    if command in productive_commands:
        # A successful command that probes the environment or processes data.
        return 0.6
    elif command in management_commands:
        # A successful command that manipulates the file system.
        return 0.4

    # 4. Fallback: If the command is not recognized but didn't fail,
    # we treat it as a neutral step (could be a custom script or unknown tool).
    return 0.3