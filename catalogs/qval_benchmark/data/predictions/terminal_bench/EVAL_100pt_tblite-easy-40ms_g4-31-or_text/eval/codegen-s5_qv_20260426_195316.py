import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a TerminalBench environment.
    The value reflects the expected discounted reward (binary 0/1) based on whether the action
    appears to progress the agent toward completing the system administration/coding task.
    """
    # Baseline value for a neutral action
    q_val = 0.3

    # 1. Check for explicit failure signals in the resulting state
    error_patterns = [
        r"command not found",
        r"No such file or directory",
        r"Permission denied",
        r"SyntaxError",
        r"Traceback",
        r"Invalid argument",
        r"failed to",
        r"error:.*",
        r"cannot open"
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_val -= 0.2
            break

    # 2. Check for explicit success signals in the resulting state
    success_patterns = [
        r"success",
        r"completed",
        r"done",
        r"passed",
        r"correct",
        r"verification successful",
        r"100%",
        r"All tests passed"
    ]
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_val += 0.4
            break

    # 3. Categorize the action to weigh its intent
    # Submission/Verification: High impact
    submission_keywords = ["submit", "verify", "pytest", "test_script", "check_answer"]
    # Modification/Creation: High progress potential
    modification_keywords = ["vim", "nano", "sed", "awk", "echo", "python", "gcc", "pip", "openssl", "mkdir", "cp", "mv"]
    # Exploration: Low to medium progress (essential but not final)
    exploration_keywords = ["ls", "pwd", "cat", "grep", "find", "head", "tail", "df", "du"]

    if any(kw in action for kw in submission_keywords):
        # If it's a submission, the outcome in next_state is the primary driver
        if any(re.search(p, next_state, re.IGNORECASE) for p in success_patterns):
            q_val += 0.5
        else:
            q_val -= 0.1
    elif any(kw in action for kw in modification_keywords):
        # Modification is good if it didn't crash
        if not any(re.search(p, next_state, re.IGNORECASE) for p in error_patterns):
            q_val += 0.2
    elif any(kw in action for kw in exploration_keywords):
        # Exploration is moderately useful
        if not any(re.search(p, next_state, re.IGNORECASE) for p in error_patterns):
            q_val += 0.1

    # 4. Penalize likely redundant or useless actions
    # For example, running 'ls' repeatedly without change or just 'clear'
    if action.strip() == "clear" or action.strip() == "ls":
        q_val -= 0.05

    # Ensure the return value is clamped between 0.0 and 1.0
    return max(0.0, min(1.0, q_val))