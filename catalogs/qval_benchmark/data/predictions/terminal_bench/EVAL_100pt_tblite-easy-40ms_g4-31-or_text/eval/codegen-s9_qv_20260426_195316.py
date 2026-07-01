import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The Q-value represents the expected discounted cumulative reward (probability of success).
    """
    # Base value: uncertainty
    q = 0.2
    
    # Lowercase for easier analysis
    state_l = state.lower()
    next_state_l = next_state.lower()
    action_l = action.lower()

    # 1. Immediate Success/Failure Indicators
    # These are strong signals that the action led to a terminal or highly decisive state.
    success_keywords = ["success", "completed", "passed", "correct", "done", "verified"]
    failure_keywords = ["command not found", "permission denied", "syntax error", "no such file or directory", "failed", "error:"]
    
    # If the action was 'submit' or similar, check for success markers in next_state
    if any(word in action_l for word in ["submit", "verify", "test"]):
        if any(word in next_state_l for word in success_keywords):
            return 1.0
        if any(word in next_state_l for word in failure_keywords):
            return 0.0

    # General failure in the resulting state
    if any(word in next_state_l for word in failure_keywords):
        return 0.05

    # 2. Action Analysis: Productive vs. Exploratory
    # Productive tools for sysadmin, crypto, ML, and data processing
    productive_tools = [
        "openssl", "ssh", "scp", "grep", "sed", "awk", "python", "perl", "gcc", "g++", 
        "make", "chmod", "chown", "find", "curl", "wget", "pip", "apt", "systemctl",
        "netstat", "tcpdump", "iptables", "cat", "less", "tail", "head"
    ]
    
    is_productive = any(tool in action_l for tool in productive_tools)
    is_exploratory = any(tool == action_l.strip() for tool in ["ls", "pwd", "whoami", "uname", "history"])

    if is_productive:
        q += 0.2
    elif is_exploratory:
        q -= 0.1

    # 3. State Transition Analysis: Information Gain
    # Check if the next_state contains meaningful new information compared to the state.
    # We look for lines that are not just the shell prompt.
    def count_meaningful_lines(text):
        lines = text.splitlines()
        # Heuristic: ignore lines that look like prompts (e.g., user@host:~/path$ )
        meaningful = [line for line in lines if not re.search(r'[\w\d.-]+@[\w\d.-]+:.*\$', line)]
        return len(meaningful)

    state_lines = count_meaningful_lines(state)
    next_lines = count_meaningful_lines(next_state)

    if next_lines > state_lines:
        # The action produced output, which is generally a sign of progress (unless it was an error, handled above)
        q += 0.15
    elif next_lines < state_lines:
        # The output was cleared or shrunk, which might be neutral or negative
        q -= 0.05

    # 4. Repetitive Action Penalty
    # If the action is repeated in the state history, it suggests the agent is stuck.
    # We check if the action appears multiple times in the state.
    action_occurrences = state.count(action)
    if action_occurrences >= 2:
        q -= 0.2 * action_occurrences

    # Final clamping to ensure the value is within [0.0, 1.0]
    return max(0.0, min(1.0, q))