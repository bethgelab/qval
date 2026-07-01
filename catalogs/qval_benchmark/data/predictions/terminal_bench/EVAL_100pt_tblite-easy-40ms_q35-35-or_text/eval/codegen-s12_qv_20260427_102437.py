import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Define patterns for success and failure indicators
    success_patterns = [
        r'\bsuccess\b', r'\bverified\b', r'\bspassed\b',
        r'\bcomplete\b', r'\bdone\b', r'\bcorrect\b',
        r'\bflag\b', r'\bsolution\b', r'\bcongratulations\b'
    ]
    error_patterns = [
        r'\berror\b', r'\bfail\b', r'\bexception\b',
        r'\btraceback\b', r'\bdenied\b', r'\bnot found\b',
        r'\binvalid\b', r'\bmissing\b', r'\bpermission\b'
    ]

    # Base Q-value estimate
    score = 0.5

    # Check for success indicators in next_state
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            score += 0.4
            break

    # Check for error indicators in next_state
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            score -= 0.5
            break

    # Penalize empty or whitespace-only actions
    if not action or not action.strip():
        score -= 0.2

    # Penalize if no state change occurred (potential no-op)
    if len(next_state) <= len(state):
        score -= 0.1

    # Clamp score to valid range [0.0, 1.0]
    return max(0.0, min(1.0, score))