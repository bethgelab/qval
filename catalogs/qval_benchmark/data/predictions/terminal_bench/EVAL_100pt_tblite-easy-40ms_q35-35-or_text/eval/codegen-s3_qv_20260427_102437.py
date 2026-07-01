import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Success patterns indicating task completion
    success_keywords = [
        r'\b(success|passed|completed|verified|done|ok|congrats|100%)\b',
        r'\bverification\s*passed\b',
        r'\btask\s*completed\b',
        r'\ball\s*tests\s*passed\b'
    ]
    # Error patterns indicating failure
    error_keywords = [
        r'\b(error|fail|exception|traceback|denied|not\s*found|broken|crash)\b',
        r'\bpermission\s*denied\b',
        r'\btraceback\b'
    ]
    # High-value actions
    good_actions = [
        r'\bsubmit\b',
        r'\bverify\b',
        r'\brun\s*test\b',
        r'\bcheck\b',
        r'\bdeploy\b',
        r'\bcommit\b'
    ]
    # Low-value actions
    bad_actions = [
        r'\bexit\b',
        r'\bclear\b',
        r'\bq\b'
    ]

    # Check for success in next_state
    for pattern in success_keywords:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0

    # Check for errors in next_state
    for pattern in error_keywords:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0

    # Check if state changed (progress indicator)
    if state == next_state:
        return 0.1

    # Initialize score based on action quality
    score = 0.5
    for pattern in good_actions:
        if re.search(pattern, action, re.IGNORECASE):
            score = max(score, 0.8)

    for pattern in bad_actions:
        if re.search(pattern, action, re.IGNORECASE):
            score = min(score, 0.3)

    # Reward for output generation (progress)
    if len(next_state) > len(state):
        score = max(score, 0.6)

    return score