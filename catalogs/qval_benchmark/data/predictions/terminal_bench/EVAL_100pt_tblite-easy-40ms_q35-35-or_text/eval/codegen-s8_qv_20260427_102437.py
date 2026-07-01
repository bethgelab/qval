import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Define patterns for success, failure, and progress indicators
    success_patterns = [
        r'\bpass\b', r'\bsuccess\b', r'\bverified\b', r'\bdone\b',
        r'\bcorrect\b', r'\bsolution\b', r'\bpassed\b', r'\baccepted\b',
        r'\b100%\b', r'\bPASSED\b', r'\bSUCCESS\b', r'\bCOMPLETE\b'
    ]
    failure_patterns = [
        r'\bfail\b', r'\berror\b', r'\bexception\b', r'\btraceback\b',
        r'\bdenied\b', r'\bnot found\b', r'\bsyntax\b', r'\binvalid\b',
        r'\bfailed\b', r'\b0%\b', r'\bFAILED\b', r'\bERROR\b', r'\bCRITICAL\b'
    ]
    progress_patterns = [
        r'\bcreated\b', r'\bsaved\b', r'\bwritten\b', r'\brunning\b',
        r'\bexecuted\b', r'\bcompleted\b', r'\binstalled\b'
    ]
    submit_keywords = [
        'submit', 'verify', 'check', 'run', 'execute', 'test', 'solve'
    ]

    # Helper to check if text contains any pattern from a list
    def has_match(text, patterns):
        if not text:
            return False
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    # Check state for terminal conditions (task already done or failed before action)
    if has_match(state, success_patterns):
        return 1.0
    if has_match(state, failure_patterns):
        return 0.0

    # Check if action is empty or whitespace (no progress)
    if not action or not action.strip():
        return 0.0

    # Check next_state for outcomes
    is_success = has_match(next_state, success_patterns)
    is_failure = has_match(next_state, failure_patterns)
    has_progress = has_match(next_state, progress_patterns)

    # Check if action is a submission/verification step
    action_lower = action.lower()
    is_submit_action = any(kw in action_lower for kw in submit_keywords)

    # Calculate Q-value based on signals
    # Priority: Success > Progress > Neutral > Failure
    if is_success:
        if is_submit_action:
            score = 1.0
        else:
            score = 0.95
    elif is_failure:
        score = 0.05
    elif has_progress:
        score = 0.6
    else:
        score = 0.5

    # Ensure score is within valid range [0.0, 1.0]
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return score