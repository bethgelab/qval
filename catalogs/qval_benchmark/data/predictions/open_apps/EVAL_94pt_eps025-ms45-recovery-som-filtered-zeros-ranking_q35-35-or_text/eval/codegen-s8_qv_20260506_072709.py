import re

def signal_function(state: str, action: str, next_state: str) -> float:
    success_keywords = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bcreated\b', r'\badded\b',
        r'\bsent\b', r'\bsaved\b', r'\bdone\b', r'\btask completed\b',
        r'\bevent created\b', r'\bmessage sent\b'
    ]
    error_keywords = [
        r'\berror\b', r'\bfailed\b', r'\binvalid\b'
    ]

    success_pattern = '|'.join(success_keywords)
    error_pattern = '|'.join(error_keywords)

    is_success = re.search(success_pattern, next_state, re.IGNORECASE) is not None
    is_state_success = re.search(success_pattern, state, re.IGNORECASE) is not None
    
    if is_success or is_state_success:
        return 1.0

    is_error = re.search(error_pattern, next_state, re.IGNORECASE) is not None
    if is_error:
        return 0.0

    if 'noop' in action:
        return 0.0

    return 0.0