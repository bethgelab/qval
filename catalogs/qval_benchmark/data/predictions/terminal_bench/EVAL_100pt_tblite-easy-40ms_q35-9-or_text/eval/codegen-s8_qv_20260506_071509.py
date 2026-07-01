import re
import collections
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.5

    positive_patterns = [
        r'success', r'created', r'made', r'completed', r'done',
        r'wrote', r'copied', r'moved', r'changed', r'updated',
        r'ok', r'0 errors', r'no errors', r'passed', r'exit code 0',
        r'true', r'valid', r'accepted', r'installed', r'built'
    ]

    negative_patterns = [
        r'error', r'failed', r'permission denied', r'not found',
        r'no such file', r'denied', r'invalid', r'exception',
        r'traceback', r'fatal', r'crash', r'broken', r'failed',
        r'rejected', r'invalid', r'abort', r'connection refused',
        r'command not found', r'syntax error', r'too many arguments'
    ]

    positive_count = sum(1 for pattern in positive_patterns if re.search(pattern, next_state, re.IGNORECASE))
    negative_count = sum(1 for pattern in negative_patterns if re.search(pattern, next_state, re.IGNORECASE))

    if positive_count > 0:
        q_value += positive_count * 0.15
    if negative_count > 0:
        q_value -= negative_count * 0.25

    if re.search(r'$', next_state) and re.search(r'$\s*\S', next_state):
        q_value += 0.1

    if re.search(r'$', state) and re.search(r'$', next_state):
        q_value += 0.05

    if re.search(r'$', state) and not re.search(r'$', next_state):
        q_value -= 0.05

    q_value = max(0.0, min(1.0, q_value))

    return float(q_value)