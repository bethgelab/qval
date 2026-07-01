import re
from collections import Counter

def signal_function(state: str) -> float:
    value = 0.5

    state_lower = state.lower()

    if 'goal achieved' in state_lower or 'task completed' in state_lower or 'success' in state_lower:
        return 1.0

    if 'error' in state_lower or 'failed' in state_lower or 'invalid' in state_lower:
        return 0.1

    if 'no actions' in state_lower or 'dead end' in state_lower or 'terminal' in state_lower:
        return 0.0

    bid_count = len(re.findall(r'\bid\d+\b', state))
    if bid_count >= 5:
        value = min(0.9, 0.3 + bid_count * 0.08)
    elif bid_count >= 2:
        value = min(0.8, 0.2 + bid_count * 0.15)
    elif bid_count >= 1:
        value = 0.3

    goal_indicators = ['calendar', 'event', 'message', 'send', 'todo', 'add', 'complete', 'submit', 'form']
    goal_matches = sum(1 for g in goal_indicators if g in state_lower)
    if goal_matches >= 2:
        value = min(0.95, value + 0.2)
    elif goal_matches == 1:
        value = min(0.85, value + 0.1)

    if 'page title' in state_lower or 'url' in state_lower:
        if 'calendar' in state_lower:
            value = min(0.9, value + 0.15)
        elif 'message' in state_lower:
            value = min(0.9, value + 0.15)
        elif 'todo' in state_lower:
            value = min(0.9, value + 0.15)

    if 'step' in state_lower and 'remaining' in state_lower:
        remaining_match = re.search(r'(\d+)\s*remaining', state_lower)
        if remaining_match:
            remaining = int(remaining_match.group(1))
            if remaining <= 3:
                value = min(0.95, value + 0.2)
            elif remaining <= 10:
                value = min(0.85, value + 0.1)

    value = max(0.0, min(1.0, value))
    return value