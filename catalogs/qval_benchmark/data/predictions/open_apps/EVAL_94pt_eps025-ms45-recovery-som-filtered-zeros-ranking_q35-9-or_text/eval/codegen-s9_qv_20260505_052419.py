def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math

    def extract_bids(text: str) -> set:
        return set(re.findall(r'bid_\d+', text))

    current_bids = extract_bids(state)
    next_bids = extract_bids(next_state)

    new_bids = next_bids - current_bids
    removed_bids = current_bids - next_bids

    q_value = 0.0

    if len(new_bids) > 0:
        q_value += 0.3

    if len(removed_bids) > 0:
        q_value += 0.2

    if 'click' in action.lower():
        q_value += 0.15
    elif 'fill' in action.lower():
        q_value += 0.15
    elif 'press' in action.lower():
        q_value += 0.1
    elif 'scroll' in action.lower():
        q_value += 0.05
    elif 'noop' in action.lower():
        q_value -= 0.1

    goal_keywords = ['added', 'created', 'sent', 'saved', 'completed', 'success', 'goal', 'event', 'message', 'todo', 'calendar', 'code', 'open', 'edit', 'new', 'change', 'update', 'delete', 'remove']
    next_state_lower = next_state.lower()
    if any(kw in next_state_lower for kw in goal_keywords):
        q_value += 0.25

    if len(next_state) > len(state) * 0.8:
        q_value += 0.1

    if len(next_state) < len(state) * 0.7:
        q_value -= 0.1

    if 'error' in next_state_lower or 'failed' in next_state_lower or 'invalid' in next_state_lower:
        q_value -= 0.3

    if 'loading' in next_state_lower or 'waiting' in next_state_lower:
        q_value -= 0.15

    q_value = max(-0.5, min(1.0, q_value))

    return q_value