import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """Estimate Q-value based on state features and action appropriateness."""
    
    # Check if goal is achieved in next_state (binary reward = 1.0)
    goal_achieved_patterns = [
        r'goal.*completed', r'task.*done', r'success', r'completed',
        r'task.*succeeded', r'confirmation', r'added', r'sent',
        r'created', r'saved', r'updated'
    ]
    
    next_state_lower = next_state.lower()
    for pattern in goal_achieved_patterns:
        if re.search(pattern, next_state_lower):
            return 1.0
    
    # Check if goal is achieved in current state (unlikely but possible)
    state_lower = state.lower()
    for pattern in goal_achieved_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    # Identify task type from state
    task_types = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor']
    current_task = None
    for task in task_types:
        if task in state_lower:
            current_task = task
            break
    
    # Analyze action appropriateness
    action_lower = action.lower()
    action_score = 0.0
    
    # Good actions that typically make progress
    productive_actions = ['click', 'fill', 'press', 'submit']
    for prod_action in productive_actions:
        if prod_action in action_lower:
            action_score += 0.2
    
    # Avoid noop unless necessary (penalize excessive noops)
    if 'noop' in action_lower:
        action_score -= 0.1
    
    # Check for navigation progress (are we moving toward goal?)
    progress_indicators = [
        r'form', r'input', r'button', r'submit', r'save', r'create',
        r'add', r'new', r'edit', r'delete', r'search', r'view',
        r'list', r'menu', r'page', r'section'
    ]
    
    state_progress = 0
    for indicator in progress_indicators:
        if indicator in state_lower:
            state_progress += 1
    
    # More progress indicators = closer to goal
    progress_bonus = min(state_progress * 0.05, 0.3)
    
    # Compare state to next_state for progress
    state_elements = set(re.findall(r'\w+', state_lower))
    next_elements = set(re.findall(r'\w+', next_state_lower))
    
    # Check if we moved to a more specific/advanced state
    if len(next_elements) > len(state_elements):
        progress_bonus += 0.1
    
    # Check if we removed irrelevant elements (more focused state)
    if len(next_elements & state_elements) > 0.5 * len(state_elements):
        progress_bonus += 0.05
    
    # Base Q-value starts at 0 (no reward yet)
    base_q = 0.0
    
    # Combine factors
    estimated_q = base_q + action_score + progress_bonus
    
    # Discount based on remaining steps (estimate ~10-20 steps remaining)
    remaining_steps_estimate = 15
    discount_factor = 0.95 ** remaining_steps_estimate
    estimated_q *= discount_factor
    
    # Ensure Q-value is in valid range [0, 1]
    estimated_q = max(0.0, min(1.0, estimated_q))
    
    return estimated_q