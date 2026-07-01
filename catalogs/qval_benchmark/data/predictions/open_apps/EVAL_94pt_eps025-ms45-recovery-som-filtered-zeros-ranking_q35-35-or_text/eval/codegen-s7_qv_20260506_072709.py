import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for OpenApps browser automation task.
    Returns a float between 0.0 and 1.0 representing expected return.
    """
    
    # Check for task completion indicators in next_state
    completion_keywords = [
        r'success', r'completed', r'saved', r'sent', r'added',
        r'created', r'confirmed', r'checked', r'ok', r'✓',
        r'event created', r'task completed', r'message sent',
        r'directions', r'location found', r'code saved'
    ]
    
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Check if goal appears to be achieved
    goal_achieved = False
    for pattern in completion_keywords:
        if re.search(pattern, next_state_lower):
            goal_achieved = True
            break
    
    if goal_achieved:
        return 1.0
    
    # Analyze action type and its productivity
    action_productivity = 0.0
    
    # Productive actions
    productive_actions = [
        r'fill\(', r'click\(', r'press\(', r'navigate',
        r'send', r'add', r'create', r'edit', r'update'
    ]
    
    for pattern in productive_actions:
        if re.search(pattern, action_lower):
            action_productivity = 0.3
            break
    
    # Check for error indicators
    error_keywords = [
        r'error', r'failed', r'unable', r'invalid', r'required',
        r'not found', r'cannot', r'denied', r'×', r'✗'
    ]
    
    has_error = False
    for pattern in error_keywords:
        if re.search(pattern, next_state_lower):
            has_error = True
            break
    
    if has_error:
        return 0.1
    
    # Check for navigation progress (different page/section)
    state_sections = _extract_sections(state)
    next_state_sections = _extract_sections(next_state)
    
    section_changed = len(set(state_sections) & set(next_state_sections)) < max(len(state_sections), len(next_state_sections))
    
    if section_changed:
        action_productivity = max(action_productivity, 0.4)
    
    # Check for form filling progress
    form_fields_state = _count_form_fields(state)
    form_fields_next = _count_form_fields(next_state)
    
    if form_fields_next > form_fields_state:
        action_productivity = max(action_productivity, 0.35)
    
    # Bonus for actions that seem to complete a step
    step_completion_bonus = 0.0
    if 'fill' in action_lower and 'value' in action_lower:
        step_completion_bonus = 0.15
    elif 'click' in action_lower and any(word in next_state_lower for word in ['button', 'submit', 'confirm', 'done']):
        step_completion_bonus = 0.2
    
    # Step efficiency consideration (we don't have step count, but we can infer from state depth)
    state_depth = state_lower.count('role=') + state_lower.count('aria-')
    next_state_depth = next_state_lower.count('role=') + next_state_lower.count('aria-')
    
    # Deeper states might indicate more progress in complex tasks
    depth_progress = 0.0
    if next_state_depth > state_depth:
        depth_progress = 0.1
    
    # Calculate final Q-value estimate
    base_value = 0.2  # Baseline for any productive action
    
    q_value = base_value + action_productivity + step_completion_bonus + depth_progress
    
    # Apply error penalty
    if has_error:
        q_value *= 0.5
    
    # Clamp to valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return round(q_value, 3)


def _extract_sections(text: str) -> list:
    """Extract section/page indicators from accessibility tree."""
    sections = []
    # Look for role indicators that suggest navigation
    roles = re.findall(r'role=[\'"]?(\w+)', text.lower())
    sections.extend(roles)
    
    # Look for heading levels
    headings = re.findall(r'h[1-6]', text, re.IGNORECASE)
    sections.extend(headings)
    
    # Look for page-specific keywords
    keywords = ['calendar', 'todo', 'messenger', 'map', 'code', 'editor', 'inbox', 'events', 'tasks']
    for kw in keywords:
        if kw in text.lower():
            sections.append(kw)
    
    return sections


def _count_form_fields(text: str) -> int:
    """Count form-related elements in the accessibility tree."""
    count = 0
    text_lower = text.lower()
    
    # Count input fields
    count += len(re.findall(r'input', text_lower))
    count += len(re.findall(r'textarea', text_lower))
    count += len(re.findall(r'checkbox', text_lower))
    count += len(re.findall(r'radio', text_lower))
    count += len(re.findall(r'select', text_lower))
    
    # Count form-related roles
    count += len(re.findall(r'role=[\'"]?(textbox|button|checkbox|radio|combobox)', text_lower))
    
    return count