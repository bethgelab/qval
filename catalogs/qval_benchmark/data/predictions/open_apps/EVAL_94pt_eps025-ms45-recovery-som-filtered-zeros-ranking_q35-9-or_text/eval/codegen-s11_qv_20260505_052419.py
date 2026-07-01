import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    # Extract bid numbers from state text
    def extract_bids(text):
        matches = re.findall(r'\bid\s*=\s*(\d+)', text)
        return [int(bid) for b in matches if b.isdigit()]
    
    # Count interactive elements (bids)
    state_bids = extract_bids(state)
    next_state_bids = extract_bids(next_state)
    
    # Check if action seems to interact with a key element
    def is_action_relevant(action_text):
        keywords = ['click', 'fill', 'press', 'submit', 'select']
        action_lower = action_text.lower()
        for kw in keywords:
            if kw in action_lower:
                return True
        return False
    
    # Check if next state appears different (progress indicator)
    def state_changed(s1, s2):
        s1_normalized = s1.lower().replace('\n', ' ').strip()
        s2_normalized = s2.lower().replace('\n', ' ').strip()
        return s1_normalized != s2_normalized
    
    # Check for error indicators
    def has_error_indicator(text):
        error_patterns = ['error', 'failed', 'invalid', 'unavailable', 'not found']
        text_lower = text.lower()
        return any(p in text_lower for p in error_patterns)
    
    # Check for success indicators
    def has_success_indicator(text):
        success_patterns = ['success', 'saved', 'completed', 'done', 'added', 'sent']
        text_lower = text.lower()
        return any(p in text_lower for p in success_patterns)
    
    # Check if state has goal-related elements
    def count_form_elements(text):
        form_indicators = ['input', 'textarea', 'select', 'button', 'submit', 'form']
        count = 0
        for indicator in form_indicators:
            count += len(re.findall(rf'\b{indicator}\b', text.lower()))
        return count
    
    # Check for navigation elements
    def count_links(text):
        return len(re.findall(r'\bid\s*=\s*\d+', text))
    
    # Calculate base Q-value based on state features
    state_score = 0.0
    
    # More interactive elements = potentially more options to reach goal
    state_score += min(len(state_bids) / 10.0, 0.3)
    
    # Check for errors in current state
    if has_error_indicator(state):
        state_score -= 0.2
    
    # Check for success indicators in current state
    if has_success_indicator(state):
        state_score += 0.3
    
    # Check action relevance
    action_relevant = is_action_relevant(action)
    if action_relevant:
        state_score += 0.2
    
    # Check if state changed (progress)
    state_changed_flag = state_changed(state, next_state)
    if state_changed_flag:
        state_score += 0.1
    
    # Check form elements
    form_count = count_form_elements(state)
    if form_count > 0:
        state_score += min(form_count / 5.0, 0.2)
    
    # Check links available
    link_count = count_links(state)
    if link_count > 0:
        state_score += min(link_count / 5.0, 0.15)
    
    # Normalize to [0, 1] range
    q_value = max(0.0, min(1.0, state_score))
    
    # Apply discount for actions that don't seem to advance
    if not action_relevant and not state_changed_flag:
        q_value *= 0.5
    
    return round(q_value, 4)