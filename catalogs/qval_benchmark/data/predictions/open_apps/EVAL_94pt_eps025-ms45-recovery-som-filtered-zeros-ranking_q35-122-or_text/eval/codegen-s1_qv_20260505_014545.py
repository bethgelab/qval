def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Extract key information from state representations
    def count_interactive_elements(s):
        # Count bid numbers which indicate interactive elements
        bids = re.findall(r'bid[\'"]?\s*[:=]\s*[\'"]?(\d+)', s)
        return len(bids)
    
    def has_success_indicators(s):
        # Check for common success/completion indicators
        success_patterns = [
            r'success', r'completed', r'saved', r'sent', r'added',
            r'event created', r'task completed', r'message sent',
            r'goal achieved', r'task done', r'submitted'
        ]
        s_lower = s.lower()
        for pattern in success_patterns:
            if re.search(pattern, s_lower):
                return True
        return False
    
    def action_meaningfulness(action_str):
        # Evaluate if action is meaningful (not noop or ineffective)
        action_lower = action_str.lower()
        
        # Noop actions have minimal value
        if 'noop' in action_lower:
            return 0.1
        
        # Scroll without context has limited value
        if 'scroll' in action_lower and 'fill' not in action_lower and 'click' not in action_lower:
            return 0.3
        
        # Fill and click actions are more meaningful
        if 'fill' in action_lower or 'click' in action_lower:
            return 0.7
        
        # Press actions can be meaningful (enter, tab, etc.)
        if 'press' in action_lower:
            if any(k in action_lower for k in ['enter', 'return', 'tab', 'delete', 'backspace']):
                return 0.6
            return 0.4
        
        return 0.2
    
    def state_progress_indicator(s):
        # Estimate progress based on state features
        # More interactive elements might indicate being on the right page
        elements = count_interactive_elements(s)
        
        # Check for task-specific indicators
        task_indicators = [
            r'todo', r'calendar', r'message', r'event', r'task',
            r'map', r'code', r'editor', r'form', r'input'
        ]
        
        s_lower = s.lower()
        indicator_count = sum(1 for pattern in task_indicators if re.search(pattern, s_lower))
        
        # Base score from element count (normalized)
        element_score = min(elements / 20.0, 1.0)
        
        # Bonus for task-related content
        indicator_score = min(indicator_count / 3.0, 1.0)
        
        return (element_score * 0.4) + (indicator_score * 0.6)
    
    def state_similarity(s1, s2):
        # Simple similarity check - if states are too similar, action had little effect
        if len(s1) == 0 or len(s2) == 0:
            return 1.0
        
        # Check if significant content changed
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 1.0
        
        overlap = len(words1 & words2)
        similarity = overlap / max(len(words1), len(words2))
        
        return similarity
    
    # Check if goal is achieved in next state
    if has_success_indicators(next_state):
        return 1.0
    
    # Evaluate action meaningfulness
    action_value = action_meaningfulness(action)
    
    # Check if state changed meaningfully
    similarity = state_similarity(state, next_state)
    
    # If state didn't change much, action was likely ineffective
    if similarity > 0.95:
        action_value *= 0.3
    
    # Estimate progress toward goal
    progress = state_progress_indicator(next_state)
    
    # Combine factors into Q-value estimate
    # Higher progress + meaningful action = higher Q-value
    q_value = (progress * 0.5) + (action_value * 0.3) + ((1.0 - similarity) * 0.2)
    
    # Normalize to reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value