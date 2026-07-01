def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    def analyze_state(text):
        score = 0.0
        text_lower = text.lower()
        
        # Count interactive elements (bids indicate available actions)
        bid_matches = re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+|bid\s*\d+", text)
        bid_count = len(bid_matches)
        
        # Completion-related keywords
        completion_keywords = [
            'success', 'saved', 'sent', 'created', 'added', 'done', 
            'completed', 'submitted', 'published', 'confirmed'
        ]
        
        for keyword in completion_keywords:
            if keyword in text_lower:
                score += 0.08
        
        # Task-specific indicators
        task_indicators = [
            'task', 'event', 'message', 'location', 'code', 'file',
            'item', 'entry', 'note', 'appointment'
        ]
        
        for indicator in task_indicators:
            if indicator in text_lower:
                score += 0.03
        
        # Navigation indicators
        if any(page in text_lower for page in ['home', 'dashboard', 'main', 'app']):
            score += 0.05
        
        # Error indicators (penalty)
        error_keywords = ['error', 'failed', 'invalid', 'not found', 'missing', 'empty']
        for error in error_keywords:
            if error in text_lower:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def evaluate_action(action_text):
        action_lower = action_text.lower()
        
        # Productive actions
        productive_patterns = [
            r"click\s*\(",
            r"fill\s*\(",
            r"press\s*\(\s*['\"]?enter['\"]?\s*\)",
            r"submit",
            r"save"
        ]
        
        for pattern in productive_patterns:
            if re.search(pattern, action_lower):
                return 0.75
        
        # Navigation/scrolling actions
        if 'scroll' in action_lower:
            return 0.4
        
        # No-op or idle
        if 'noop' in action_lower:
            return 0.2
        
        return 0.5
    
    def measure_change(state1, state2):
        words1 = set(state1.lower().split())
        words2 = set(state2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 1.0
        return 1.0 - similarity
    
    current_score = analyze_state(state)
    next_score = analyze_state(next_state)
    
    progress = next_score - current_score
    state_change = measure_change(state, next_state)
    action_quality = evaluate_action(action)
    
    q_value = (
        0.5 * next_score +
        0.2 * max(0.0, progress) +
        0.2 * action_quality +
        0.1 * state_change
    )
    
    if progress > 0.1:
        q_value += 0.1
    
    if progress < -0.1:
        q_value -= 0.1
    
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)