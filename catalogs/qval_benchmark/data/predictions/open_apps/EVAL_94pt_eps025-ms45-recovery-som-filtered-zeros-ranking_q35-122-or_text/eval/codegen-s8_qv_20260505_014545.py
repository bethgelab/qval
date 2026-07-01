def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Parse state text for completion and progress indicators
    def get_progress_score(text):
        score = 0.0
        text_lower = text.lower()
        
        # Task completion signals
        completion_terms = ['success', 'completed', 'saved', 'sent', 'created',
                           'added', 'confirmed', 'submitted', 'published', 'done']
        for term in completion_terms:
            if term in text_lower:
                score += 0.15
        
        # Form interaction signals
        form_terms = ['filled', 'entered', 'typed', 'input']
        for term in form_terms:
            if term in text_lower:
                score += 0.1
        
        # Navigation signals
        nav_terms = ['home', 'dashboard', 'main', 'welcome']
        for term in nav_terms:
            if term in text_lower:
                score += 0.05
        
        # Bid/element presence indicates interactive state
        bid_count = len(re.findall(r'bid[:\s]*\d+', text_lower))
        if bid_count > 0:
            score += min(0.1, bid_count * 0.02)
        
        return min(score, 1.0)
    
    # Measure state change magnitude
    def get_state_change(state1, state2):
        words1 = set(state1.lower().split())
        words2 = set(state2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 1.0
        return 1.0 - similarity
    
    # Evaluate action effectiveness
    def evaluate_action(action_text, state_change):
        action_lower = action_text.lower()
        score = 0.0
        
        # Productive actions
        productive = ['click', 'fill', 'press']
        for act in productive:
            if act in action_lower:
                score += 0.25
                break
        
        # Unproductive actions
        unproductive = ['noop', 'scroll']
        for act in unproductive:
            if act in action_lower:
                score -= 0.1
                break
        
        # Reward meaningful state changes
        if state_change > 0.15:
            score += 0.2
        elif state_change > 0.05:
            score += 0.1
        
        return score
    
    # Extract step information
    def get_step_info(state):
        step_match = re.search(r'step[:\s]*(\d+)', state.lower())
        if step_match:
            return int(step_match.group(1))
        return None
    
    # Main Q-value computation
    progress_current = get_progress_score(state)
    progress_next = get_progress_score(next_state)
    state_change = get_state_change(state, next_state)
    action_score = evaluate_action(action, state_change)
    
    # Base Q-value from progress
    q_value = progress_next * 0.5
    
    # Add action quality
    q_value += action_score * 0.25
    
    # Add state change contribution
    q_value += state_change * 0.15
    
    # Bonus for improvement
    if progress_next > progress_current:
        q_value += 0.1
    
    # Small penalty for stagnation
    if state_change < 0.02 and progress_next <= progress_current:
        q_value -= 0.05
    
    # Step efficiency bonus (earlier success is better)
    step = get_step_info(next_state)
    if step is not None:
        step_bonus = max(0, (45 - step) / 45) * 0.1
        q_value += step_bonus
    
    # Normalize to [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value