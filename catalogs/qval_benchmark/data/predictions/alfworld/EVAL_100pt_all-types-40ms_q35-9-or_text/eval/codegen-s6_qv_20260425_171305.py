def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    from collections import Counter
    
    def extract_locations(text: str) -> list:
        locations = ['kitchen', 'bedroom', 'living room', 'bathroom', 'hallway', 
                     'dining room', 'study', 'office', 'garage', 'basement', 'attic']
        found = []
        for loc in locations:
            if re.search(rf'\b{loc}\b', text, re.IGNORECASE):
                found.append(loc)
        return found
    
    def extract_objects(text: str) -> list:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [w for w in words if len(w) >= 3 and w not in ['the', 'a', 'an', 'is', 'are', 
                     'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'and', 'or']]
    
    def get_agent_location(text: str) -> str:
        patterns = [
            r'my location is ([a-z\s]+)',
            r'at ([a-z\s]+)',
            r'in ([a-z\s]+)',
            r'location: ([a-z\s]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "unknown"
    
    def is_goal_mentioned(text: str) -> bool:
        goal_keywords = ['goal', 'task', 'move', 'clean', 'put', 'take', 'bring', 
                        'wash', 'dry', 'fold', 'iron', 'open', 'close', 'lock', 'unlock']
        return any(re.search(rf'\b{kw}\b', text, re.IGNORECASE) for kw in goal_keywords)
    
    def count_goal_progress(text: str) -> int:
        progress_words = ['done', 'complete', 'finished', 'success', 'successfully',
                         'moved', 'cleaned', 'placed', 'taken', 'brought', 'opened', 'closed']
        return sum(1 for word in progress_words if re.search(rf'\b{word}\b', text, re.IGNORECASE))
    
    def action_matches_state(text: str, action: str) -> float:
        action_lower = action.lower()
        state_lower = text.lower()
        if 'navigate' in action_lower:
            return 1.0 if 'move' in state_lower or 'go' in state_lower else 0.3
        if 'pick' in action_lower or 'take' in action_lower:
            return 1.0 if 'pick' in state_lower or 'take' in state_lower else 0.2
        if 'place' in action_lower or 'put' in action_lower:
            return 1.0 if 'place' in state_lower or 'put' in state_lower else 0.2
        if 'clean' in action_lower:
            return 1.0 if 'clean' in state_lower else 0.2
        if 'open' in action_lower:
            return 1.0 if 'open' in state_lower else 0.2
        if 'close' in action_lower:
            return 1.0 if 'close' in state_lower else 0.2
        return 0.5
    
    current_locations = extract_locations(state)
    next_locations = extract_locations(next_state)
    
    current_objects = extract_objects(state)
    next_objects = extract_objects(next_state)
    
    agent_loc_current = get_agent_location(state)
    agent_loc_next = get_agent_location(next_state)
    
    goal_mentioned = is_goal_mentioned(state)
    goal_mentioned_next = is_goal_mentioned(next_state)
    
    progress_current = count_goal_progress(state)
    progress_next = count_goal_progress(next_state)
    
    action_relevance = action_matches_state(state, action)
    
    location_change = 0.0
    if agent_loc_current != agent_loc_next:
        location_change = 0.5
    elif current_locations != next_locations:
        location_change = 0.3
    
    progress_improvement = 0.0
    if progress_next > progress_current:
        progress_improvement = 0.5
    elif progress_next < progress_current:
        progress_improvement = -0.5
    
    state_similarity = 0.0
    common_objects = set(current_objects) & set(next_objects)
    if len(common_objects) > 0:
        state_similarity = 0.3
    
    base_score = 0.5
    if goal_mentioned:
        base_score += 0.3
    if progress_next > progress_current:
        base_score += 0.2
    if action_relevance > 0.7:
        base_score += 0.2
    if location_change > 0:
        base_score += 0.1
    
    penalty = 0.0
    if progress_next < progress_current:
        penalty = 0.4
    if action_relevance < 0.3:
        penalty += 0.2
    
    final_score = max(0.0, min(1.0, base_score - penalty + action_relevance * 0.1))
    
    return float(final_score)