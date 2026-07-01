import re

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value for an action in ALFWorld by combining:
    1. Action Utility: How well the action aligns with the task entities.
    2. State Potential: How many goal-related entities are present in the next state.
    3. Discovery Reward: The presence of new, meaningful information in the next state.
    4. Redundancy Penalty: A penalty for actions that result in little to no state change.
    """
    next_s_low = next_state.lower()
    
    # 1. Terminal Success Detection
    success_keywords = ["task is complete", "successfully", "goal reached", "finished", "you have successfully"]
    if any(kw in next_s_low for kw in success_keywords):
        return 1.0, {"success": 1.0}

    # 2. Goal Extraction
    # We aim to extract key entities (nouns/adjectives) from the instruction.
    stop_words = {
        'the', 'is', 'a', 'an', 'in', 'on', 'at', 'to', 'you', 'are', 'there', 'and', 'it', 'of', 'with', 
        'can', 'your', 'has', 'have', 'was', 'were', 'this', 'that', 'from', 'for', 'by', 'but', 'not', 
        'all', 'some', 'any', 'put', 'place', 'move', 'take', 'grab', 'pick', 'find', 'clean', 'wash', 
        'scrub', 'drop', 'into', 'go', 'walk', 'look', 'search', 'open', 'close'
    }
    
    instruction_match = re.search(r'instruction:\s*(.*?)(?:\.|$)', state, re.IGNORECASE)
    if instruction_match:
        goal_text = instruction_match.group(1).lower()
    else:
        # Fallback to the first part of the state if 'instruction:' prefix is missing
        goal_text = state.split('.')[0].lower()
            
    raw_goal_words = re.findall(r'\b\w+\b', goal_text)
    goal_words = {w for w in raw_goal_words if w not in stop_words and len(w) > 2}

    # 3. Action Utility (u)
    # High if the action targets a goal entity, medium if it's movement, low otherwise.
    act_words = re.findall(r'\b\w+\b', action.lower())
    u = 0.1
    if act_words:
        verb = act_words[0]
        if any(w in goal_words for w in act_words[1:]):
            u = 0.4
        elif verb in ['go', 'move', 'walk', 'navigate', 'look', 'search']:
            u = 0.2
        
        # Penalty for failure/stagnation
        fail_keywords = ["nothing happens", "cannot", "not here", "doesn't work", "is not", "not found"]
        if any(kw in next_s_low for kw in fail_keywords):
            u = 0.0
    
    # 4. State Potential (s)
    # Measures how many goal-relevant objects or locations are now visible or being interacted with.
    s = 0.0
    if goal_words:
        found_count = sum(1 for w in goal_words if w in next_s_low)
        s = 0.4 * (found_count / len(goal_words))
    
    # 5. Discovery Reward (d)
    # Measures the introduction of new semantic tokens in the observation.
    def get_words(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    words_s = get_words(state)
    words_n = get_words(next_state)
    new_words = words_n - words_s - stop_words
    new_words = {w for w in new_words if len(w) > 2}
    d = min(0.2, len(new_words) * 0.05)

    # 6. Redundancy Penalty (p)
    # Penalize repetitive actions that do not progress the state.
    p = 0.0
    if len(words_n - words_s) <= 1 and not any(v in action.lower() for v in ['go', 'move', 'walk', 'look', 'search']):
        p = -0.2
    
    # 7. Weighted Sum Calculation
    # We cap individual components to ensure the sum u+s+d+p is within [0, 1].
    u = min(0.4, u)
    s = min(0.4, s)
    d = min(0.2, d)
    p = max(-0.2, p)
    
    total = u + s + d + p
    
    # Adjust u to ensure total is non-negative and the sum rule is respected.
    if total < 0:
        u = -s - d - p
        total = 0.0

    return total, {
        "action_utility": u,
        "state_potential": s,
        "discovery_reward": d,
        "redundancy_penalty": p
    }