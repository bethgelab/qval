import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) by analyzing the transition from state to next_state
    and the semantic efficacy of the action taken.
    """
    # 1. Check for immediate goal achievement (Success)
    # If the next state contains strong indicators of success, return 1.0.
    # These patterns are common in web application feedback loops.
    success_patterns = [
        r"successfully", r"sent", r"added", r"created", r"saved",
        r"updated", r"done", r"completed", r"task added", r"event created",
        r"message sent", r"confirmation"
    ]
    
    next_state_lower = next_state.lower()
    for pattern in success_patterns:
        if re.search(pattern, next_state_lower):
            return 1.0

    # 2. Detect stagnation
    # If the state hasn't changed and the action wasn't a noop, the action was ineffective.
    if state == next_state:
        if "noop" in action.lower():
            return 0.1  # Noop is a low-value neutral action
        return 0.0

    # 3. Parse Action and evaluate Efficacy
    efficacy = 0.0
    
    # regex for fill('bid', 'text')
    fill_match = re.search(r"fill\('([^']+)',\s*'([^']*)'\)", action)
    # regex for click('bid')
    click_match = re.search(r"click\('([^']+)'\)", action)
    # regex for press('bid', 'key')
    press_match = re.search(r"press\('([^']+)',\s*'([^']*)'\)", action)

    if fill_match:
        text_to_fill = fill_match.group(2).lower()
        # If the text we intended to fill actually appears in the next state, it's a good sign.
        if text_to_fill and text_to_fill in next_state_lower:
            efficacy = 0.6
    
    elif click_match:
        # For a click, efficacy is often tied to whether the page content changed significantly.
        s1_words = set(re.findall(r'\w+', state.lower()))
        s2_words = set(re.findall(r'\w+', next_state_lower))
        
        if s1_words and s2_words:
            intersection = s1_words.intersection(s2_words)
            union = s1_words.union(s2_words)
            jaccard = len(intersection) / len(union)
            
            # A moderate change (0.3 to 0.9) often indicates an updated element or menu.
            # A very low Jaccard (< 0.3) suggests a significant navigation/page load.
            if 0.3 <= jaccard <= 0.95:
                efficacy = 0.4
            elif jaccard < 0.3:
                efficacy = 0.5
            else:
                efficacy = 0.0  # Jaccard > 0.95 means almost no change
        else:
            efficacy = 0.2

    elif press_match:
        # Pressing a key (like Enter) is good if it results in state change.
        if "enter" in press_match.group(2).lower():
            s1_words = set(re.findall(r'\w+', state.lower()))
            s2_words = set(re.findall(r'\w+', next_state_lower))
            if s1_words and s2_words and s1_words != s2_words:
                efficacy = 0.4

    # 4. Calculate Semantic Progress
    # Measure the introduction of new, non-trivial words in the accessibility tree.
    s1_words = set(re.findall(r'\w+', state.lower()))
    s2_words = set(re.findall(r'\w+', next_state_lower))
    
    # Common web noise to ignore when calculating "new information"
    noise = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'to', 'of', 
             'in', 'on', 'at', 'by', 'with', 'for', 'it', 'this', 'that', 'from'}
    
    new_words = (s2_words - s1_words) - noise
    # Normalize progress score based on the density of new information.
    progress = min(len(new_words) / 15.0, 0.35)

    # 5. Aggregate Q-value
    # Q(s, a) = E[Reward | s, a]. 
    # A high efficacy + high progress should yield a high Q-value.
    q_val = efficacy + progress
    
    # Cap the value to 0.95 to leave room for the ultimate success (1.0).
    return min(max(q_val, 0.0), 0.95)