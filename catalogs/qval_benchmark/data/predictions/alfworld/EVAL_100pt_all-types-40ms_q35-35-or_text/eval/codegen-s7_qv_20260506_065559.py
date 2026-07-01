import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Pattern to detect successful completion
    success_pattern = re.compile(r'\b(success|completed|finished|done)\b', re.IGNORECASE)
    
    # Check for immediate success in the next state
    if success_pattern.search(next_state):
        return 1.0
    
    # Check if the state has changed at all (ineffective action)
    # If state and next_state are identical, no progress was made
    if state == next_state:
        return 0.0
    
    # Patterns for valid actions and progress indicators
    action_pattern = re.compile(r'\b(go|take|put|clean|heat|cool|wash|examine)\b', re.IGNORECASE)
    progress_pattern = re.compile(r'\b(holding|on|in|clean|hot|cold|dirty)\b', re.IGNORECASE)
    location_pattern = re.compile(r'\b(sink|countertop|table|microwave|stove|fridge|drawer|cabinet|bed|bathtub|toilet)\b', re.IGNORECASE)
    
    score = 0.0
    
    # Reward for taking a valid action
    if action_pattern.search(action):
        score += 0.3
    
    # Reward for state changes indicating object manipulation
    if progress_pattern.search(next_state):
        score += 0.3
        
    # Reward for being in relevant locations
    if location_pattern.search(next_state):
        score += 0.2
        
    # Cap the score at 1.0
    if score > 1.0:
        score = 1.0
        
    return score