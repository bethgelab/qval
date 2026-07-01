import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for ALFWorld state-action-next_state tuple.
    Uses heuristic analysis of state features, action types, and progress indicators.
    """
    
    # Extract key features from state text
    def extract_features(text):
        features = {
            'has_object': bool(re.search(r'\b(?:knife|apple|bun|tomato|eggplant|lettuce|potato|spatula|pen|cd|pencil|penknife|scissors|creditcard|dvd)\b', text, re.I)),
            'at_location': bool(re.search(r'\b(?:on|in|at)\b', text, re.I)),
            'is_cooked': bool(re.search(r'\b(cooked|heated|frying)\b', text, re.I)),
            'is_clean': bool(re.search(r'\b(clean|washed|wiped)\b', re.I)),
            'is_open': bool(re.search(r'\b(open|opened|openable)\b', text, re.I)),
            'is_closed': bool(re.search(r'\b(closed|closed)\b', text, re.I)),
            'has_location': bool(re.search(r'\b(?:kitchen|living room|bedroom|bathroom|garbagecan|sidetable|dining table|coffeetable|countertop|desk|shelf|sinkbasin|toilet|bathtub)\b', text, re.I)),
            'object_present': len(re.findall(r'\b(knife|apple|bun|tomato|eggplant|lettuce|potato|spatula|pen|cd|pencil|penknife|scissors|creditcard|dvd)\b', text, re.I))
        }
        return features
    
    # Analyze action type
    def classify_action(action):
        action_lower = action.lower()
        if any(x in action_lower for x in ['go to', 'go to']):
            return 'navigate'
        elif any(x in action_lower for x in ['take', 'pick', 'grasp']):
            return 'pickup'
        elif any(x in action_lower for x in ['put', 'place']):
            return 'put'
        elif any(x in action_lower for x in ['heat', 'cook', 'fry']):
            return 'heat'
        elif any(x in action_lower for x in ['clean', 'wipe', 'wash']):
            return 'clean'
        elif any(x in action_lower for x in ['open', 'close']):
            return 'toggle'
        else:
            return 'other'
    
    # Check progress between states
    def detect_progress(state_features, next_state_features):
        progress_score = 0.0
        
        # Object presence change
        if next_state_features['object_present'] >= state_features['object_present']:
            progress_score += 0.1
        
        # Location-related changes
        if next_state_features['at_location'] and not state_features['at_location']:
            progress_score += 0.2
        
        # State changes (cooked, clean, open)
        if next_state_features['is_cooked'] and not state_features['is_cooked']:
            progress_score += 0.3
        if next_state_features['is_clean'] and not state_features['is_clean']:
            progress_score += 0.25
        if next_state_features['is_open'] and not state_features['is_open']:
            progress_score += 0.15
        if next_state_features['is_closed'] and not state_features['is_closed']:
            progress_score += 0.1
        
        return progress_score
    
    # Check if action is appropriate for state
    def is_appropriate_action(state_features, action_type):
        score = 0.5  # Base score
        
        # Navigation should lead to location presence
        if action_type == 'navigate':
            if state_features['has_location']:
                score += 0.2
            else:
                score += 0.1
        
        # Pickup requires object presence
        if action_type == 'pickup':
            if state_features['object_present'] > 0:
                score += 0.3
            else:
                score += 0.0
        
        # Put requires location
        if action_type == 'put':
            if state_features['at_location']:
                score += 0.3
            else:
                score += 0.1
        
        # Heat/Clean require appropriate state
        if action_type in ['heat', 'clean']:
            if state_features['object_present'] > 0:
                score += 0.2
        
        return score
    
    # Check for terminal-like conditions
    def is_likely_terminal(state_text):
        # Success indicators
        success_patterns = [
            r'\b(done|success|completed|finished)\b',
            r'\b(task completed)\b',
            r'\b(object is now at target)\b',
            r'\bgoal achieved\b'
        ]
        
        for pattern in success_patterns:
            if re.search(pattern, state_text, re.I):
                return 1.0
        
        # Check if object is at desired location
        location_patterns = [
            r'\b(?:on|in|at)\s+\w+\s+(?:counter|table|sink|shelf)\b',
            r'\b(?:placed|located)\s+at\s+\w+\b'
        ]
        
        for pattern in location_patterns:
            if re.search(pattern, state_text, re.I):
                return 0.7
        
        return 0.0
    
    # Extract features from state and next_state
    state_features = extract_features(state)
    next_state_features = extract_features(next_state)
    
    # Classify action
    action_type = classify_action(action)
    
    # Calculate components
    progress_score = detect_progress(state_features, next_state_features)
    appropriateness_score = is_appropriate_action(state_features, action_type)
    terminal_indicator = is_likely_terminal(next_state)
    
    # Combine scores with weights
    # Progress and terminal conditions are most important
    q_estimate = (
        0.4 * terminal_indicator +
        0.3 * progress_score +
        0.2 * appropriateness_score +
        0.1 * (state_features['has_object'] and state_features['has_location'])
    )
    
    # Ensure reasonable range
    q_estimate = max(0.0, min(1.0, q_estimate))
    
    return q_estimate