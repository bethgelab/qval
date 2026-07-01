import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for ALFWorld state-action-next_state tuple.
    
    Uses heuristic features from state representation to estimate
    the expected return, without lookahead or simulation.
    """
    
    # Base Q-value (neutral starting point)
    base_q = 0.0
    
    # Extract key features from states
    state_features = _extract_state_features(state)
    next_state_features = _extract_state_features(next_state)
    
    # Calculate progress score based on state changes
    progress_score = _calculate_progress(state_features, next_state_features)
    
    # Calculate action quality score
    action_quality = _evaluate_action_quality(action, next_state)
    
    # Calculate goal proximity score
    goal_proximity = _estimate_goal_proximity(next_state)
    
    # Combine scores with weights
    q_value = (
        base_q * 0.1 +
        progress_score * 0.35 +
        action_quality * 0.25 +
        goal_proximity * 0.40
    )
    
    # Clip to reasonable range
    return max(0.0, min(1.0, q_value))


def _extract_state_features(state: str) -> dict:
    """Extract relevant features from state string."""
    features = {
        'locations': set(),
        'objects': set(),
        'has_cleaned': False,
        'has_moved': False,
        'has_taken': False,
        'has_placed': False,
        'has_used': False,
        'has_opened': False,
        'has_closed': False,
        'has_filled': False,
        'has_empty': False,
        'has_on': False,
        'has_in': False,
        'completion_indicators': 0
    }
    
    # Common ALFWorld patterns
    state_lower = state.lower()
    
    # Extract locations (common in ALFWorld)
    location_patterns = [
        r'\b(kitchen|bedroom|livingroom|bathroom|garage|table|desk|counter|sink|cabinet|drawer|shelf|floor|bathtub|toilet|nightstand)\b'
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, state_lower)
        features['locations'].update(matches)
    
    # Extract objects (common in ALFWorld)
    object_patterns = [
        r'\b(book|cellphone|creditcard|keychain|key|remote|pen|pencil|paper|newspaper|scissors|watch|alarmclock|bowl|cup|dish|fork|knife|plate|spoon|vase|window|box|pottedplant|safe|sofa|tvstand|tv|laptop|laptopcase|microwave|mug|pencil|pen|pencilcase|pillow|pot|saltshaker|saucer|scissors|showercurtain|soapbottle|soapbar|soapdish|statue|toiletpaper|toothbrush|toothpaste|towel|towelholder|tumbler|vase|wastebasket)\b'
    ]
    for pattern in object_patterns:
        matches = re.findall(pattern, state_lower)
        features['objects'].update(matches)
    
    # Action indicators
    action_keywords = {
        'clean': 'has_cleaned',
        'take': 'has_taken',
        'put': 'has_placed',
        'use': 'has_used',
        'open': 'has_opened',
        'close': 'has_closed',
        'fill': 'has_filled',
        'empty': 'has_empty',
        'on': 'has_on',
        'in': 'has_in',
        'move': 'has_moved'
    }
    
    for keyword, feature_name in action_keywords.items():
        if keyword in state_lower:
            features[feature_name] = True
    
    # Look for completion indicators
    completion_patterns = [
        r'correct',
        r'correct answer',
        r'correctly',
        r'completed',
        r'task completed',
        r'goal achieved',
        r'you have solved'
    ]
    for pattern in completion_patterns:
        if re.search(pattern, state_lower):
            features['completion_indicators'] += 1
    
    return features


def _calculate_progress(state_features: dict, next_state_features: dict) -> float:
    """Calculate progress score based on state changes."""
    score = 0.0
    
    # Check for action completion indicators
    if next_state_features['completion_indicators'] > 0:
        score += 0.5
    
    # Check for object movement progress
    new_locations = next_state_features['locations'] - state_features['locations']
    if len(new_locations) > 0:
        score += 0.15
    
    # Check for object state changes
    state_changes = [
        ('has_cleaned', False, True),
        ('has_taken', False, True),
        ('has_placed', False, True),
        ('has_opened', False, True),
        ('has_filled', False, True)
    ]
    
    for feature, old_val, new_val in state_changes:
        if state_features[feature] == old_val and next_state_features[feature] == new_val:
            score += 0.1
    
    # Check for object presence in next state
    if len(next_state_features['objects']) > len(state_features['objects']):
        score += 0.1
    
    # Normalize score
    return min(1.0, score)


def _evaluate_action_quality(action: str, next_state: str) -> float:
    """Evaluate if the action seems appropriate based on next state."""
    score = 0.5  # Base neutral score
    
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Valid action patterns in ALFWorld
    valid_patterns = [
        r'go to',
        r'go ',
        r'take .* from',
        r'take .* in',
        r'put .* in',
        r'put .* on',
        r'clean .* with',
        r'use .* with',
        r'open .*',
        r'close .*',
        r'fill .* with',
        r'empty .*',
        r'browse .*',
        r'cook .*',
        r'eat .*'
    ]
    
    # Check if action matches valid patterns
    pattern_matched = False
    for pattern in valid_patterns:
        if re.search(pattern, action_lower):
            pattern_matched = True
            break
    
    if pattern_matched:
        score += 0.2
    
    # Check if next state shows action was executed
    if len(next_state) > len(action) + 10:
        score += 0.1
    
    # Check for error messages in next state
    error_indicators = ['error', 'cannot', 'unable', 'invalid', 'wrong']
    has_error = any(ind in next_state_lower for ind in error_indicators)
    if has_error:
        score -= 0.3
    
    return max(0.0, min(1.0, score))


def _estimate_goal_proximity(next_state: str) -> float:
    """Estimate how close the next state is to goal completion."""
    score = 0.0
    
    next_state_lower = next_state.lower()
    
    # Strong completion indicators
    completion_phrases = [
        'correct',
        'correct answer',
        'correctly',
        'completed',
        'task completed',
        'goal achieved',
        'you have solved',
        'mission accomplished',
        'success'
    ]
    
    for phrase in completion_phrases:
        if phrase in next_state_lower:
            score = 1.0
            break
    
    # Medium indicators - task seems to be progressing well
    progress_phrases = [
        'put',
        'in',
        'on',
        'at',
        'to',
        'with',
        'from'
    ]
    
    if any(phrase in next_state_lower for phrase in progress_phrases):
        score = max(score, 0.4)
    
    # Check for object-location pairs that suggest task completion
    location_keywords = ['table', 'desk', 'counter', 'sink', 'cabinet', 'drawer', 
                         'shelf', 'floor', 'bathroom', 'kitchen', 'bedroom']
    if any(loc in next_state_lower for loc in location_keywords):
        score = max(score, 0.3)
    
    return min(1.0, score)