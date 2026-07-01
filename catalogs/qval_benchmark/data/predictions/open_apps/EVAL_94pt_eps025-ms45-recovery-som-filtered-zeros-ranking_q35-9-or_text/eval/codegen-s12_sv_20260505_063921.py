import re
import math

def signal_function(state: str) -> float:
    # Base value representing uncertainty when state is unclear
    base_value = 0.5
    
    # Count bid numbers (interactive elements)
    bid_matches = re.findall(r"bid\s*['\"]?\d+['\"]?", state)
    bid_count = len(bid_matches) if bid_matches else 0
    
    # Count form elements (input fields, buttons)
    form_elements = len(re.findall(r"(input|button|textarea|select)", state, re.IGNORECASE))
    
    # Check for progress indicators
    progress_keywords = ['success', 'complete', 'done', 'saved', 'added', 'created', 'sent']
    progress_found = any(kw.lower() in state.lower() for kw in progress_keywords)
    
    # Check for error indicators
    error_keywords = ['error', 'failed', 'invalid', 'missing', 'cannot', 'unable']
    error_found = any(kw.lower() in state.lower() for kw in error_keywords)
    
    # Check for navigation context (are we on the right page?)
    page_indicators = ['calendar', 'todo', 'messenger', 'maps', 'editor', 'code']
    page_context = sum(1 for pg in page_indicators if pg.lower() in state.lower())
    
    # Estimate steps remaining (fewer elements = closer to goal)
    # Heuristic: more interactive elements = more steps needed
    estimated_steps_remaining = max(1, 10 - min(bid_count, 8))
    
    # Calculate progress score
    progress_score = 1.0 if progress_found else 0.5
    
    # Calculate penalty for errors
    error_penalty = -0.3 if error_found else 0.0
    
    # Calculate page context bonus
    page_bonus = min(0.15 * page_context, 0.15)
    
    # Calculate base heuristic from element count
    # More elements visible = potentially more options to reach goal
    element_bonus = min(0.1 * min(bid_count, 5), 0.15)
    
    # Calculate steps penalty (fewer steps = better)
    steps_penalty = 0.05 * (estimated_steps_remaining - 5)
    
    # Combine all factors
    estimated_value = base_value + progress_score * 0.3 + page_bonus + element_bonus + steps_penalty + error_penalty
    
    # Clamp to reasonable range [0, 1]
    estimated_value = max(0.0, min(1.0, estimated_value))
    
    return estimated_value