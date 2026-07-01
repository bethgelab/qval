def signal_function(state: str) -> float:
    import re
    
    # Check for explicit success conditions
    if re.search(r'(task\s*complete|success|done)', state, re.IGNORECASE):
        return 1.0
        
    score = 0.0
    
    # Heuristic 1: Agent has picked up the object (High progress)
    if re.search(r'holding', state, re.IGNORECASE):
        score += 0.4
        
    # Heuristic 2: Agent knows their location (Medium progress)
    if re.search(r'in\s+the\s+\w+', state, re.IGNORECASE):
        score += 0.2
        
    # Heuristic 3: Object location is known/visible (Low progress)
    if re.search(r'on\s+the\s+\w+', state, re.IGNORECASE):
        score += 0.1
        
    # Heuristic 4: Step limit penalty (Risk of timeout)
    step_match = re.search(r'step\s+(\d+)', state, re.IGNORECASE)
    if step_match:
        steps = int(step_match.group(1))
        if steps > 35:
            score -= 0.5
        elif steps > 25:
            score -= 0.2
            
    # Heuristic 5: State complexity/length penalty (Proxy for steps taken)
    # Longer state strings often imply more history/steps
    if len(state) > 4000:
        score -= 0.1
        
    # Clamp score to [0.0, 1.0]
    return min(1.0, max(0.0, score))