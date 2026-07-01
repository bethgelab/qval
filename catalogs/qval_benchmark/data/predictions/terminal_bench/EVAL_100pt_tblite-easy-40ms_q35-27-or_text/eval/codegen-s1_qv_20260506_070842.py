import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for TerminalBench environment.
    Analyzes state transitions to estimate likelihood of successful task completion.
    """
    # Extract step count if available
    step_match = re.search(r'step[:\s]*(\d+)', state, re.IGNORECASE)
    current_step = int(step_match.group(1)) if step_match else 0
    
    # Completion indicators - strong positive signals
    completion_patterns = [
        r'\b(success|completed|done|finished|passed|verified)\b',
        r'\b(true|yes|ok)\b',
        r'✓|✔',
        r'exit code 0',
        r'status.*200',
    ]
    
    # Error indicators - strong negative signals
    error_patterns = [
        r'\b(error|fail|failed|exception|traceback)\b',
        r'\b(denied|refused|invalid|incorrect)\b',
        r'exit code [1-9]',
        r'status.*4[0-9][0-9]|status.*5[0-9][0-9]',
    ]
    
    # Progress indicators - moderate positive signals
    progress_patterns = [
        r'\b(processing|running|executing|working)\b',
        r'\b(step|stage|phase)\s*\d+',
        r'\b(created|written|generated|found|detected|identified)\b',
        r'\b(loaded|read|parsed|analyzed)\b',
    ]
    
    next_lower = next_state.lower()
    
    # Calculate scores based on pattern matches
    completion_score = sum(0.3 for p in completion_patterns if re.search(p, next_lower))
    error_score = sum(0.25 for p in error_patterns if re.search(p, next_lower))
    progress_score = sum(0.1 for p in progress_patterns if re.search(p, next_lower))
    
    # Check for meaningful output (non-empty response)
    has_output = len(next_state.strip()) > 10
    output_bonus = 0.05 if has_output else 0.0
    
    # Step efficiency factor - fewer remaining steps = higher value
    max_steps = 40
    remaining = max(0, max_steps - current_step)
    step_factor = remaining / max_steps
    
    # Compute Q-value with base neutral value
    base = 0.5
    q = base + completion_score - error_score + progress_score * 0.5 + output_bonus
    
    # Efficiency adjustment - completion is more valuable with fewer steps remaining
    if completion_score > 0:
        q += step_factor * 0.25
    elif error_score > 0:
        q -= step_factor * 0.1
    
    # Clamp to valid Q-value range
    return max(0.0, min(1.0, q))