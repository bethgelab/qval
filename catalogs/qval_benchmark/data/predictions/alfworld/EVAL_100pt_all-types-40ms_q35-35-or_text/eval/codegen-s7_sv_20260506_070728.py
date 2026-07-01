import re

def signal_function(state: str) -> float:
    # Normalize state for pattern matching
    s = state.lower()
    
    # Success indicators: immediate return of 1.0
    success_patterns = [
        r'\bsuccess\b',
        r'\bcomplete\b',
        r'\bclean\b',
        r'\bcorrect\b',
        r'\btask\s+complete\b'
    ]
    for pattern in success_patterns:
        if re.search(pattern, s):
            return 1.0
            
    # Failure/Error indicators: immediate return of low value
    failure_patterns = [
        r'\berror\b',
        r'\binvalid\b',
        r'\bfailed\b',
        r'\bno\s+such\b',
        r'\bcannot\b'
    ]
    for pattern in failure_patterns:
        if re.search(pattern, s):
            return 0.05
            
    # Base value for a valid state
    value = 0.1
    
    # Progress Indicator: Holding an object (manipulation milestone)
    # Check for "you are holding" or just "holding" (excluding "not holding")
    if re.search(r'\byou\s+are\s+holding\b', s):
        value += 0.4
    elif re.search(r'\bholding\b', s) and not re.search(r'\bnot\s+holding\b', s):
        value += 0.3
        
    # Progress Indicator: Object placed on surface (placement milestone)
    if re.search(r'\bis\s+on\s+the\b', s):
        value += 0.2
        
    # Progress Indicator: Agent location known (navigation milestone)
    if re.search(r'\byou\s+are\s+in\b', s):
        value += 0.1
        
    # Clamp value between 0.0 and 1.0
    if value > 1.0:
        value = 1.0
    if value < 0.0:
        value = 0.0
        
    return value