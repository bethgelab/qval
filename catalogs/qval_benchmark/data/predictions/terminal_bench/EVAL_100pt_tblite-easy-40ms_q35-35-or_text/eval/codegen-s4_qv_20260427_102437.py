import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for terminal environment based on state analysis.
    """
    q_value = 0.0
    
    # Check if we're at or near completion
    completion_patterns = [
        r'\bcomplete\b', r'\bsuccess\b', r'\bpass\b', r'\bdone\b',
        r'\bverified\b', r'\bverified\b', r'\btest passed\b',
        r'\bresult: success\b', r'\bstatus: success\b',
        r'\ball tests passed\b', r'\b✓\b', r'\bOK\b'
    ]
    
    combined_state = state + " " + next_state
    
    completion_matches = sum(1 for p in completion_patterns if re.search(p, combined_state, re.IGNORECASE))
    
    if completion_matches >= 2:
        q_value = 1.0
    elif completion_matches >= 1:
        q_value = 0.8
    else:
        # Check for error/failure patterns
        error_patterns = [
            r'\berror\b', r'\bfailed\b', r'\bfailure\b', r'\bexception\b',
            r'\btraceback\b', r'\bnot found\b', r'\bpermission denied\b',
            r'\bexit code \d+\b', r'\bno such file\b', r'\bcommand not found\b'
        ]
        
        error_matches = sum(1 for p in error_patterns if re.search(p, combined_state, re.IGNORECASE))
        
        if error_matches >= 2:
            q_value = 0.1
        elif error_matches >= 1:
            q_value = 0.3
        else:
            # Check for progress indicators
            progress_patterns = [
                r'step \d+ of \d+', r'progress \d+%', r'task \d+/\d+',
                r'processing', r'running', r'executing', r'working'
            ]
            
            progress_matches = sum(1 for p in progress_patterns if re.search(p, combined_state, re.IGNORECASE))
            
            if progress_matches >= 2:
                q_value = 0.5
            elif progress_matches >= 1:
                q_value = 0.4
            else:
                # Check for productive actions
                productive_actions = [
                    r'cat', r'grep', r'ls', r'cd', r'mkdir', r'touch',
                    r'echo', r'cp', r'mv', r'rm', r'chmod', r'chown',
                    r'pip install', r'apt', r'docker', r'git', r'python',
                    r'node', r'npm', r'conda', r'export', r'source'
                ]
                
                action_lower = action.lower()
                productive_action_matches = sum(1 for p in productive_actions if p in action_lower)
                
                if productive_action_matches >= 2:
                    q_value = 0.4
                elif productive_action_matches >= 1:
                    q_value = 0.3
                else:
                    # Default baseline
                    q_value = 0.2
    
    # Adjust based on state transition quality
    if next_state and not state:
        q_value = min(q_value + 0.1, 1.0)
    
    # If next_state shows improvement over state
    if re.search(r'\bcomplete\b', next_state, re.IGNORECASE) and not re.search(r'\bcomplete\b', state, re.IGNORECASE):
        q_value = max(q_value, 0.9)
    
    return q_value