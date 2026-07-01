def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    q_value = 0.0
    
    # Check for goal completion indicators in next_state
    success_patterns = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bdone\b', r'\bfinished\b',
        r'\bpassed\b', r'\bverification.*success', r'\btest.*pass',
        r'100%', r'\bcorrect\b', r'\bsolved\b', r'\banswer\b'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.6
            break
    
    # Check for error indicators (penalize)
    error_patterns = [
        r'\berror\b', r'\bfail', r'\bexception\b', r'\bpermission denied\b',
        r'\bnot found\b', r'\bno such', r'\binvalid\b', r'\bwrong\b',
        r'\bincorrect\b', r'\brefused\b', r'\bdenied\b', r'\btimeout\b'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value -= 0.4
            break
    
    # Check for progress indicators (file operations, installations, etc.)
    progress_patterns = [
        r'\bcreated\b', r'\bwrote\b', r'\bsaved\b', r'\bgenerated\b',
        r'\binstalled\b', r'\bbuilt\b', r'\bcompiled\b', r'\bexecuted\b',
        r'\bdownloaded\b', r'\bextracted\b', r'\brunning\b', r'\bstarted\b'
    ]
    
    for pattern in progress_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.25
    
    # Check if action is meaningful (non-trivial commands)
    action_lower = action.lower().strip()
    meaningful_actions = [
        r'cat\b', r'grep\b', r'find\b', r'awk\b', r'sed\b', r'curl\b',
        r'wget\b', r'pip\b', r'apt\b', r'yum\b', r'git\b', r'python\b',
        r'python3\b', r'gcc\b', r'g++\b', r'make\b', r'npm\b', r'node\b',
        r'echo\b', r'touch\b', r'mkdir\b', r'cp\b', r'mv\b', r'rm\b',
        r'chmod\b', r'chown\b', r'ln\b', r'echo\b', r'printf\b', r'bc\b',
        r'sort\b', r'uniq\b', r'wc\b', r'head\b', r'tail\b', r'cut\b',
        r'xargs\b', r'while\b', r'for\b', r'if\b', r'fi\b', r'done\b'
    ]
    
    action_is_meaningful = False
    for pattern in meaningful_actions:
        if re.search(pattern, action_lower):
            action_is_meaningful = True
            break
    
    if action_is_meaningful:
        q_value += 0.15
    elif len(action_lower) > 5 and action_lower not in ['ls', 'pwd', 'cd', 'clear']:
        q_value += 0.05
    
    # Check for output/progress in next_state (non-empty meaningful output)
    if next_state and len(next_state) > 20:
        q_value += 0.1
    
    # Check if state contains task context indicators
    task_keywords = [
        'terminal', 'bench', 'task', 'challenge', 'problem',
        'exercise', 'assignment', 'goal', 'objective'
    ]
    
    state_lower = state.lower()
    for keyword in task_keywords:
        if keyword in state_lower:
            q_value += 0.05
            break
    
    # Check for step count indicators (prefer fewer steps)
    step_match = re.search(r'step\s*[:\s]*(\d+)', state, re.IGNORECASE)
    if step_match:
        step_num = int(step_match.group(1))
        # Reward earlier completion (40 step limit)
        step_bonus = max(0, (40 - step_num) / 40) * 0.2
        q_value += step_bonus
    
    # Check for file/directory state changes
    if 'new' in next_state.lower() or 'changed' in next_state.lower():
        q_value += 0.1
    
    # Cap the Q-value in reasonable range [0, 1.2]
    q_value = max(0.0, min(1.2, q_value))
    
    return float(q_value)