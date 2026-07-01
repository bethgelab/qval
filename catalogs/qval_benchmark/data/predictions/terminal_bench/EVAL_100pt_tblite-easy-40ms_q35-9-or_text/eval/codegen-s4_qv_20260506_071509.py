def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    def count_occurrences(text, patterns):
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    success_patterns = ['success', 'complete', 'done', 'verified', 'passed', 'ok', 'true', 'created', 'wrote', 'generated', 'set', 'config', 'task', 'goal', 'finish', 'succeed']
    error_patterns = ['error', 'failed', 'exception', 'crash', 'fatal', 'invalid', 'missing', 'cannot', 'unable', 'false', 'broken', 'fail', 'fatal', 'abort']
    prompt_patterns = ['\$ ', '# ', 'user@', 'root@', 'bash$', 'zsh$', 'PS1', 'prompt']
    
    success_count_state = count_occurrences(state, success_patterns)
    error_count_state = count_occurrences(state, error_patterns)
    success_count_next = count_occurrences(next_state, success_patterns)
    error_count_next = count_occurrences(next_state, error_patterns)
    
    has_prompt_state = bool(re.search(r'[\$#]|\s*user@|root@|bash|zsh', state))
    has_prompt_next = bool(re.search(r'[\$#]|\s*user@|root@|bash|zsh', next_state))
    
    progress_score = 0.0
    
    if error_count_next < error_count_state:
        progress_score += 0.3
    if success_count_next > success_count_state:
        progress_score += 0.3
    if has_prompt_next and not has_prompt_state:
        progress_score += 0.2
    if has_prompt_state and has_prompt_next:
        progress_score += 0.1
    
    if success_count_next >= 3 and error_count_next == 0:
        progress_score += 0.2
    
    if success_count_next >= 2 and success_count_state == 0:
        progress_score += 0.15
    
    if error_count_next >= 2:
        progress_score -= 0.3
    
    q_value = max(0.0, min(1.0, progress_score))
    
    return q_value