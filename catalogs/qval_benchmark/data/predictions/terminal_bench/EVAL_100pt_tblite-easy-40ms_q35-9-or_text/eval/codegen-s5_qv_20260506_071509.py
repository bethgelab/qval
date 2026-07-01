import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    success_keywords = ['success', 'completed', 'passed', 'done', 'ok', 
                       'exit code 0', 'true', 'yes', 'verified',
                       'created', 'wrote', 'saved', 'generated', 'solved',
                       'test passed', 'passed all', 'all tests passed',
                       'ok', 'ready', 'running', 'active']
    
    failure_keywords = ['error', 'failed', 'warning', 'fatal',
                       'exit code 1', 'exit code 2', 'exception',
                       'false', 'no', 'invalid', 'broken', 'missing',
                       'not found', 'permission denied', 'timeout',
                       'fail', 'crash', 'broken', 'invalid']
    
    progress_keywords = ['created', 'wrote', 'saved', 'generated',
                        'processed', 'analyzed', 'completed', 'finished',
                        'step', 'progress', 'done', 'success',
                        'output', 'result', 'found', 'detected']
    
    def count_keyword_matches(text, keywords):
        text_lower = text.lower()
        count = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                count += 1
        return count
    
    def extract_numbers(text):
        numbers = re.findall(r'\d+', text)
        return [int(n) for n in numbers if n.isdigit()]
    
    state_success = count_keyword_matches(state, success_keywords)
    state_failure = count_keyword_matches(state, failure_keywords)
    state_progress = count_keyword_matches(state, progress_keywords)
    
    next_state_success = count_keyword_matches(next_state, success_keywords)
    next_state_failure = count_keyword_matches(next_state, failure_keywords)
    next_state_progress = count_keyword_matches(next_state, progress_keywords)
    
    state_quality = state_success - state_failure
    next_state_quality = next_state_success - next_state_failure
    
    improvement = (next_state_success + next_state_progress - state_success - state_progress) - (next_state_failure - state_failure)
    
    max_quality = max(10, state_success + state_failure + next_state_success + next_state_failure)
    normalized_quality = (state_quality + next_state_quality) / max_quality
    
    improvement_bonus = min(0.5, improvement / 10)
    
    q_value = normalized_quality + improvement_bonus
    
    return max(0.0, min(1.0, q_value))