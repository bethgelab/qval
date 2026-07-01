def signal_function(state: str, action: str, next_state: str):
    import re
    
    # Error patterns with severity weights for better discrimination
    error_patterns = {
        'error:': 0.18, 'failed:': 0.18, 'fail:': 0.15, 'not found': 0.12,
        'no such': 0.12, 'permission denied': 0.22, 'denied': 0.15,
        'invalid': 0.12, 'syntax error': 0.25, 'command not found': 0.18,
        'cannot': 0.12, 'unable': 0.12, 'refused': 0.15, 'missing': 0.12,
        'broken': 0.18, 'corrupted': 0.22, 'timeout': 0.15,
        'exit code 1': 0.18, 'exit code 2': 0.15, 'exit code 127': 0.22,
        'no such file': 0.15, 'directory not found': 0.15, 'command failed': 0.18,
        'traceback': 0.25, 'exception:': 0.22, 'assertionerror': 0.25,
        'pandas error': 0.18, 'keyerror': 0.18, 'valueerror': 0.18,
        'typeerror': 0.18, 'module not found': 0.22, 'import error': 0.22,
        'file not found': 0.15, 'could not': 0.12, 'unable to': 0.12,
        'failed to': 0.15
    }
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Calculate weighted error score with more granularity
    state_error_score = sum(weight for pattern, weight in error_patterns.items() if pattern in state_lower)
    next_error_score = sum(weight for pattern, weight in error_patterns.items() if pattern in next_state_lower)
    
    error_increase = max(0, next_error_score - state_error_score)
    total_error_score = next_error_score
    
    # Error penalty - more sensitive to error increases
    error_penalty = 0.12 * total_error_score + 0.22 * error_increase
    
    # Terminal success patterns
    terminal_success_patterns = [
        'verified', 'passed', 'completed', 'done', 'success',
        'exit code 0', 'all tests passed', '100%', 'benchmark passed',
        'task complete', 'result: pass', 'status: success',
        'correct', 'correctly', 'match', 'matches', 'equal',
        'hash verified', 'signature verified', 'proof verified',
        'zk verified', 'protocol verified', 'circuit verified',
        'output verified', 'final', 'submission', 'submit'
    ]
    
    is_terminal = any(p in next_state_lower for p in terminal_success_patterns)
    
    # Progress detection with better patterns
    partial_progress_patterns = [
        r'\d+/\d+', r'\d+%', r'\d+ of \d+', r'\d+ out of \d+',
        r'(\d+)/(\d+)', r'(\d+)%', r'(\d+) of (\d+)', r'(\d+) out of (\d+)'
    ]
    
    progress_fraction = 0.0
    progress_count = 0
    
    for pattern in partial_progress_patterns:
        matches = re.findall(pattern, next_state_lower)
        if matches:
            progress_count += len(matches)
            for match in matches:
                if isinstance(match, tuple):
                    try:
                        if len(match) >= 2 and match[0] and match[1]:
                            numerator = int(match[0])
                            denominator = int(match[1])
                            if denominator > 0:
                                progress_fraction = max(progress_fraction, numerator / denominator)
                    except:
                        pass
                else:
                    try:
                        if '/' in match:
                            parts = match.split('/')
                            if len(parts) == 2:
                                numerator = int(parts[0])
                                denominator = int(parts[1])
                                if denominator > 0:
                                    progress_fraction = max(progress_fraction, numerator / denominator)
                        elif '%' in match:
                            pct = int(match.replace('%', ''))
                            progress_fraction = max(progress_fraction, pct / 100.0)
                        elif 'of' in match or 'out of' in match:
                            parts = re.split(r'\s+of\s+|\s+out\s+of\s+', match)
                            if len(parts) >= 2:
                                numerator = int(parts[0])
                                denominator = int(parts[1])
                                if denominator > 0:
                                    progress_fraction = max(progress_fraction, numerator / denominator)
                    except:
                        pass
    
    # Success indicators with varying weights
    success_patterns = {
        'success': 0.14, 'completed': 0.14, 'done': 0.12, 'verified': 0.18,
        'passed': 0.14, 'ok': 0.10, 'correct': 0.14, 'found': 0.10,
        'created': 0.12, 'wrote': 0.10, 'saved': 0.12, 'written': 0.10,
        'installed': 0.12, 'configured': 0.10, 'deployed': 0.12,
        'test passed': 0.18, 'all tests': 0.18, '100%': 0.22, 'exit code 0': 0.25,
        'successfully': 0.12, 'task complete': 0.18, 'benchmark passed': 0.22,
        'result: pass': 0.18, 'status: success': 0.18, 'csv': 0.06,
        'transform': 0.06, 'convert': 0.06, 'data': 0.04, 'processed': 0.10,
        'output': 0.06, 'result': 0.06, 'generated': 0.10, 'file': 0.04,
        'record': 0.04, 'row': 0.04, 'column': 0.04, 'table': 0.04,
        'import': 0.06, 'export': 0.06, 'load': 0.06, 'write': 0.06,
        'read': 0.06, 'parse': 0.06, 'mlflow': 0.10, 'model': 0.06,
        'trained': 0.12, 'evaluated': 0.10, 'accuracy': 0.10, 'merkle': 0.12,
        'tree': 0.04, 'hash': 0.10, 'tile': 0.06, 'analysis': 0.06,
        'crypt': 0.10, 'encrypt': 0.12, 'decrypt': 0.12, 'sha': 0.10,
        'md5': 0.10, 'transformed': 0.10, 'converted': 0.10, 'saved to': 0.12,
        'wrote to': 0.12, 'output file': 0.12, 'result file': 0.12,
        'created file': 0.12, 'file created': 0.12, 'written to': 0.12,
        'stored': 0.10, 'loaded': 0.10, 'read successfully': 0.12
    }
    
    success_score = sum(weight for pattern, weight in success_patterns.items() if pattern in next_state_lower)
    success_count = sum(1 for pattern in success_patterns if pattern in next_state_lower)
    
    # Strong success indicators
    strong_success_indicators = [
        'exit code 0', 'all tests passed', '100%', 'benchmark passed',
        'task complete', 'result: pass', 'status: success', 'verified',
        'passed', 'completed', 'done', 'success', 'correct', 'correctly'
    ]
    
    has_strong_success = any(p in next_state_lower for p in strong_success_indicators)
    
    # Debugging/ongoing patterns
    debugging_patterns = [
        'debug', 'checking', 'verifying', 'analyzing', 'processing',
        'loading', 'parsing', 'reading', 'writing', 'compiling',
        'building', 'running', 'executing', 'testing', 'evaluating',
        'searching', 'finding', 'collecting', 'gathering', 'fetching',
        'downloading', 'uploading', 'installing', 'configuring',
        'initializing', 'starting', 'beginning', 'continuing',
        'in progress', 'pending', 'waiting', 'retrying',
        'log', 'output', 'display', 'print', 'show', 'showing'
    ]
    
    debugging_count = sum(1 for p in debugging_patterns if p in next_state_lower)
    
    # Calculate base probability with better calibration
    base_success_probability = 0.38
    
    if is_terminal and total_error_score < 0.1:
        if has_strong_success:
            success_probability = min(0.99, 0.92 + 0.03 * success_count)
        elif progress_fraction >= 0.95:
            success_probability = min(0.96, 0.88 + 0.04 * progress_fraction)
        else:
            progress_factor = min(1.0, progress_fraction * 1.3 + 0.1)
            success_probability = 0.78 + 0.18 * progress_factor + min(0.12, 0.025 * success_count)
    elif is_terminal and total_error_score >= 0.1:
        success_probability = max(0.18, 0.48 - 0.18 * total_error_score + 0.10 * progress_fraction)
    elif progress_count > 0:
        success_probability = 0.38 + 0.55 * progress_fraction + min(0.18, 0.035 * success_count)
    elif has_strong_success:
        success_probability = 0.72 + min(0.28, 0.045 * success_count)
    elif debugging_count > 0:
        debug_base = 0.42 + 0.28 * (debugging_count / 5.0)
        debug_bonus = min(0.18, 0.035 * success_count)
        state_words = set(state_lower.split())
        next_state_words = set(next_state_lower.split())
        new_words = len(next_state_words - state_words)
        debug_progress_bonus = min(0.14, 0.025 * new_words)
        success_probability = debug_base + debug_bonus + debug_progress_bonus
    elif success_count >= 4:
        success_probability = 0.68 + min(0.22, 0.055 * success_count)
    elif success_count >= 2:
        success_probability = 0.52 + min(0.20, 0.045 * success_count)
    elif success_count == 1:
        success_probability = 0.40 + 0.07
    else:
        success_probability = base_success_probability
    
    # Apply error penalty
    success_probability -= 0.14 * total_error_score - 0.18 * error_increase
    
    # Progress bonus - measure meaningful state changes
    state_words = set(state_lower.split())
    next_state_words = set(next_state_lower.split())
    new_words = len(next_state_words - state_words)
    progress_bonus = min(0.38, 0.04 * new_words)
    success_probability += progress_bonus
    
    # Step efficiency
    state_length = len(state)
    next_length = len(next_state)
    
    # Estimate steps based on output size and action complexity
    steps_taken = max(1, (state_length // 200) + (len(action.split()) if action else 0))
    remaining_steps = max(1, 40 - steps_taken)
    
    if is_terminal and total_error_score < 0.1:
        step_efficiency = min(1.0, 0.92 + 0.08 * (remaining_steps / 40.0))
    elif is_terminal and total_error_score >= 0.1:
        step_efficiency = max(0.28, 0.48 - 0.12 * total_error_score)
    elif progress_count > 0:
        step_efficiency = min(1.0, 0.62 + 0.38 * progress_fraction + 0.20 * (remaining_steps / 40.0))
    elif debugging_count > 0:
        debug_efficiency_base = 0.58 + 0.30 * (debugging_count / 5.0)
        step_efficiency = min(1.0, debug_efficiency_base + 0.20 * (remaining_steps / 40.0) + 0.07 * new_words)
    elif success_count >= 1 or 'exit code 0' in next_state_lower:
        step_efficiency = min(1.0, 0.78 + 0.22 * (remaining_steps / 40.0))
    elif total_error_score >= 0.4:
        step_efficiency = max(0.18, 0.38 - 0.12 * total_error_score)
    else:
        step_efficiency = 0.48 + 0.35 * (remaining_steps / 40.0)
    
    # Calculate Q-value with improved calibration
    if is_terminal and total_error_score < 0.1:
        if has_strong_success:
            q_value = min(1.0, 0.97 + 0.03 * step_efficiency)
        elif progress_fraction >= 0.95:
            q_value = min(1.0, 0.94 + 0.06 * step_efficiency)
        else:
            q_value = min(1.0, 0.90 + 0.08 * progress_fraction + 0.02 * step_efficiency)
    elif is_terminal and total_error_score >= 0.1:
        q_value = max(0.15, 0.38 - 0.14 * total_error_score + 0.14 * progress_fraction)
    elif total_error_score >= 0.6:
        q_value = max(0.04, 0.08 - 0.03 * total_error_score)
    elif total_error_score >= 0.3:
        q_value = max(0.18, 0.30 - 0.10 * total_error_score)
    elif progress_count > 0:
        q_value = 0.45 + 0.40 * progress_fraction + 0.24 * step_efficiency - error_penalty + progress_bonus
    elif debugging_count > 0:
        debug_q_base = 0.45 + 0.24 * (debugging_count / 5.0)
        debug_q_bonus = 0.20 * step_efficiency + progress_bonus - error_penalty
        q_value = debug_q_base + debug_q_bonus
    else:
        q_value = success_probability * step_efficiency - error_penalty + progress_bonus
    
    # Ensure Q-value is in valid range
    q_value = max(0.02, min(1.0, q_value))
    
    # Clamp components for interpretability
    success_probability = max(0.05, min(1.0, success_probability))
    step_efficiency = max(0.18, min(1.0, step_efficiency))
    progress_bonus = max(0.0, min(0.38, progress_bonus))
    error_penalty = max(0.0, min(0.75, error_penalty))
    
    return q_value, {
        "success_probability": success_probability,
        "step_efficiency": step_efficiency,
        "progress_bonus": progress_bonus,
        "error_penalty": error_penalty,
    }