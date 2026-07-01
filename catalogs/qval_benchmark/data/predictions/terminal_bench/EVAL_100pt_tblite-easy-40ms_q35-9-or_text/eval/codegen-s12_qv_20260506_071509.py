import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Heuristic scoring based on text analysis
    # Higher values indicate better estimated Q-value
    
    # Success indicators in next_state
    success_patterns = [
        r'\b(success|completed|done|created|copied|moved|saved|wrote|generated|built|installed|verified|passed)\b',
        r'\b(\.pyc|\.so|\.o|\.out|\.log|\.txt|\.csv|\.json)\b',
        r'^\$|^$|^\s*$',  # Clean prompt or empty output
        r'\b(0 errors|no errors|no warnings|no failures)\b'
    ]
    
    # Error indicators in next_state
    error_patterns = [
        r'\b(error|failed|exception|traceback|failure|timeout|killed|denied|permission|invalid|syntax|broken|crashed)\b',
        r'^\s*$',  # Empty output when expecting results
        r'\b(return code|exit code|status)\b.*\b([1-9]|error)\b'
    ]
    
    # Action type analysis
    action_type = action.lower()
    productive_commands = ['mkdir', 'touch', 'cp', 'mv', 'cat', 'echo', 'python', 'python3', 'pip', 'apt', 'yum', 'grep', 'find', 'chmod', 'chown', 'tee', 'cat', 'write', 'compile', 'build', 'install', 'run', 'test', 'check', 'verify', 'encrypt', 'decrypt', 'hash', 'md5', 'sha', 'base64', 'curl', 'wget', 'ssh', 'scp', 'tar', 'zip', 'unzip', 'gzip', 'gunzip', 'awk', 'sed', 'sort', 'uniq', 'head', 'tail', 'wc', 'ls', 'pwd', 'cd', 'clear', 'date', 'whoami', 'hostname', 'env', 'export', 'source', 'bash', 'sh', 'make', 'gcc', 'g++', 'makefile', 'cmake', 'docker', 'container', 'network', 'firewall', 'cron', 'systemctl', 'service', 'log', 'tail', 'grep', 'find', 'rm', 'rmdir', 'rm', 'chmod', 'chown', 'passwd', 'useradd', 'usermod', 'groupadd', 'groupmod', 'mount', 'umount', 'df', 'du', 'free', 'top', 'htop', 'ps', 'kill', 'pkill', 'nice', 'ionice', 'nice', 'nice', 'nice', 'nice']
    
    # Count matches
    success_count = sum(1 for pattern in success_patterns if re.search(pattern, next_state))
    error_count = sum(1 for pattern in error_patterns if re.search(pattern, next_state))
    
    # Check if action is productive
    action_score = 1.0 if any(cmd in action for cmd in productive_commands) else 0.5
    
    # Base Q-value calculation
    # Start with neutral value
    q_value = 0.5
    
    # Adjust based on next_state analysis
    if success_count > 0:
        q_value += success_count * 0.15
    if error_count > 0:
        q_value -= error_count * 0.25
    
    # Adjust based on action productivity
    q_value += action_score * 0.1
    
    # Clamp to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    # Additional heuristic: check if next_state looks like a clean prompt (goal state)
    prompt_indicators = [r'^\$\s', r'^user@\w+', r'^root@\w+', r'^\~/', r'^\s*#', r'^\s*%']
    is_prompt = any(re.match(pattern, next_state.strip()) for pattern in prompt_indicators)
    if is_prompt:
        q_value += 0.2
    
    # Check if action produced meaningful output
    output_length = len(next_state)
    if 50 < output_length < 500:
        q_value += 0.05
    
    return round(q_value, 3)