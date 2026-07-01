def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    def analyze_state(text: str) -> float:
        score = 0.0
        
        success_indicators = ['success', 'complete', 'done', 'finished', 'created', 'generated', 'wrote', 'output', 'verified', 'passed', 'ok', 'true', 'yes', '0 errors', 'no errors', 'exit code 0', 'successful']
        failure_indicators = ['error', 'fail', 'failed', 'exception', 'traceback', 'syntax', 'invalid', 'permission', 'denied', 'refused', 'timeout', 'killed', 'crash', 'broken', 'false', 'no', 'missing', 'not found', 'exit code 1', 'error:']
        
        success_count = sum(1 for indicator in success_indicators if indicator.lower() in text.lower())
        failure_count = sum(1 for indicator in failure_indicators if indicator.lower() in text.lower())
        
        if success_count > 0:
            score += 0.3 * success_count
        if failure_count > 0:
            score -= 0.5 * failure_count
        
        if success_count > failure_count:
            score += 0.2
        elif failure_count > success_count:
            score -= 0.2
        
        progress_indicators = ['step', 'line', 'file', 'output', 'result', 'status', 'command', 'prompt', 'user@', '~', '$', '#', 'root@', 'admin@']
        progress_count = sum(1 for indicator in progress_indicators if indicator.lower() in text.lower())
        if progress_count > 0:
            score += 0.1 * min(progress_count, 3)
        
        return score
    
    state_score = analyze_state(state)
    next_state_score = analyze_state(next_state)
    
    action_keywords = ['ls', 'cd', 'mkdir', 'touch', 'echo', 'cat', 'grep', 'find', 'python', 'bash', 'chmod', 'chown', 'scp', 'ssh', 'git', 'make', 'pip', 'apt', 'yum', 'docker', 'curl', 'wget', 'tar', 'zip', 'unzip', 'rm', 'cp', 'mv', 'chmod', 'passwd', 'sudo', 'service', 'systemctl', 'crontab', 'nginx', 'mysql', 'redis', 'postgres', 'mongo', 'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'notebook', 'conda', 'virtualenv', 'pip3', 'python3', 'python2', 'node', 'npm', 'java', 'gcc', 'g++', 'make', 'cmake', 'docker', 'kubernetes', 'k8s', 'helm', 'terraform', 'ansible', 'jenkins', 'gitlab', 'github', 'gitlab-ci', 'github-actions', 'docker-compose', 'dockerfile', 'dockerignore', 'requirements.txt', 'setup.py', 'pyproject.toml', 'Makefile', 'Dockerfile', 'docker-compose.yml', 'README.md', 'LICENSE', '.gitignore', 'requirements.txt', 'setup.py', 'pyproject.toml']
    
    action_relevance = 0.0
    if any(keyword in action.lower() for keyword in action_keywords):
        action_relevance = 0.2
    
    if next_state_score > state_score:
        improvement = 0.3
    elif state_score > next_state_score:
        improvement = -0.3
    else:
        improvement = 0.0
    
    base_estimate = 0.5
    state_factor = min(1.0, max(0.0, state_score / 10.0))
    next_state_factor = min(1.0, max(0.0, next_state_score / 10.0))
    
    q_value = base_estimate + state_factor * 0.3 + next_state_factor * 0.3 + action_relevance * 0.2 + improvement * 0.2
    
    q_value = min(1.0, max(0.0, q_value))
    
    return round(q_value, 4)