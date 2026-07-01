def signal_function(state: str, action: str, next_state: str):
    # Q-value estimation for terminal RL tasks - improved calibration
    
    # Success indicators - definitive completion signals
    success_patterns = [
        "verified", "passed", "success", "complete", "done",
        "all tests passed", "verification passed", "task complete",
        "all checks passed", "solution verified", "output correct",
        "correct", "accepted", "success!", "ok", "all good",
        "✓", "✔", "PASS", "PASSED", "OK", "COMPLETE",
        "exit code 0", "return code 0", "no errors", "no failures",
        "test passed", "build successful", "installation complete",
        "all tests passed successfully", "verification successful",
        "task completed", "solution verified", "done!",
        "verification successful", "tests passed", "all passed",
        "✓ all tests passed", "✓ verification passed",
        "solution accepted", "task verified", "✓", "✔✓", "success ✓", "✓ done"
    ]
    
    # Error indicators - negative signal for wrong paths
    error_patterns = [
        "error", "failed", "exception", "traceback", "not found",
        "permission denied", "syntax error", "invalid", "missing",
        "cannot", "unable to", "failed to", "no such", "undefined",
        "refused", "timeout", "connection refused", "denied",
        "command not found", "access denied", "forbidden",
        "segfault", "core dumped", "killed", "terminated",
        "module not found", "package not found", "file not found",
        "cannot open", "failed to open", "unable to open",
        "error occurred", "error while", "error in"
    ]
    
    # Critical error patterns - severe penalties
    critical_errors = [
        "segfault", "core dumped", "killed", "terminated", "permission denied",
        "connection refused", "access denied", "forbidden"
    ]
    
    # Moderate error patterns
    moderate_errors = [
        "error", "failed", "exception", "traceback", "syntax error",
        "invalid", "missing", "cannot", "unable to", "failed to",
        "no such", "undefined", "refused", "timeout"
    ]
    
    # Minor error patterns
    minor_errors = [
        "not found", "denied", "command not found", "error occurred",
        "error while", "error in"
    ]
    
    # Progress indicators - active work in progress
    progress_patterns = [
        "step", "stage", "progress", "attempt", "iteration",
        "working on", "processing", "generating", "creating",
        "writing", "building", "compiling", "running",
        "executing", "testing", "validating", "checking",
        "loading", "reading", "parsing", "analyzing", "solving",
        "installing", "configuring", "setting up", "preparing"
    ]
    
    # Task-specific patterns for different domains
    task_patterns = {
        "system_admin": ["chmod", "chown", "apt", "yum", "service", "systemctl", "user", "group", "mount", "disk", "sudo", "ssh"],
        "cryptography": ["openssl", "gpg", "encrypt", "decrypt", "hash", "sha", "md5", "aes", "rsa", "key", "cert", "ssl", "tls", "cipher"],
        "ml": ["model", "train", "predict", "accuracy", "loss", "epoch", "batch", "tensor", "numpy", "pandas", "sklearn", "tensorflow", "pytorch", "keras"],
        "data_processing": ["csv", "json", "xml", "parse", "extract", "transform", "filter", "sort", "merge", "join", "sql", "sqlite", "database", "pandas"]
    }
    
    # Task completion patterns - improved detection
    task_completion_patterns = {
        "system_admin": ["service started", "service running", "user created", "group created", "mount point", "disk formatted", "ssh connected", "connection established", "service restarted", "service stopped", "config updated", "permission set"],
        "cryptography": ["encrypted", "decrypted", "hash computed", "key generated", "certificate", "cipher", "signature", "verification", "decryption successful", "encryption successful", "hash generated", "key pair created"],
        "ml": ["model trained", "accuracy", "loss converged", "prediction", "epoch completed", "inference", "training completed", "validation passed", "test passed", "model saved", "weights loaded"],
        "data_processing": ["data loaded", "csv parsed", "json extracted", "transformed", "merged", "filtered", "data processed", "file written", "output generated", "data validated", "records inserted"]
    }
    
    # Calculate state content
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    combined_state = next_state_lower + state_lower
    action_lower = action.lower()
    
    # Calculate progress reward (0.0 to 0.85)
    progress_reward = 0.0
    
    # Check for definitive completion - highest reward
    for pattern in success_patterns:
        if pattern.lower() in next_state_lower or pattern.lower() in state_lower:
            progress_reward = 0.85
            break
    
    # If not definitive completion, check for active progress - medium reward
    if progress_reward < 0.5:
        for pattern in progress_patterns:
            if pattern.lower() in next_state_lower:
                progress_reward = 0.45
                break
    
    # Check for task-specific progress - lower reward
    if progress_reward < 0.35:
        for task_name, patterns in task_patterns.items():
            for pattern in patterns:
                if pattern.lower() in next_state_lower:
                    progress_reward = max(progress_reward, 0.3)
                    break
    
    # Check for partial progress indicators - very low reward
    if progress_reward < 0.2:
        partial_patterns = ["try", "attempt", "running", "loading", "starting", "initializing"]
        for pattern in partial_patterns:
            if pattern.lower() in next_state_lower:
                progress_reward = max(progress_reward, 0.18)
                break
    
    # Calculate error penalty with improved variance (-0.8 to 0)
    error_penalty = 0.0
    error_count = 0
    critical_count = 0
    moderate_count = 0
    minor_count = 0
    
    # Count different severity errors
    for pattern in critical_errors:
        if pattern in next_state_lower:
            critical_count += 1
            error_count += 1
    
    for pattern in moderate_errors:
        if pattern in next_state_lower:
            moderate_count += 1
            error_count += 1
    
    for pattern in minor_errors:
        if pattern in next_state_lower:
            minor_count += 1
            error_count += 1
    
    # Severity-based penalty with better calibration
    if critical_count >= 3:
        error_penalty = -0.8  # Critical failure
    elif critical_count >= 2:
        error_penalty = -0.7  # Major critical failure
    elif critical_count >= 1:
        error_penalty = -0.55  # Single critical error
    elif error_count >= 6:
        error_penalty = -0.65  # Multiple moderate/minor errors
    elif error_count >= 5:
        error_penalty = -0.55  # Significant error count
    elif error_count >= 4:
        error_penalty = -0.45  # Moderate failure
    elif error_count >= 3:
        error_penalty = -0.35  # Minor failure
    elif error_count >= 2:
        error_penalty = -0.2  # Single error
    elif error_count >= 1:
        error_penalty = -0.12  # Minor issue
    
    # Calculate action quality bonus (0.0 to 0.35)
    action_bonus = 0.0
    action_lower = action.lower()
    
    # High-value actions
    high_value_actions = ["python", "pip", "npm", "git", "ssh", "curl", "wget",
                          "vim", "nano", "sed", "awk", "grep", "find", "cat",
                          "mkdir", "touch", "chmod", "chown", "apt", "yum"]
    
    # Medium-value actions
    medium_value_actions = ["ls", "cd", "pwd", "echo", "head", "tail", "less", "more",
                            "wc", "sort", "uniq", "cut", "paste", "diff", "cmp"]
    
    # Low-value or potentially problematic actions
    low_value_actions = ["rm", "rm -rf", "sudo", "su", "bash", "sh", "exit", "kill"]
    
    for prod_action in high_value_actions:
        if prod_action in action_lower:
            action_bonus = 0.35
            break
    if action_bonus == 0.0:
        for prod_action in medium_value_actions:
            if prod_action in action_lower:
                action_bonus = 0.25
                break
    if action_bonus == 0.0:
        for prod_action in low_value_actions:
            if prod_action in action_lower:
                action_bonus = 0.12
                break
    
    # Penalty for dangerous mistakes
    mistake_patterns = ["sudo su", "rm -rf /", "rm -rf *", "chmod 777", "chmod 666",
                        "rm -rf home", "rm -rf root", "rm -rf /tmp", "rm -rf var"]
    for mistake in mistake_patterns:
        if mistake in action_lower:
            action_bonus -= 0.3
    
    # Task completion bonus - improved detection
    completion_bonus = 0.0
    if progress_reward >= 0.75:
        completion_bonus = 0.5
    elif progress_reward >= 0.45:
        completion_bonus = 0.4
    elif progress_reward >= 0.3:
        completion_bonus = 0.3
    elif progress_reward >= 0.18:
        completion_bonus = 0.2
    
    # Task completion bonus - improved detection
    task_completion_bonus = 0.0
    for task_name, patterns in task_completion_patterns.items():
        for pattern in patterns:
            if pattern.lower() in next_state_lower:
                task_completion_bonus = max(task_completion_bonus, 0.25)
                break
    
    # Efficiency bonus based on progress and completion (0.0 to 0.45)
    efficiency_bonus = 0.25
    if progress_reward >= 0.75:
        efficiency_bonus = 0.45
    elif progress_reward >= 0.45:
        efficiency_bonus = 0.4
    elif progress_reward >= 0.3:
        efficiency_bonus = 0.35
    elif progress_reward >= 0.18:
        efficiency_bonus = 0.3
    elif error_penalty < -0.5:
        efficiency_bonus = 0.15
    else:
        efficiency_bonus = 0.25
    
    # Penalty for terminal errors (connection/session issues)
    terminal_error_penalty = 0.0
    terminal_error_terms = ["terminal", "tty", "pty", "session", "connection", "ssh"]
    for term_error in terminal_error_terms:
        if term_error in next_state_lower and ("error" in next_state_lower or "failed" in next_state_lower):
            terminal_error_penalty = -0.3
            break
    
    # Bonus for clean terminal state
    clean_terminal_bonus = 0.0
    clean_terms = ["root@", "user@", "bash", "zsh", "sh", "command"]
    for term in clean_terms:
        if term in next_state_lower and "error" not in next_state_lower and "failed" not in next_state_lower:
            clean_terminal_bonus = 0.15
            break
    
    # Bonus for successful verification indicators
    verification_bonus = 0.0
    verification_terms = ["verification", "verified", "test", "check", "pass", "accept", "complete", "success", "✓", "✔"]
    for ver_term in verification_terms:
        if ver_term.lower() in next_state_lower:
            verification_bonus = 0.22
            break
    
    # Distance to completion heuristic based on state patterns
    distance_penalty = 0.0
    if error_penalty < -0.5:
        distance_penalty = -0.25
    elif error_penalty < -0.3:
        distance_penalty = -0.15
    elif error_penalty < -0.1:
        distance_penalty = -0.08
    elif progress_reward < 0.2:
        distance_penalty = -0.05
    
    # Total Q-value - calibrated for binary reward task
    total_q = progress_reward + error_penalty + action_bonus + completion_bonus + efficiency_bonus + terminal_error_penalty + clean_terminal_bonus + verification_bonus + task_completion_bonus + distance_penalty
    
    return total_q, {
        "progress_reward": progress_reward,
        "error_penalty": error_penalty,
        "action_bonus": action_bonus,
        "completion_bonus": completion_bonus,
        "efficiency_bonus": efficiency_bonus,
        "terminal_error_penalty": terminal_error_penalty,
        "clean_terminal_bonus": clean_terminal_bonus,
        "verification_bonus": verification_bonus,
        "task_completion_bonus": task_completion_bonus,
        "distance_penalty": distance_penalty,
    }