import re

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value Q(s, a) for a TerminalBench task.
    Goal: Maximize expected discounted reward (binary 1.0 on success).
    
    Refined to provide more granular progress signals to distinguish between
    early exploration, active processing, and finalization steps.
    """
    completion_signal = 0.0
    progress_signal = 0.0
    penalty_signal = 0.0

    # 1. High-Confidence Success Markers
    strong_success_keywords = [
        r"all tests passed", r"congratulations", r"goal reached", 
        r"verified successfully", r"correct answer", r"flag\{", r"ctf\{", 
        r"solution found", r"successfully completed", r"verification successful", 
        r"passed all tests", r"mission\s*accomplished", r"correctly\s*computed",
        r"success\s*!\s*$", r"correct\s*!\s*$"
    ]
    moderate_success_keywords = [
        r"passed", r"solved", r"the answer is", r"found the result", 
        r"output produced", r"correctly\s*saved"
    ]
    
    state_is_solved = False
    for pattern in strong_success_keywords + moderate_success_keywords:
        if re.search(pattern, state, re.IGNORECASE):
            state_is_solved = True
            break

    # Immediate success detection in the next state
    if re.search(r"flag\{.*?\}", next_state, re.IGNORECASE):
        completion_signal = 1.0
    else:
        found_strong_success_next = False
        for pattern in strong_success_keywords:
            if re.search(pattern, next_state, re.IGNORECASE):
                completion_signal = 1.0
                found_strong_success_next = True
                break
                
        if not found_strong_success_next:
            for pattern in moderate_success_keywords:
                if re.search(pattern, next_state, re.IGNORECASE):
                    completion_signal = 0.6
                    break

    # 2. Failure and Error Analysis
    failure_keywords = [
        r"assertionerror", r"incorrect", r"wrong answer", 
        r"invalid input", r"access denied", r"failed to", r"error occurred",
        r"no matches found", r"could not find", r"not found", r"failed",
        r"valueerror", r"typeerror", r"indexerror", r"keyerror"
    ]
    technical_errors = [
        r"command not found", r"no such file or directory", r"permission denied", 
        r"syntaxerror", r"traceback", r"invalid option", r"failed with exit code",
        r"segmentation fault", r"core dumped"
    ]

    found_failure = False
    for pattern in failure_keywords:
        if re.search(pattern, next_state, re.IGNORECASE):
            if not re.search(r"(\d+)\s*fails?\s*:\s*0", next_state, re.IGNORECASE):
                penalty_signal -= 0.5
                found_failure = True
                break

    for pattern in technical_errors:
        if re.search(pattern, next_state, re.IGNORECASE):
            penalty_signal -= 0.6
            found_failure = True
            break

    # 3. Verification and Finalization Analysis
    verification_keywords = [r"verify", r"check", r"test", r"submit", r"eval", r"finalize", r"aggregate", r"validate"]
    is_verification_attempt = any(re.search(kw, action.lower()) for kw in verification_keywords)
    if re.search(r"python.*(verify|check|test|submit|eval)\.py", action, re.IGNORECASE):
        is_verification_attempt = True
    
    is_reading_solution = re.search(r"(cat|head|tail|less|more)\s+.*(flag|solution|answer|result|summary|output)\.(txt|csv|json|smiles|fasta)", action, re.IGNORECASE)

    if is_verification_attempt or is_reading_solution:
        if found_failure:
            penalty_signal -= 0.4
        elif is_reading_solution:
            content = next_state.strip()
            if not content:
                penalty_signal -= 0.4
            elif re.search(r'^[\{\[].*[\}\]]\s*$', content, re.DOTALL) or \
                 re.search(r'^[a-zA-Z0-9_]+,.*$', content, re.MULTILINE) or \
                 re.search(r'[CNOSP\[\]\(\)\#\=\.\-\+]{15,}', content) or \
                 re.search(r'^[ACDEFGHIKLMNPQRSTVWY]{15,}', content, re.MULTILINE):
                completion_signal = 0.9
            elif len(content) > 5:
                completion_signal = 0.6
            else:
                penalty_signal -= 0.2
        elif is_verification_attempt:
            # "Silent Success": Verification tool run, no errors, minimal output.
            output_diff = len(next_state) - len(state)
            if output_diff < 300:
                completion_signal = 0.95 
            else:
                progress_signal += 0.2
        else:
            progress_signal += 0.1

    # 4. Granular Progress Tiers
    if not state_is_solved:
        # Tier 4: Finalizing/Producing Output (Highest Progress)
        is_creating_answer = re.search(r"(echo|printf|cat).*>\s*(solution|answer|flag|result)", action, re.IGNORECASE)
        
        # Tier 3: Active Computation/Transformation
        processing_tools = [r"python", r"gcc", r"make", r"grep", r"sed", r"awk", r"perl", r"pip"]
        is_processing = any(re.search(tool, action) for tool in processing_tools)
        
        # Tier 2: Setup/Environment Modification
        setup_tools = [r"vim", r"nano", r"mkdir", r"touch", r"chmod", r"chown"]
        is_setup = any(re.search(tool, action) for tool in setup_tools) or ">" in action
        
        # Tier 1: Pure Exploration
        exploratory_tools = [r"ls", r"cat", r"find", r"head", r"tail", r"pwd", r"whoami"]
        is_exploring = any(re.search(tool, action) for tool in exploratory_tools)

        if is_creating_answer:
            progress_signal += 0.6
        elif is_processing:
            # Higher signal if there's a change in the state (output produced)
            if len(next_state) != len(state):
                progress_signal += 0.35
            else:
                progress_signal += 0.20
        elif is_setup:
            progress_signal += 0.15
        elif is_exploring:
            progress_signal += 0.05

    # 5. Next-State Progress Indicators (Heuristics for "Halfway Done")
    intermediate_markers = [
        (r"processing\s*.*", 0.2),
        (r"loading\s*.*", 0.1),
        (r"found\s*\d+\s*matches", 0.3),
        (r"successfully\s*extracted", 0.3),
        (r"compiled\s*successfully", 0.4),
        (r"finished\s*parsing", 0.3)
    ]
    for pattern, val in intermediate_markers:
        if re.search(pattern, next_state, re.IGNORECASE):
            progress_signal += val

    # 6. Inefficiency and Looping
    action_stripped = action.strip()
    if action_stripped:
        # Check if the same command is repeated and output remains the same
        if action_stripped in state.splitlines()[-3:]:
            if next_state[-200:] == state[-200:]:
                penalty_signal -= 0.6

    # 7. Test Suite Progress
    test_progress_match = re.search(r"(\d+)\s*/\s*(\d+)\s*tests?\s*passed", next_state, re.IGNORECASE)
    if test_progress_match:
        passed = int(test_progress_match.group(1))
        total_tests = int(test_progress_match.group(2))
        if passed == total_tests:
            completion_signal = 1.0
        elif passed > 0:
            # Scales from 0.1 to 0.4 based on completion percentage
            progress_signal += 0.4 * (passed / total_tests)
        else:
            penalty_signal -= 0.2

    # 8. Final Aggregation
    if completion_signal >= 1.0 or state_is_solved:
        total = 1.0
    else:
        total = completion_signal + progress_signal + penalty_signal
        total = max(0.0, min(1.0, total))

    return total, {
        "completion_signal": completion_signal,
        "progress_signal": progress_signal,
        "penalty_signal": penalty_signal,
    }