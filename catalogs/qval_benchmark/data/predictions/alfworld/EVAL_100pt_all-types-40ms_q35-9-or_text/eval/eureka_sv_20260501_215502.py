import re

def signal_function(state: str):
    total = 0.0
    components = {}
    state_lower = state.lower()
    
    # Step remaining: extract actual step number from state
    step_match = re.search(r"step\s*(\d+)", state_lower)
    if step_match:
        step_num = int(step_match.group(1))
        steps_remaining = max(1, 40 - step_num)
        # More aggressive step value: earlier steps = higher, later steps = much lower
        step_value = 0.6 - 0.45 * (steps_remaining / 40.0)
        components["step_remaining"] = step_value
        total += step_value * 0.4
    else:
        # Default based on typical episode progress - assume mid-episode
        step_value = 0.35
        components["step_remaining"] = step_value
        total += step_value * 0.4
    
    # Task completion detection - high confidence patterns
    task_complete = 0.0
    complete_patterns = [
        r"task\s*complete", r"task\s*completed", r"task\s*done", r"task\s*finished",
        r"goal\s*reached", r"goal\s*achieved", r"success", r"completed", r"done",
        r"all\s*done", r"finished", r"successfully", r"complete", r"done\.", r"complete\.",
        r"goal\s*complete", r"task\s*succeed", r"succeeded", r"done\.", r"success\.",
        r"i\s*did\s*it", r"i\s*have\s*completed", r"task\s*is\s*done"
    ]
    for pattern in complete_patterns:
        if re.search(pattern, state_lower):
            task_complete = 1.0
            break
    components["task_complete"] = task_complete
    total += task_complete
    
    # Parse goal object name from state - more specific patterns
    goal_object = ""
    object_patterns = [
        r"pick\s*(?:up|out|the)?\s*(\w+)",
        r"take\s*(?:up|out|the)?\s*(\w+)",
        r"move\s*(?:the)?\s*(\w+)",
        r"clean\s*(?:the)?\s*(\w+)",
        r"wash\s*(?:the)?\s*(\w+)",
        r"open\s*(?:the)?\s*(\w+)",
        r"fold\s*(?:the)?\s*(\w+)",
        r"put\s*(?:the)?\s*(\w+)",
        r"remove\s*(?:the)?\s*(\w+)",
        r"place\s*(?:the)?\s*(\w+)",
        r"object\s*is\s*(\w+)", r"the\s*(\w+)",
        r"found\s*(?:the)?\s*(\w+)", r"located\s*(?:the)?\s*(\w+)",
        r"clean\s*the\s*(\w+)", r"fold\s*the\s*(\w+)", r"wash\s*the\s*(\w+)",
    ]
    for pattern in object_patterns:
        match = re.search(pattern, state_lower)
        if match and match.group(1):
            goal_object = match.group(1).lower()
            break
    
    # Goal object should only be 1.0 if we're confident it's found, not just mentioned
    goal_object_val = 0.0
    if goal_object:
        # Check if object is actually in state as found/located
        if re.search(rf"found\s*{goal_object}|located\s*{goal_object}|at\s*location\s*{goal_object}", state_lower):
            goal_object_val = 0.95
        elif re.search(rf"near\s*{goal_object}|next\s*to\s*{goal_object}", state_lower):
            goal_object_val = 0.6
        elif re.search(rf"holding\s*{goal_object}|carrying\s*{goal_object}", state_lower):
            goal_object_val = 0.8
        else:
            goal_object_val = 0.3
    components["goal_object"] = goal_object_val
    total += goal_object_val * 0.25
    
    # Object at target location - more specific patterns
    at_target = 0.0
    target_patterns = [
        r"on\s*table", r"on\s*desk", r"on\s*shelf", r"on\s*floor",
        r"in\s*box", r"in\s*drawer", r"in\s*cupboard", r"in\s*room",
        r"at\s*target", r"at\s*goal", r"placed\s*on", r"placed\s*at",
        r"placed\s*in", r"dropped\s*on", r"dropped\s*at", r"put\s*on",
        r"put\s*at", r"put\s*in", r"in\s*target", r"in\s*destination",
        r"goal\s*location", r"target\s*location", r"destination", r"placed\s*at\s*goal",
        r"on\s*the\s*goal", r"in\s*the\s*goal", r"at\s*the\s*goal"
    ]
    for pattern in target_patterns:
        if re.search(pattern, state_lower):
            at_target = 0.9
            break
    components["at_target"] = at_target
    total += at_target * 0.3
    
    # Object in hand/carrying - more specific patterns
    goal_in_hand = 0.0
    hand_patterns = [
        r"holding\s*(?:the)?\s*(\w+)", r"carrying\s*(?:the)?\s*(\w+)",
        r"picked\s*up\s*(?:the)?\s*(\w+)", r"grabbed\s*(?:the)?\s*(\w+)",
        r"taken\s*(?:the)?\s*(\w+)", r"in\s*hand", r"holding\s*the",
        r"carrying\s*the", r"picked\s*up\s*the", r"grasping\s*(?:the)?\s*(\w+)",
        r"has\s*picked", r"is\s*holding", r"in\s*my\s*hand", r"carrying\s*it",
        r"holding\s*it", r"carrying\s*it", r"holding\s*the\s*goal",
        r"carrying\s*the\s*goal"
    ]
    for pattern in hand_patterns:
        if re.search(pattern, state_lower):
            goal_in_hand = 0.85
            break
    components["goal_in_hand"] = goal_in_hand
    total += goal_in_hand * 0.2
    
    # Object found/located in state (before being picked up)
    goal_found = 0.0
    if task_complete == 0 and at_target == 0 and goal_in_hand == 0:
        found_patterns = [
            r"found\s*(?:the)?\s*(\w+)", r"located\s*(?:the)?\s*(\w+)",
            r"at\s*location", r"near\s*goal", r"next\s*to", r"beside",
            r"goal\s*is", r"target\s*is", r"object\s*is", r"the\s*(\w+)",
            r"discovered", r"see", r"visible", r"here", r"can\s*see",
            r"i\s*see", r"i\s*can\s*see", r"i\s*found", r"i\s*see",
            r"the\s*object\s*is", r"found\s*it", r"located\s*it"
        ]
        for pattern in found_patterns:
            if re.search(pattern, state_lower):
                goal_found = 0.45
                break
    components["goal_found"] = goal_found
    total += goal_found * 0.15
    
    # Object in correct state
    correct_state = 0.0
    state_patterns = [
        r"dry", r"folded", r"open", r"cleaned", r"washed", r"ready",
        r"clean", r"fold", r"opened", r"washed", r"prepared",
        r"correct\s*state", r"proper\s*state", r"clean\s*enough",
        r"properly\s*folded", r"properly\s*cleaned", r"properly\s*dry",
        r"properly\s*opened", r"properly\s*washed", r"properly\s*cleaned"
    ]
    for pattern in state_patterns:
        if re.search(pattern, state_lower):
            correct_state = 0.35
            break
    components["correct_state"] = correct_state
    total += correct_state * 0.1
    
    # Agent location relative to goal
    agent_loc = 0.0
    loc_patterns = [
        r"at\s*goal", r"at\s*target", r"at\s*destination", r"near\s*goal",
        r"near\s*target", r"in\s*front\s*of", r"facing", r"standing\s*at",
        r"moving\s*to", r"approaching", r"at\s*the", r"nearby",
        r"i\s*am\s*at", r"i\s*am\s*near", r"currently\s*at", r"in\s*room",
        r"in\s*living\s*room", r"in\s*bedroom", r"in\s*kitchen", r"in\s*hallway",
        r"in\s*office", r"in\s*bathroom", r"in\s*garage", r"in\s*storage",
        r"i\s*am\s*in", r"currently\s*in", r"i\s*am\s*near", r"positioned\s*at",
        r"located\s*at", r"standing\s*near"
    ]
    for pattern in loc_patterns:
        if re.search(pattern, state_lower):
            agent_loc = 0.25
            break
    components["agent_loc"] = agent_loc
    total += agent_loc * 0.1
    
    # Negative states
    negative_state = 0.0
    neg_patterns = [
        r"dirty", r"wet", r"broken", r"closed", r"stuck", r"blocked",
        r"waiting", r"blocked\s*by", r"cannot\s*reach", r"impossible",
        r"unreachable", r"too\s*far", r"too\s*heavy", r"too\s*large",
        r"too\s*many", r"unable", r"not\s*able", r"not\s*possible"
    ]
    for pattern in neg_patterns:
        if re.search(pattern, state_lower):
            negative_state = 0.25
            break
    components["negative_state"] = -negative_state
    total += -negative_state
    
    # Failure/error indicators
    failure_penalty = 0.0
    fail_patterns = [
        r"fail", r"error", r"failed", r"impossible", r"failed\s*to",
        r"unreachable", r"not\s*able", r"unsuccessful", r"give\s*up",
        r"quit", r"abandon", r"i\s*cannot", r"i\s*failed", r"i\s*can't",
        r"i\s*am\s*unable", r"unable\s*to", r"failed\.", r"error\.",
        r"i\s*give\s*up", r"i\s*quit", r"i\s*abandon"
    ]
    for pattern in fail_patterns:
        if re.search(pattern, state_lower):
            failure_penalty = 0.6
            break
    components["failure_penalty"] = -failure_penalty
    total += -failure_penalty
    
    # Progress indicators
    progress = 0.0
    prog_patterns = [
        r"task\s*in\s*progress", r"working\s*on", r"in\s*progress",
        r"processing", r"attempting", r"trying\s*to", r"plan\s*to",
        r"planning", r"need\s*to", r"should", r"must", r"have\s*to",
        r"want\s*to", r"i\s*need", r"i\s*should", r"i\s*must",
        r"currently\s*working", r"working\s*on\s*the"
    ]
    for pattern in prog_patterns:
        if re.search(pattern, state_lower):
            progress = 0.18
            break
    components["progress"] = progress
    total += progress
    
    # Object location relative to agent
    object_loc = 0.0
    if goal_object:
        # Check if object is nearby or accessible
        if re.search(rf"{goal_object}\s*at\s*(?:table|desk|shelf|floor|box|drawer)", state_lower):
            object_loc = 0.18
            components["object_loc"] = object_loc
            total += object_loc
        elif re.search(rf"holding\s*{goal_object}|carrying\s*{goal_object}", state_lower):
            object_loc = 0.18
            components["object_loc"] = object_loc
            total += object_loc
    
    # Combined intermediate state bonus
    if goal_found > 0.1 and task_complete == 0 and at_target == 0 and goal_in_hand == 0:
        intermediate_bonus = 0.1
        components["intermediate_bonus"] = intermediate_bonus
        total += intermediate_bonus
    
    # Object in hand AND being at/near target location
    if goal_in_hand > 0.3 and (agent_loc > 0.1 or at_target > 0.1):
        hand_at_target_bonus = 0.15
        components["hand_at_target_bonus"] = hand_at_target_bonus
        total += hand_at_target_bonus
    
    # Correct state with object in hand
    if correct_state > 0.1 and goal_in_hand > 0.3:
        correct_state_hand_bonus = 0.12
        components["correct_state_hand_bonus"] = correct_state_hand_bonus
        total += correct_state_hand_bonus
    
    # Early episode bonus - more nuanced based on actual step value
    if step_value > 0.4:
        early_bonus = 0.08
        components["early_bonus"] = early_bonus
        total += early_bonus
    
    # Time pressure penalty for late steps - more aggressive
    if step_value < 0.35:
        time_penalty = 0.1
        components["time_pressure"] = -time_penalty
        total += -time_penalty
    
    # Low goal object found penalty for late stages
    if goal_object_val < 0.3 and step_value < 0.4:
        low_progress_penalty = 0.08
        components["low_progress_penalty"] = -low_progress_penalty
        total += -low_progress_penalty
    
    # No goal object found penalty
    if goal_object_val == 0.0 and step_value < 0.45:
        no_object_penalty = 0.05
        components["no_object_penalty"] = -no_object_penalty
        total += -no_object_penalty
    
    # Clamping and final adjustments
    total = max(0.0, min(1.0, total))
    
    return total, components