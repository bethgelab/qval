import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given ALFWorld state.
    The value is an approximation of the probability of reaching the goal.
    """
    s = state.lower()

    # 1. Check for immediate terminal success
    # ALFWorld typically signals completion with success-related keywords.
    if any(kw in s for kw in ["success", "task complete", "goal achieved", "finished"]):
        return 1.0

    # 2. Extract the task from the state string
    # Tasks in ALFWorld are often preceded by "Task:"
    task_match = re.search(r"task:\s*([^.]+)", s)
    if not task_match:
        # If task is not explicitly labeled, return a baseline low value
        return 0.1
    task = task_match.group(1).strip()

    # 3. Identify the main entities (object and target location/state)
    # We look for nouns following articles (the, a, an) as they are most likely the objects.
    entities = re.findall(r"(?:the|a|an)\s+([a-z]+)", task)
    
    # Fallback: If no articles are used, use non-stopword words
    if len(entities) < 1:
        stopwords = {'put', 'move', 'place', 'clean', 'get', 'find', 'the', 'a', 'an', 'in', 'to', 'on', 'into', 'with'}
        entities = [w for w in task.split() if w not in stopwords]

    if not entities:
        return 0.1

    # Primary object is the first entity, target is the second (if exists)
    obj = entities[0]
    target = entities[1] if len(entities) > 1 else None

    # 4. Analyze the relationship and presence of entities in the current state
    obj_present = obj in s
    target_present = (target in s) if target else False

    satisfied = False
    if target and obj_present and target_present:
        # Case A: Multi-entity task (e.g., "put the apple in the fridge")
        # Check if the object and target appear together in a satisfying way.
        # Patterns: "object is in target", "target contains object", etc.
        pattern_obj_to_target = rf"{obj}.*?\b(?:is|was|in|on|at|inside|placed|put|on)\b.*?\b{target}\b"
        pattern_target_to_obj = rf"{target}.*?\b(?:has|contains|holds|is|was|with)\b.*?\b{obj}\b"
        if re.search(pattern_obj_to_target, s) or re.search(pattern_target_to_obj, s):
            satisfied = True
    elif not target and obj_present:
        # Case B: Single-entity task (e.g., "clean the apple")
        # We check if the object is currently in the state described by the action.
        task_words = set(re.findall(r"\w+", task))
        action_words = {'put', 'move', 'place', 'clean', 'get', 'find', 'the', 'a', 'an', 'in', 'to', 'on', 'into', 'with'}
        # Potential descriptive words from the task
        descriptors = task_words - action_words
        for desc in descriptors:
            # If the object and a descriptive word (like 'clean') appear near each other
            if re.search(rf"{obj}.*?\b{desc}\b", s) or re.search(rf"\b{desc}\b.*?{obj}", s):
                satisfied = True
                break

    # 5. Rank the state value based on findings
    if satisfied:
        return 1.0
    
    if obj_present and target_present:
        # Both the object and the location are visible, but the task isn't finished.
        return 0.6
    elif obj_present:
        # The object is in the current room/view, but the target is not (or not satisfied).
        return 0.3
    elif target_present:
        # The destination is visible, but the object hasn't been found yet.
        return 0.1
    
    # Baseline value for when nothing relevant is seen.
    return 0.05