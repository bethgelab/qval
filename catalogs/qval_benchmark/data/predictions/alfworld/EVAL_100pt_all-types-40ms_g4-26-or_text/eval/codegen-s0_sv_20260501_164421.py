import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given state in the ALFWorld environment.
    The value is based on progress towards a goal extracted from the state.
    """
    s_low = state.lower()

    # 1. Check for immediate success via terminal indicators
    success_indicators = ["task complete", "goal reached", "success!", "task successful", "you have completed"]
    if any(indicator in s_low for indicator in success_indicators):
        return 1.0

    # 2. Extract the goal from the state
    # ALFWorld observations typically include the goal as "Goal: ..." or "Task: ..."
    goal = ""
    goal_match = re.search(r"(?:goal|task)[:\s]+(.*?)(?:\.|\n|$)", state, re.IGNORECASE)
    if goal_match:
        goal = goal_match.group(1).strip().lower()
    
    if not goal:
        # If no goal is explicitly found, we return a baseline value.
        return 0.0

    # Helper to strip common articles to improve matching robustness
    def clean_term(term: str) -> str:
        return re.sub(r'^(the|a|an)\s+', '', term.strip())

    # 3. Task-specific heuristic analysis
    
    # Case A: "Put/Move/Place [object] in/on/to/inside [location]"
    # Matches: "put the apple in the fridge", "move apple to table", "place apple inside the box"
    match_a = re.search(r"(?:put|move|place)\s+(?:the\s+)?(.*?)\s+(?:in|on|to|into|inside|at)\s+(?:the\s+)?(.*)", goal)
    if match_a:
        obj_raw, loc_raw = match_a.groups()
        obj = clean_term(obj_raw)
        loc = clean_term(loc_raw)

        # Check if the object is already in/on the location within the same sentence
        # We use [^.]*? to ensure the relationship is described within one sentence
        pattern_obj_in_loc = rf"{obj}[^.]*?\s+(?:in|on|inside|at|to)\s+(?:the\s+)?{loc}"
        pattern_loc_has_obj = rf"{loc}[^.]*?\s+(?:contains|has|holds|is\s+holding)\s+(?:an?\s+)?{obj}"
        
        if re.search(pattern_obj_in_loc, s_low) or re.search(pattern_loc_has_obj, s_low):
            return 1.0
        
        # Progress markers: Presence of both, or just the object
        obj_visible = obj in s_low
        loc_visible = loc in s_low
        
        if obj_visible and loc_visible:
            return 0.6
        elif obj_visible:
            return 0.4
        elif loc_visible:
            return 0.3
        else:
            return 0.1

    # Case B: "Find [object]" or "Locate [object]"
    match_b = re.search(r"(?:find|locate)\s+(?:the\s+)?(.*)", goal)
    if match_b:
        obj = clean_term(match_b.group(1))
        if obj in s_low:
            return 0.8
        return 0.1

    # Case C: "Clean [object]"
    match_c = re.search(r"(?:clean|wash)\s+(?:the\s+)?(.*)", goal)
    if match_c:
        obj = clean_term(match_c.group(1))
        if obj in s_low:
            # If the word 'clean' appears in the same sentence as the object
            # or 'dirty' does not appear, we assume progress/completion
            if re.search(rf"{obj}[^.]*?clean", s_low) or "dirty" not in s_low:
                return 0.9
            return 0.5
        return 0.1

    # Default fallback for unrecognized goal structures
    return 0.1