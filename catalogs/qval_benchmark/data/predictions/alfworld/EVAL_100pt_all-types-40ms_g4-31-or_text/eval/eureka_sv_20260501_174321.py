import re

def signal_function(state: str):
    """
    Estimates the state-value V(s) for ALFWorld tasks.
    
    The signal provides a granular progress gradient based on the achievement of subgoals:
    Success (1.0) > Placed/Cleaned (0.95) > Holding Target at Destination (0.9) > 
    Holding Target (0.8) > Target Spotted Now (0.7) > Target Spotted Ever (0.6) > 
    Container Opened (0.4) > Destination Spotted (0.3) > Exploration (0.1).
    
    Key Improvement:
    - Stricter object matching to avoid false positives (e.g., matching 'shaker' in 'pepper shaker' 
      when the target is 'salt shaker').
    - Granular value tiers to provide a denser training signal.
    """
    state_lower = state.lower()
    
    # 1. Goal Extraction
    # Goal is typically before the first "Observation:"
    goal_section = state.split("Observation:")[0].lower()
    
    target_obj = None
    target_dest = None
    is_cleaning_task = False
    
    # Priority 1: "put X in/on/into Y"
    match_put = re.search(r"put (?:the|a) (.*?) (?:in|on|into) (?:the|a) (.*?)(?:\.|\n|$)", goal_section)
    if match_put:
        target_obj = match_put.group(1).strip()
        target_dest = match_put.group(2).strip()
    else:
        # Priority 2: "clean X"
        match_clean = re.search(r"clean (?:the|a) (.*?)(?:\.|\n|$)", goal_section)
        if match_clean:
            target_obj = match_clean.group(1).strip()
            is_cleaning_task = True
        else:
            # Priority 3: "pick up X" or "take X"
            match_pick = re.search(r"(?:pick up|take) (?:the|a) (.*?)(?:\.|\n|$)", goal_section)
            if match_pick:
                target_obj = match_pick.group(1).strip()

    def get_keywords(text):
        """Extracts identifying keywords for matching."""
        if not text: return []
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]$', '', text)
        text = re.sub(r'^(the|a|an)\s+', '', text)
        stop_words = {"of", "in", "on", "into", "to", "with", "a", "an", "the"}
        words = [w for w in text.split() if w not in stop_words]
        return words

    target_keywords = get_keywords(target_obj)
    dest_keywords = get_keywords(target_dest)
    
    def matches(text, keywords):
        """
        Strict matching logic:
        1. Returns True if the object phrase (keywords joined) is in the text.
        2. Returns True if ALL keywords are present as distinct words (word boundaries).
        This prevents false positives like 'shaker' matching 'pepper shaker' when target is 'salt shaker'.
        """
        if not keywords: return False
        
        # Phrase match
        phrase = " ".join(keywords)
        if phrase in text:
            return True
            
        # All-keywords match
        if all(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in keywords):
            return True
            
        return False

    # 2. Observation Analysis
    obs_parts = state.split("Observation:")
    observations = [p.lower() for p in obs_parts[1:]]
    last_obs = observations[-1] if observations else state_lower
    
    # 3. State Assessment
    
    # Success Condition
    if "task completed" in last_obs or "successfully" in last_obs:
        return 1.0, {"success_val": 1.0}
    
    # Interaction Logic: Placed or Cleaned
    if is_cleaning_task:
        if matches(last_obs, target_keywords) and any(word in last_obs for word in ["clean", "cleaned", "shiny"]):
            return 0.95, {"cleaned_target_val": 0.95}
    else:
        if target_obj and target_dest:
            if matches(last_obs, target_keywords) and matches(last_obs, dest_keywords):
                if any(word in last_obs for word in ["put", "placed", "set", "into"]):
                    return 0.95, {"placed_target_val": 0.95}

    # Inventory Tracking: Holding the target
    is_holding_target = False
    if not is_cleaning_task:
        if "holding" in last_obs and matches(last_obs, target_keywords):
            is_holding_target = True
        else:
            # Check history for pick-up event without subsequent put-down event
            last_pickup_idx = -1
            last_putdown_idx = -1
            for idx, obs in enumerate(observations):
                if ("picked up" in obs or "holding" in obs) and matches(obs, target_keywords):
                    last_pickup_idx = idx
                if ("put" in obs or "placed" in obs) and matches(obs, target_keywords):
                    last_putdown_idx = idx
            if last_pickup_idx > last_putdown_idx:
                is_holding_target = True

    # Feature detection
    target_spotted_now = matches(last_obs, target_keywords)
    target_spotted_ever = any(matches(obs, target_keywords) for obs in observations)
    dest_spotted_now = matches(last_obs, dest_keywords)
    container_opened = any(phrase in last_obs for phrase in ["is open", "opened the", "open the"])

    # 4. Value Gradient
    if is_holding_target:
        if dest_spotted_now:
            return 0.9, {"holding_near_dest_val": 0.9}
        return 0.8, {"holding_target_val": 0.8}

    if target_spotted_now:
        return 0.7, {"target_spotted_now_val": 0.7}
    
    if target_spotted_ever:
        return 0.6, {"target_spotted_ever_val": 0.6}
    
    if container_opened:
        return 0.4, {"container_opened_val": 0.4}
        
    if dest_spotted_now:
        return 0.3, {"near_dest_val": 0.3}
        
    return 0.1, {"exploration_val": 0.1}