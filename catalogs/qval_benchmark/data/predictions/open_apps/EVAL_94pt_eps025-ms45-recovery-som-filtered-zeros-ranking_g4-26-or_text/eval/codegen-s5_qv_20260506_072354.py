import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given transition in the OpenApps environment.
    Uses heuristics based on task completion indicators, action productivity, 
    and state changes in the accessibility tree.
    """
    # 1. Immediate Success Check
    # If the next_state contains words that strongly imply the task is finished.
    success_indicators = [
        'success', 'sent', 'created', 'added', 'saved', 'done', 
        'complete', 'confirmed', 'scheduled', 'recorded', 'message sent',
        'event created', 'appointment scheduled'
    ]
    if any(kw in next_state.lower() for kw in success_indicators):
        return 1.0

    # 2. Parse Action type and target ID
    # Expected formats: click('1'), fill('1', 'text'), press('1', 'key'), scroll(x, y), noop(ms)
    action_type = 'noop'
    target_id = None
    
    if 'click' in action:
        action_type = 'click'
        match = re.search(r"click\(['\"](\d+)['\"]\)", action)
        if match:
            target_id = match.group(1)
    elif 'fill' in action:
        action_type = 'fill'
        match = re.search(r"fill\(['\"](\d+)['\"]", action)
        if match:
            target_id = match.group(1)
    elif 'press' in action:
        action_type = 'press'
        match = re.search(r"press\(['\"](\d+)['\"]", action)
        if match:
            target_id = match.group(1)
    elif 'scroll' in action:
        action_type = 'scroll'

    # 3. Analyze target element in the current state
    # We extract the name or role of the element associated with the target_id.
    target_info = ""
    if target_id:
        # Find the substring belonging to this specific element (from its bid to the next bid)
        bid_pattern = rf'bid=["\']?{target_id}["\']?'
        bid_match = re.search(bid_pattern, state)
        if bid_match:
            start_idx = bid_match.start()
            # Find the next occurrence of a bid to bound the search
            next_bid_pattern = r'bid=["\']?\d+["\']?'
            next_matches = list(re.finditer(next_bid_pattern, state))
            
            end_idx = len(state)
            for nm in next_matches:
                if nm.start() > start_idx:
                    end_idx = nm.start()
                    break
            
            element_context = state[start_idx:end_idx]
            
            # Look for name or role within this element's context
            name_match = re.search(r'name=["\']?([^"\']*)["\']?', element_context)
            if name_match:
                target_info = name_match.group(1).lower()
            else:
                role_match = re.search(r'role=["\']?([^"\']*)["\']?', element_context)
                if role_match:
                    target_info = role_match.group(1).lower()

    # 4. Calculate Q-value based on heuristic scoring
    q_value = 0.0
    
    # Increment for productive actions (non-noop, non-scroll)
    if action_type in ['click', 'fill', 'press']:
        q_value += 0.2
        
        # Significant boost for actions on "finishing" elements (e.g., "Send" button)
        finishing_keywords = ['send', 'submit', 'add', 'save', 'create', 'confirm', 'ok', 'done', 'schedule', 'go']
        if any(kw in target_info for kw in finishing_keywords):
            q_value += 0.5
            
    # Progress check: monitor changes in input-related elements
    def count_inputs(text):
        # Heuristic to count elements that look like input fields
        return len(re.findall(r'role=["\']?(?:input|textbox|edit|text)[^"\']*["\']?', text, re.I))

    state_inputs = count_inputs(state)
    next_inputs = count_inputs(next_state)
    
    if next_inputs < state_inputs:
        # Input fields disappearing often indicates a form was submitted or a step completed
        q_value += 0.3
    elif next_inputs > state_inputs and action_type == 'fill':
        # New inputs appearing after a fill might indicate a multi-step form
        q_value += 0.1
        
    # Minor boost for general state progression (detecting structural change)
    if len(state) > 0 and next_state != state:
        # Only add if not already highly valued
        if q_value < 0.8:
            q_value += 0.1

    # Cap the value at 0.95 (reserve 1.0 for actual task achievement)
    return min(0.95, q_value)