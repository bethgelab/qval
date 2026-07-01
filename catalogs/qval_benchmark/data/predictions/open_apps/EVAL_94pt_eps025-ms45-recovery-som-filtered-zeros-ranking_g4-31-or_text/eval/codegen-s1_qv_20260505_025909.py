import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) based on the transition from state to next_state.
    The reward is binary (1.0 for goal achievement). Q-value approximates the 
    likelihood and proximity of reaching the goal.
    """
    # 1. Check for Goal Achievement (Terminal State)
    # Success markers common across the synthetic apps (Todo, Calendar, Messenger, Maps, Code Editor)
    success_keywords = [
        "successfully", "created", "added", "sent", "saved", 
        "completed", "confirmed", "reached", "executed"
    ]
    
    # Heuristic: If the next_state contains a success marker and the state didn't (or the action was a submit)
    # we are likely at the goal.
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    is_success = any(kw in next_state_lower for kw in success_keywords)
    # To avoid false positives (e.g., "Success" appearing in a list), we check if it's a new addition
    # or associated with a submission action.
    if is_success:
        # If it's a confirmation and we just clicked a button like 'Save' or 'Send'
        if any(word in action.lower() for word in ["click", "press"]):
            return 1.0
        # If the success indicator appeared and wasn't there before
        if not any(kw in state_lower for kw in success_keywords):
            return 1.0

    # 2. Analyze Action Progress
    # Base value for taking any action
    q_val = 0.2
    
    # Penalty for non-productive actions
    if "noop" in action:
        return 0.0
    if "scroll" in action:
        # Scrolling is only useful if it reveals new interactive elements (represented by bid tags)
        # We approximate this by comparing the number of bids (digits in quotes)
        bids_state = re.findall(r"'\d+'", state)
        bids_next = re.findall(r"'\d+'", next_state)
        if len(bids_next) > len(bids_state):
            q_val = 0.3
        else:
            q_val = 0.1
        return q_val

    # 3. State Transition Analysis (Progress toward goal)
    # Moving from a list/dashboard to a creation form
    form_indicators = ["input", "textarea", "placeholder", "fill", "type"]
    state_has_form = any(ind in state_lower for ind in form_indicators)
    next_has_form = any(ind in next_state_lower for ind in form_indicators)
    
    if not state_has_form and next_has_form:
        # Progress: Opened a creation/edit form
        q_val += 0.3
    elif state_has_form and not next_has_form:
        # Progress: Submitted a form or navigated away (potentially to a success page)
        q_val += 0.4
    elif state_has_form and next_has_form:
        # Staying in the form: filling it out is progress
        if "fill" in action:
            q_val += 0.3
        elif "click" in action:
            # Clicking 'Save', 'Submit', 'Add', 'Send' inside a form
            submit_keywords = ["save", "submit", "add", "send", "create", "confirm"]
            # We check if the transition looks like a submission (form disappears or success appears)
            if any(kw in action.lower() for kw in submit_keywords):
                q_val += 0.4
            else:
                q_val += 0.1

    # 4. Identifying high-value targets in the Action
    # If the action target (bid) corresponds to a likely goal-advancing button
    # Note: We can't see the labels in the action string directly, but we can look at common patterns.
    if "click" in action:
        # If the transition results in a significant change in the DOM tree length/content
        if abs(len(next_state) - len(state)) > 500:
            q_val += 0.1

    # Cap the value between 0 and 1
    return max(0.0, min(1.0, q_val))