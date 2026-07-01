import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value by analyzing the accessibility tree for markers of 
    task progress, interaction readiness, and goal completion.
    """
    # 1. Immediate Goal Achievement
    # If keywords indicating the task is done are present, the value is maximum.
    if re.search(r'\b(success|completed|done|sent|saved|created|added|scheduled|confirmed|finished)\b', state, re.I):
        return 1.0

    # 2. Feature Extraction
    # Identifying potential action elements (buttons, links) via common action verbs.
    action_words = r'send|submit|save|create|add|confirm|update|post|delete|edit|ok|yes|go|search|apply|schedule'
    has_action_word = re.search(rf'\b({action_words})\b', state, re.I)

    # Identifying input-oriented elements (textboxes, fields).
    input_roles = r'input|text|textarea|edit|textbox|search|field|email|subject|body|title|date|time'
    has_input_role = re.search(rf'\b({input_roles})\b', state, re.I)

    # Extracting quoted strings to determine if the user has already entered information.
    # In accessibility trees, quoted strings often represent the 'name' or 'value' attribute.
    quoted_values = re.findall(r'["\']([^"\']+)["\']', state)
    
    # Heuristic to detect "filled" inputs.
    # We consider an input filled if there's a non-trivial string that isn't a common UI command.
    has_filled_content = False
    if has_input_role and quoted_values:
        for v in quoted_values:
            v_s = v.strip()
            if len(v_s) > 1 and v_s.lower() not in ['send', 'submit', 'save', 'ok', 'yes', 'edit', 'add', 'search', 'confirm', 'done']:
                has_filled_content = True
                break

    # 3. Heuristic Scoring (Progress-based)
    # High value: We have provided information and an action (like 'Submit') is available.
    if has_action_word and has_filled_content:
        return 0.85
    
    # Medium-high value: We have provided info but are still in a data-entry phase.
    if has_filled_content:
        return 0.5
    
    # Medium value: An action button is visible, suggesting we can transition to the next phase.
    if has_action_word:
        return 0.35
    
    # Medium-low value: We are looking at input fields, indicating a task has started.
    if has_input_role:
        return 0.2
    
    # Low value: We are on a relevant application page but haven't started the specific task.
    if re.search(r'\b(home|inbox|calendar|todo|messenger|maps|code|editor|dashboard|menu|settings)\b', state, re.I):
        return 0.1
        
    return 0.0