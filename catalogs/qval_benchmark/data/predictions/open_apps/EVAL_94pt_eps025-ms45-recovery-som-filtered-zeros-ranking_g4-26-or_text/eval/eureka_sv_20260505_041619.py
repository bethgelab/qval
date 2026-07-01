import re

def signal_function(state: str):
    """
    Estimates the state-value V(s) for an OpenApps agent.
    
    The value function uses a graduated progression:
    - Success Detection: 1.0 (Terminal)
    - Error Detection: 0.0 (Terminal)
    - Context (App Presence): 0.05
    - Task Depth (Form/Navigation): 0.15
    - Interaction (Data Entry): 0.40
    - Readiness (Submit Button): 0.30
    
    Max non-terminal value is 0.9.
    """
    state_lower = state.lower()
    
    # Component initialization
    success_score = 0.0
    context_score = 0.0
    task_depth_score = 0.0
    interaction_score = 0.0
    readiness_score = 0.0

    # 1. Error Detection (Immediate failure)
    error_keywords = ['error', 'failed', 'invalid', 'wrong', 'alert', 'not found', 'not available', 'denied']
    if any(ek in state_lower for ek in error_keywords):
        return 0.0, {
            "success_score": 0.0,
            "context_score": 0.0,
            "task_depth_score": 0.0,
            "interaction_score": 0.0,
            "readiness_score": 0.0
        }

    # 2. Success Detection (Terminal state)
    success_patterns = [
        'successfully', 'is saved', 'was added', 'is scheduled', 
        'task added', 'message sent', 'event created', 'appointment scheduled',
        'destination found', 'completed successfully'
    ]
    if any(sp in state_lower for sp in success_patterns):
        return 1.0, {
            "success_score": 1.0,
            "context_score": 0.0,
            "task_depth_score": 0.0,
            "interaction_score": 0.0,
            "readiness_score": 0.0
        }

    # 3. Context Score (0.05): Presence in a target application.
    app_keywords = ['todo', 'calendar', 'messenger', 'map', 'code', 'editor', 'mail', 'email']
    if any(kw in state_lower for kw in app_keywords):
        context_score = 0.05

    # 4. Task Depth Score (0.15): Navigation to a form, dialog, or task-oriented view.
    form_indicators = [
        'textbox', 'combobox', 'checkbox', 'radio', 'input', 'textarea', 
        'select', 'form', 'dialog', 'compose', 'edit', 'new', 'create'
    ]
    if any(ind in state_lower for ind in form_indicators):
        task_depth_score = 0.15

    # 5. Interaction Score (0.40): Progress through filling out data.
    # We look for non-empty values in 'value' or 'aria-label' attributes.
    interaction_matches = re.findall(r'(?:value|aria-label)=["\']([^"\']+)["\']', state_lower)
    
    noise = {'none', 'undefined', 'null', 'empty', 'select', 'type', 'enter', 'input', 'text', 'false', 'true', 'placeholder'}
    valid_fills = set()
    for m in interaction_matches:
        m_strip = m.strip()
        if m_strip and m_strip.lower() not in noise and len(m_strip) > 1:
            valid_fills.add(m_strip.lower())
            
    if valid_fills:
        # Incrementally award points based on unique non-noise inputs filled
        interaction_score = 0.4 * (min(len(valid_fills), 3) / 3)

    # 6. Readiness Score (0.30): Presence of a terminal action element.
    terminal_keywords = [
        'send', 'save', 'submit', 'confirm', 'schedule', 'post', 
        'update', 'add', 'create', 'done', 'go', 'search'
    ]
    # Check for button roles or button tags combined with terminal semantic keywords.
    has_button_element = 'button' in state_lower or 'role="button"' in state_lower
    has_terminal_text = any(tk in state_lower for tk in terminal_keywords)
    
    if has_button_element and has_terminal_text:
        readiness_score = 0.3

    # Final Calculation
    # Total non-terminal max: 0.05 + 0.15 + 0.40 + 0.30 = 0.90
    total = context_score + task_depth_score + interaction_score + readiness_score
    total = max(0.0, min(0.9, total))

    return total, {
        "success_score": success_score,
        "context_score": context_score,
        "task_depth_score": task_depth_score,
        "interaction_score": interaction_score,
        "readiness_score": readiness_score
    }