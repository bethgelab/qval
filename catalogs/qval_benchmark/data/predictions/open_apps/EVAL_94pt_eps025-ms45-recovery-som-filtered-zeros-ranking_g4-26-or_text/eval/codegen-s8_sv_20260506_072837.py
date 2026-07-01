import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps agent by analyzing the accessibility tree text.
    The function uses heuristics to determine if the agent is navigating (low value),
    performing a task (medium value), or nearing/at completion (high value).
    """
    # Convert to lowercase for case-insensitive analysis
    s = state.lower()
    
    # 1. Terminal State Detection (Outcome-based reward check)
    # If the state indicates the task is already completed, return a high value.
    success_keywords = ['success', 'completed', 'sent', 'added', 'saved', 'created', 'done', 'finished', 'confirmed']
    if any(word in s for word in success_keywords):
        return 0.98
        
    # If the state indicates an error or failure, return a low value.
    error_keywords = ['error', 'failed', 'invalid', 'denied', 'not found', 'could not', 'unauthorized']
    if any(word in s for word in error_keywords):
        return 0.05

    # 2. Feature Extraction (Analyzing the accessibility tree structure)
    # Check for presence of interactive elements crucial for task completion
    # We look for common roles and HTML elements found in web accessibility trees.
    has_input = bool(re.search(r'input|textarea|role="textbox"|contenteditable', s))
    has_button = bool(re.search(r'button|role="button"', s))
    
    # Task-oriented verbs often indicate the agent is within a functional context
    task_verbs = ['new', 'create', 'add', 'edit', 'compose', 'send', 'save', 'search', 'modify', 'delete']
    has_task_verb = any(verb in s for verb in task_verbs)
    
    # Navigation indicators (menus, link-heavy pages) suggest early-stage exploration
    has_navigation = bool(re.search(r'link|role="link"|menu|nav|breadcrumb', s))

    # 3. Heuristic Value Estimation
    # Base value represents being inside the environment/app
    value = 0.15
    
    # Proximity to task: input fields and buttons suggest being in the "action" phase
    if has_input:
        value += 0.25
    if has_button:
        value += 0.25
    
    # Bonus: A combination of inputs and buttons strongly suggests a form-based task
    if has_input and has_button:
        value += 0.15
        
    # Contextual bonus: Presence of task-specific verbs increases confidence of task progress
    if has_task_verb:
        value += 0.15
        
    # Penalty: If the state is primarily navigational with no interactive task elements,
    # it is likely a starting/menu state, which has lower expected future reward.
    if has_navigation and not (has_input or has_button):
        value -= 0.10

    # 4. Final Normalization
    # Ensure the returned value is strictly within the [0.0, 1.0] range.
    if value > 1.0:
        value = 1.0
    elif value < 0.0:
        value = 0.0
        
    return float(value)