import re

def signal_function(state: str):
    # Initialize component scores
    goal_score = 0.0
    progress_score = 0.0
    error_penalty = 0.0
    stage_penalty = 0.0
    task_type_score = 0.0
    form_progress = 0.0
    action_progress = 0.0
    efficiency_penalty = 0.0
    navigation_penalty = 0.0
    
    # Normalize state for pattern matching
    text = state.lower()
    
    # 1. Goal Achievement Detection
    success_keywords = [
        "event added successfully", "event created successfully", "event saved successfully",
        "message sent successfully", "message delivered", "message sent",
        "task added successfully", "task created successfully", "todo added successfully",
        "file saved successfully", "changes saved", "code saved",
        "operation completed successfully", "task completed successfully",
        "successfully added", "successfully created", "successfully saved",
        "successfully sent", "successfully updated", "successfully deleted",
        "event created", "task created", "message sent", "todo added",
        "completed", "done", "finished", "saved", "added", "created", "sent"
    ]
    
    success_matches = sum(1 for kw in success_keywords if kw in text)
    if success_matches >= 1:
        goal_score = 1.0
        return 1.0, {
            "goal_score": 1.0,
            "progress_score": 1.0,
            "error_penalty": 0.0,
            "stage_penalty": 0.0,
            "task_type_score": 0.0,
            "form_progress": 0.0,
            "action_progress": 0.0,
            "efficiency_penalty": 0.0,
            "navigation_penalty": 0.0,
        }
    
    # 2. Error Detection
    error_keywords = [
        "error", "failed", "not found", "404", "cannot",
        "invalid", "blocked", "timeout", "permission denied",
        "unauthorized", "access denied", "invalid input",
        "required", "missing", "empty", "no results", "not available"
    ]
    
    error_matches = sum(1 for k in error_keywords if k in text)
    error_penalty = min(0.3, 0.06 * error_matches)
    
    # 3. Task Type Detection
    calendar_indicators = ["calendar", "event", "date", "time", "schedule", "appointment", "meeting"]
    todo_indicators = ["todo", "task", "list", "checklist", "item", "assign"]
    messenger_indicators = ["messenger", "message", "chat", "compose", "recipient", "subject", "inbox"]
    maps_indicators = ["maps", "location", "directions", "route", "place", "address"]
    code_indicators = ["code editor", "code", "file", "editor", "syntax", "run", "execute"]
    
    calendar_matches = sum(1 for k in calendar_indicators if k in text)
    todo_matches = sum(1 for k in todo_indicators if k in text)
    messenger_matches = sum(1 for k in messenger_indicators if k in text)
    maps_matches = sum(1 for k in maps_indicators if k in text)
    code_matches = sum(1 for k in code_indicators if k in text)
    
    max_matches = max(calendar_matches, todo_matches, messenger_matches, maps_matches, code_matches)
    if max_matches > 0:
        task_type_score = 0.10 * min(1.0, max_matches / 3)
    else:
        task_type_score = 0.0
    
    # 4. Form Progress - improved with better granularity
    form_open_keywords = ["new event", "new task", "compose", "create", "add", "new message", "add"]
    form_fill_keywords = ["title", "description", "content", "body", "text", "input", "subject", "to", "date", "time", "enter", "fill"]
    form_complete_keywords = ["save", "submit", "send", "confirm", "done", "finish", "create", "add", "check"]
    
    form_open_matches = sum(1 for k in form_open_keywords if k in text)
    form_fill_matches = sum(1 for k in form_fill_keywords if k in text)
    form_complete_matches = sum(1 for k in form_complete_keywords if k in text)
    
    form_progress = 0.0
    if form_fill_matches >= 3 and form_complete_matches > 0:
        form_progress = 0.50
    elif form_fill_matches >= 2 and form_complete_matches > 0:
        form_progress = 0.40
    elif form_fill_matches >= 1 and form_complete_matches > 0:
        form_progress = 0.30
    elif form_fill_matches >= 2 and form_open_matches > 0:
        form_progress = 0.25
    elif form_fill_matches >= 1 and form_open_matches > 0:
        form_progress = 0.18
    elif form_open_matches > 0 and form_fill_matches == 0:
        form_progress = 0.12
    elif form_complete_matches > 0:
        form_progress = 0.15
    else:
        form_progress = 0.0
    
    # 5. Action Progress
    action_keywords = ["click", "press", "button", "select", "choose"]
    form_action_keywords = ["fill", "type", "enter", "input", "text"]
    submission_action_keywords = ["submit", "send", "save", "confirm", "complete", "done", "add", "create"]
    
    action_matches = sum(1 for k in action_keywords if k in text)
    form_action_matches = sum(1 for k in form_action_keywords if k in text)
    submission_action_matches = sum(1 for k in submission_action_keywords if k in text)
    
    if submission_action_matches > 0:
        action_progress = 0.20
    elif form_action_matches > 0:
        action_progress = 0.18
    elif action_matches > 0:
        action_progress = 0.12
    else:
        action_progress = 0.0
    
    # 6. Navigation Penalty - improved detection
    nav_indicators = ["home", "dashboard", "inbox", "main", "overview", "list", "menu", "back", "previous", "navigate", "go to", "return", "go back", "navigate back", "previous page"]
    nav_matches = sum(1 for k in nav_indicators if k in text)
    
    if nav_matches >= 4 and form_fill_matches == 0 and form_complete_matches == 0:
        navigation_penalty = 0.20
    elif nav_matches >= 3 and form_fill_matches == 0 and form_complete_matches == 0:
        navigation_penalty = 0.15
    elif nav_matches >= 2 and form_fill_matches == 0 and form_complete_matches == 0:
        navigation_penalty = 0.10
    elif nav_matches >= 2 and form_open_matches > 0 and form_fill_matches == 0:
        navigation_penalty = 0.08
    elif nav_matches >= 1 and form_fill_matches == 0 and form_complete_matches == 0 and form_open_matches == 0:
        navigation_penalty = 0.06
    else:
        navigation_penalty = 0.0
    
    # 7. Efficiency Penalty - penalize states that seem inefficient
    repeated_nav = nav_matches >= 2 and form_progress < 0.15
    if repeated_nav:
        efficiency_penalty = 0.12
    elif nav_matches >= 1 and form_progress < 0.10:
        efficiency_penalty = 0.08
    elif nav_matches == 0 and form_progress < 0.08:
        efficiency_penalty = 0.06
    else:
        efficiency_penalty = 0.0
    
    # 8. Stage Penalty - detect incorrect task stages
    if task_type_score < 0.05 and form_fill_matches >= 2:
        stage_penalty = 0.12
    elif form_progress >= 0.35 and task_type_score < 0.05:
        stage_penalty = 0.10
    elif task_type_score >= 0.05 and form_progress < 0.10:
        stage_penalty = 0.08
    elif action_progress >= 0.15 and task_type_score == 0.0:
        stage_penalty = 0.08
    elif task_type_score == 0.0 and form_progress < 0.12 and action_progress < 0.08:
        stage_penalty = 0.10
    elif task_type_score == 0.0 and form_progress < 0.08 and action_progress < 0.05:
        stage_penalty = 0.12
    elif task_type_score < 0.05 and form_progress < 0.10:
        stage_penalty = 0.06
    else:
        stage_penalty = 0.0
    
    # 9. Calculate total progress score with proper capping
    progress_score = task_type_score + form_progress + action_progress
    progress_score = min(0.85, progress_score)
    
    # 10. Calculate final value with penalties, properly bounded
    total_value = progress_score - error_penalty - navigation_penalty - efficiency_penalty - stage_penalty
    total_value = max(0.0, min(1.0, total_value))
    
    return total_value, {
        "goal_score": goal_score,
        "progress_score": progress_score,
        "error_penalty": error_penalty,
        "stage_penalty": stage_penalty,
        "task_type_score": task_type_score,
        "form_progress": form_progress,
        "action_progress": action_progress,
        "efficiency_penalty": efficiency_penalty,
        "navigation_penalty": navigation_penalty,
    }