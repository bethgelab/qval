import re
from collections import Counter

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check if goal is already achieved
    if any(term in state_lower for term in ['goal achieved', 'task complete', 'success', 'verified']):
        return 1.0
    
    # Check for error or failure indicators
    if any(term in state_lower for term in ['error', 'failed', 'invalid', 'not found', 'blocked']):
        return 0.1
    
    # Identify which app we're in
    apps = {
        'todo': ['todo', 'task', 'list', 'item'],
        'calendar': ['calendar', 'event', 'date', 'time', 'appointment'],
        'messenger': ['message', 'chat', 'send', 'inbox', 'contact'],
        'maps': ['map', 'location', 'address', 'direction', 'place'],
        'code': ['code', 'editor', 'file', 'script', 'function']
    }
    
    app_scores = {}
    for app, keywords in apps.items():
        count = sum(1 for kw in keywords if kw in state_lower)
        if count > 0:
            app_scores[app] = count
    
    # Determine current progress based on state features
    progress_indicators = {
        'form_filled': ['value', 'entered', 'filled', 'typed', 'input'],
        'button_available': ['button', 'submit', 'save', 'send', 'confirm', 'add', 'create'],
        'link_available': ['link', 'navigate', 'go to', 'view'],
        'selection_available': ['select', 'choose', 'pick', 'option'],
        'scroll_needed': ['scroll', 'more', 'load more']
    }
    
    progress_score = 0.0
    for indicator, keywords in progress_indicators.items():
        count = sum(1 for kw in keywords if kw in state_lower)
        if count > 0:
            progress_score += min(count * 0.1, 0.25)
    
    # Check if we're at a critical completion step
    completion_keywords = ['submit', 'send', 'save', 'confirm', 'create', 'add', 'complete']
    if any(kw in state_lower for kw in completion_keywords):
        progress_score += 0.2
    
    # Check for breadcrumb/navigation context
    nav_context = ['home', 'back', 'previous', 'next', 'menu', 'settings']
    if any(kw in state_lower for kw in nav_context):
        progress_score += 0.1
    
    # Penalize if we seem lost (too many navigation elements, unclear context)
    if state_lower.count('button') > 10 or state_lower.count('link') > 10:
        progress_score *= 0.7
    
    # Check for goal-specific progress
    goal_keywords = ['add', 'create', 'schedule', 'send', 'find', 'edit', 'delete', 'update']
    if any(kw in state_lower for kw in goal_keywords):
        progress_score += 0.15
    
    # Calculate base value from app relevance
    if app_scores:
        best_app = max(app_scores, key=app_scores.get)
        app_relevance = min(app_scores[best_app] * 0.15, 0.3)
    else:
        app_relevance = 0.0
    
    # Combine scores with discount for remaining steps
    # Assume we have ~45 steps max, estimate remaining based on progress
    estimated_remaining_steps = max(1, int((1 - progress_score) * 20))
    discount_factor = 0.95 ** estimated_remaining_steps
    
    # Final value estimate
    base_value = progress_score + app_relevance
    final_value = min(1.0, max(0.0, base_value * discount_factor))
    
    # Clamp to reasonable range for partial progress
    if final_value > 0.95 and not any(term in state_lower for term in ['submit', 'confirm', 'send']):
        final_value = 0.85
    
    return final_value