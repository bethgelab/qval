import re
import collections

def signal_function(state: str) -> float:
    # Check for task completion indicators
    completion_keywords = ['task complete', 'done', 'success', 'goal achieved', 'finished', 'completed', 'successfully']
    failure_keywords = ['failed', 'error', 'impossible', 'cannot', 'blocked', 'no']
    
    # Check if task appears complete
    for keyword in completion_keywords:
        if keyword in state.lower():
            return 1.0
    
    # Check for failure conditions
    for keyword in failure_keywords:
        if keyword in state.lower():
            # Check if this is a definitive failure or just a constraint
            if 'cannot' in state.lower() or 'impossible' in state.lower():
                return 0.0
    
    # Estimate progress based on task mentions
    progress_score = 0.0
    
    # Look for task-related information
    task_mentions = re.findall(r'task.*?goal|goal.*?task', state.lower())
    if task_mentions:
        progress_score += 0.2
    
    # Look for object manipulation mentions
    manipulation_patterns = [r'move', r'pick', r'put', r'open', r'close', r'clean', r'wash', r'dry']
    manipulation_count = sum(1 for pattern in manipulation_patterns if re.search(pattern, state.lower()))
    if manipulation_count > 0:
        progress_score += min(manipulation_count * 0.1, 0.3)
    
    # Look for location information (suggests agent is aware of environment)
    location_patterns = [r'location', r'room', r'kitchen', r'bedroom', r'living', r'closet', r'counter', r'table']
    location_count = sum(1 for pattern in location_patterns if pattern in state.lower())
    if location_count > 0:
        progress_score += min(location_count * 0.05, 0.1)
    
    # Look for positive action confirmations
    positive_confirmations = [r'picked up', r'placed', r'opened', r'closed', r'cleaned', r'washed', r'dried', r'moved']
    confirmation_count = sum(1 for pattern in positive_confirmations if re.search(pattern, state.lower()))
    if confirmation_count > 0:
        progress_score += min(confirmation_count * 0.08, 0.25)
    
    # Check for step count if mentioned (assume 40 max steps)
    step_match = re.search(r'(\d+)\s*step[s]?', state.lower())
    if step_match:
        steps_taken = int(step_match.group(1))
        remaining_steps = max(0, 40 - steps_taken)
        # More steps remaining = more opportunity but also more risk
        if remaining_steps >= 20:
            progress_score += 0.05
        elif remaining_steps >= 10:
            progress_score += 0.1
        elif remaining_steps >= 5:
            progress_score += 0.15
        elif remaining_steps >= 1:
            progress_score += 0.2
    
    # Cap the value between 0 and 1
    return min(1.0, max(0.0, progress_score))