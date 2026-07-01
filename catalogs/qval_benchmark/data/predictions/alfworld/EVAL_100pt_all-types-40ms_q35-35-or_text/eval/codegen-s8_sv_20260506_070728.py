import re

def signal_function(state: str) -> float:
    s_lower = state.lower()
    
    # Check for terminal states
    if any(term in s_lower for term in ['success', 'completed', 'solved']):
        return 1.0
    if any(term in s_lower for term in ['failure', 'invalid', 'timeout', 'error']):
        return 0.0
        
    # Define stop words to filter noise
    stop_words = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down', 'out', 'off', 
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now', 
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
        'task', 'observation', 'inventory', 'you', 'see', 'it', 'this', 'that', 'these', 'those', 'i', 'me', 
        'my', 'we', 'our', 'us', 'they', 'them', 'their', 'he', 'him', 'his', 'she', 'her', 'its', 'what', 
        'which', 'who', 'whom', 'whose', 'if', 'because', 'although', 'though', 'while', 'unless', 'until', 
        'before', 'after', 'since', 'as', 'like', 'into', 'through', 'during', 'without', 'against', 'about', 
        'above', 'below', 'between', 'among', 'inside', 'outside', 'near', 'behind', 'next', 'across', 
        'along', 'around', 'past', 'whether', 'that', 'which', 'who', 'whom', 'whose', 'what', 'when', 
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 
        'don', 'should', 'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'need', 
        'dare', 'ought', 'used'
    }

    # Extract sections
    task_match = re.search(r'task\s*:\s*(.+?)(?=\n\s*(?:inventory|observation|task|$))', s_lower, re.DOTALL)
    obs_match = re.search(r'observation\s*:\s*(.+?)(?=\n\s*(?:inventory|task|$))', s_lower, re.DOTALL)
    inv_match = re.search(r'inventory\s*:\s*(.+?)(?=\n\s*(?:observation|task|$))', s_lower, re.DOTALL)

    task_text = task_match.group(1).strip() if task_match else ""
    obs_text = obs_match.group(1).strip() if obs_match else s_lower
    inv_text = inv_match.group(1).strip() if inv_match else ""
    
    # Combine current state text
    current_text = (obs_text + " " + inv_text).strip()
    
    # Extract words
    task_words = set(re.findall(r'\b[a-z]+\b', task_text)) - stop_words
    current_words = set(re.findall(r'\b[a-z]+\b', current_text)) - stop_words
    
    # Calculate overlap
    if not task_words:
        return 0.2
    
    overlap_count = 0
    for tw in task_words:
        for cw in current_words:
            if tw in cw or cw in tw:
                overlap_count += 1
                break
    
    ratio = overlap_count / len(task_words)
    
    # Boost if specific action words are matched
    action_words = {'clean', 'open', 'close', 'turn', 'put', 'take', 'go', 'wash', 'heat', 'cool', 'fill', 'empty', 'on', 'off'}
    task_actions = task_words & action_words
    current_actions = current_words & action_words
    action_match = len(task_actions & current_actions)
    
    score = ratio * 0.7 + min(action_match, 1) * 0.3
    
    return max(0.0, min(1.0, score))