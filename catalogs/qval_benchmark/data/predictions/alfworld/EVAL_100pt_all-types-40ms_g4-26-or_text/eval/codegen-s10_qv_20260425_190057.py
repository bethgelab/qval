import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The Q-value is a measure of how much reward is expected to be gained 
    from taking the action in the state, based on the progress made.
    """
    s_l, a_l, n_l = state.lower(), action.lower(), next_state.lower()

    # 1. Extract the task from the state description
    # ALFWorld tasks are typically formatted as "Your task is to [task description]."
    task_match = re.search(r"(?:task|goal) is to (.*?)(?:\.|$)", s_l)
    if not task_match:
        return 0.0
    task = task_match.group(1)

    # 2. Parse the task for the core object and location/target
    # This helps identify if the agent is interacting with the right things.
    def get_noun(words):
        stop_words = {'the', 'a', 'an', 'is', 'to', 'in', 'on', 'at', 'of', 'with', 'put', 'move', 'take', 'clean'}
        for word in reversed(words):
            if word not in stop_words and len(word) > 1:
                return word
        return None

    # Split task by common prepositions to isolate object and location
    parts = re.split(r'\s+(?:in|on|at|to|with|from)\s+', task)
    if len(parts) >= 2:
        obj = get_noun(parts[0].split())
        loc = get_noun(parts[1].split())
    else:
        # For simpler tasks like "clean the apple"
        obj = get_noun(task.split())
        loc = None

    # 3. Evaluate progress and assign Q-value
    
    # A. Task Completion (Highest Reward)
    # If the environment signals the task is done or specific completion criteria are met.
    if "complete" in n_l or "finished" in n_l or "success" in n_l:
        return 1.0
    
    # Special case: "clean the [object]" tasks
    if "clean" in task and "clean" in n_l and obj and obj in n_l:
        return 1.0

    # B. Successful Placement (End of task)
    # If the target object and location are both in the next state, and the action was a placement action.
    if obj and loc and obj in n_l and loc in n_l:
        if any(verb in a_l for verb in ["put", "place", "drop", "move", "go"]):
            return 1.0
        # If they are just in the same room or visible, it's high progress but not necessarily done.
        return 0.5

    # C. Intermediate Progress: Holding the object
    # If the agent is successfully holding the object mentioned in the task.
    if ("holding" in n_l or "hand" in n_l or "have" in n_l) and obj and obj in n_l:
        return 0.7

    # D. Intermediate Progress: Proximity and Manipulation
    # Check if the action moved the agent closer to the object or target location.
    if loc and loc in n_l:
        if any(v in a_l for v in ["go", "move", "walk", "navigate", "travel"]):
            return 0.4
            
    if obj and obj in n_l:
        # Agent picked up the object
        if any(v in a_l for v in ["take", "pick", "grab", "get", "acquire"]):
            return 0.6
        # Agent moved toward the object
        if any(v in a_l for v in ["go", "move", "walk", "navigate"]):
            return 0.3

    # E. Generic Task Alignment
    # If the action uses keywords from the task, even if specific progress isn't clear.
    task_words = set(re.findall(r'\w+', task))
    action_words = set(re.findall(r'\w+', a_l))
    if task_words.intersection(action_words):
        return 0.2

    return 0.0