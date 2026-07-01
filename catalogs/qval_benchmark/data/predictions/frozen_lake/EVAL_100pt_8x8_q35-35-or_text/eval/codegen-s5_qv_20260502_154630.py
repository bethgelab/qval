import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the static map features from the 'state' string to find Goal and Hole locations
    # We assume the map (S, G, H) is static and only the agent (@) moves.
    state_rows = state.strip().split('\n')
    goal_pos = None
    holes = set()
    
    for r, row in enumerate(state_rows):
        for c, char in enumerate(row):
            if char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
    
    # Parse the 'next_state' string to find the agent's current position
    next_rows = next_state.strip().split('\n')
    agent_pos = None
    
    for r, row in enumerate(next_rows):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
    
    # Safety checks
    if goal_pos is None or agent_pos is None:
        return 0.0
    
    gr, gc = goal_pos
    ar, ac = agent_pos
    
    # Check if agent is on the Goal
    # If the agent is on the goal, the character might be '@' (covering 'G')
    # We compare coordinates since 'G' might not be visible in next_state if covered by '@'
    if agent_pos == goal_pos:
        return 1.0
    
    # Check if agent is on a Hole
    if agent_pos in holes:
        return 0.0
        
    # Calculate Manhattan Distance from Agent to Goal
    dist = abs(ar - gr) + abs(ac - gc)
    
    # Estimate Q-value based on distance and proximity to holes
    # Using a discount factor gamma ~ 0.9 to reflect preference for fewer steps
    gamma = 0.9
    base_value = math.pow(gamma, dist)
    
    # Apply penalty for proximity to holes in the next_state
    # If agent is next to a hole, the risk of falling (if stochastic) or danger is higher
    rows_count = len(next_rows)
    cols_count = len(next_rows[0]) if next_rows else 0
    hole_penalty = 0.0
    
    neighbors = [
        (ar - 1, ac), (ar + 1, ac),
        (ar, ac - 1), (ar, ac + 1)
    ]
    
    for nr, nc in neighbors:
        if 0 <= nr < rows_count and 0 <= nc < cols_count:
            if next_rows[nr][nc] == 'H':
                hole_penalty += 0.15
    
    final_value = base_value - hole_penalty
    
    return max(0.0, final_value)