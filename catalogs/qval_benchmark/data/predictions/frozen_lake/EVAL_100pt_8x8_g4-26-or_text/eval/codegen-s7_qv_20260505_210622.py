def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next_state in a Frozen Lake environment.
    The Q-value is approximated based on whether the action leads to the goal, a hole, 
    or a safe state with a certain distance to the goal.
    """
    rows = state.strip().split('\n')
    next_rows = next_state.strip().split('\n')
    
    # Find the agent's current position in the state (r, c)
    r, c = -1, -1
    for i, row in enumerate(rows):
        if '@' in row:
            r, c = i, row.find('@')
            break
    
    # Find the agent's position in the next_state (nr, nc)
    nr, nc = -1, -1
    for i, row in enumerate(next_rows):
        if '@' in row:
            nr, nc = i, row.find('@')
            break
            
    # If agent location is not found, return 0.0 as a fallback
    if nr == -1 or nc == -1:
        return 0.0

    # Find the goal's position in the state
    gr, gc = -1, -1
    for i, row in enumerate(rows):
        if 'G' in row:
            gr, gc = i, row.find('G')
            break
            
    # Determine the terrain at the agent's new position (nr, nc)
    # We look at the character in the 'state' grid at the new position
    # to see what the terrain was before the agent moved there.
    terrain = 'F'
    if nr == r and nc == c:
        # Agent hit a wall/moved off-grid; the terrain is the current position
        # Since we don't know if (r,c) was 'S' or 'F', and both are safe, we treat it as 'F'.
        # (The case where (r,c) is 'G' or 'H' is impossible as the episode would have ended).
        terrain = 'F'
    else:
        # Check if nr, nc is within grid boundaries and access the terrain
        if 0 <= nr < len(rows) and 0 <= nc < len(rows[nr]):
            terrain = rows[nr][nc]
        else:
            terrain = 'F'

    # Check for immediate outcomes
    if terrain == 'G':
        return 1.0
    if terrain == 'H':
        return 0.0
    
    # For safe states, estimate the value based on Manhattan distance to the goal
    if gr != -1:
        dist = abs(nr - gr) + abs(nc - gc)
        # Use an exponential decay (0.95^dist) to represent the expected discounted reward
        return 0.95 ** dist
    
    return 0.0