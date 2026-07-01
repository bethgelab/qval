def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given Frozen Lake grid state.
    The state-value is an approximation of the expected discounted cumulative reward,
    based on distance to the goal and the risk posed by nearby holes.
    """
    # Parse the ASCII representation of the grid to find the agent, goal, and holes.
    lines = state.strip().split('\n')
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(lines):
        # The grid may be space-separated (e.g., "S . H") or non-space-separated (e.g., "S.H").
        row = line.split()
        if len(row) < 8:
            # Extract only the relevant grid symbols if it's not space-separated.
            row = [char for char in line if char in 'SGH@.']
        
        for c, cell in enumerate(row):
            if '@' in cell:
                agent_pos = (r, c)
            elif 'G' in cell:
                goal_pos = (r, c)
            elif 'H' in cell:
                holes.append((r, c))
    
    # If the agent or goal is not present, the value is 0.
    if not agent_pos or not goal_pos:
        return 0.0
    
    # Calculate Manhattan distance from agent to goal.
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the distance exceeds the episode step limit, the goal cannot be reached.
    if dist > 30:
        return 0.0
    
    # Base value estimation using a discount factor gamma = 0.9.
    # The closer the agent is to the goal, the higher the expected value.
    value = 0.9 ** dist
    
    # Adjust the value based on the risk of falling into holes.
    # In a potentially "slippery" environment, proximity to holes decreases the value.
    for hr, hc in holes:
        h_dist = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if h_dist == 1:
            # Immediate danger: high penalty for being adjacent to a hole.
            value *= 0.7
        elif h_dist == 2:
            # Nearby danger: moderate penalty for being near a hole.
            value *= 0.9
        
        # General path risk: a small penalty for holes located in the bounding box 
        # between the agent and the goal, as they likely obstruct the optimal path.
        if (min(agent_pos[0], goal_pos[0]) <= hr <= max(agent_pos[0], goal_pos[0]) and 
            min(agent_pos[1], goal_pos[1]) <= hc <= max(agent_pos[1], goal_pos[1])):
            value *= 0.99
            
    # Ensure the estimated value is within the standard [0, 1] reward range.
    return max(0.0, min(1.0, value))