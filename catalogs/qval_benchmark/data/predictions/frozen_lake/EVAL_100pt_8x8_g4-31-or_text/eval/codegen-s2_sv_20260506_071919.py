import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an agent in a Frozen Lake grid.
    V(s) is approximated based on the Manhattan distance to the goal and 
    the distance to the nearest hole.
    """
    # Parse the grid into a list of strings (rows)
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Remove whitespace to handle both space-separated and continuous strings
        clean_line = "".join(line.split())
        if clean_line:
            grid.append(clean_line)
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Identify key positions: Agent (@), Goal (G), and Holes (H)
    for r in range(rows):
        row_str = grid[r]
        for c in range(len(row_str)):
            char = row_str[c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    if agent_pos is None:
        return 0.0
        
    # Calculate Manhattan distance to goal
    # If 'G' is not visible, we assume the agent has reached the goal (the '@' is on 'G')
    if goal_pos is None:
        dist_g = 0.0
    else:
        dist_g = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        
    # Calculate Manhattan distance to the nearest hole
    min_dist_h = float('inf')
    for hr, hc in holes:
        d = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if d < min_dist_h:
            min_dist_h = d
            
    # V(s) approximation:
    # 1. Base value decays exponentially with distance to goal to reflect 
    #    efficiency and the step limit (0.9 is a common discount factor).
    # 2. A safety penalty is applied if the agent is adjacent to a hole.
    
    val = math.pow(0.9, dist_g)
    
    # Safety penalty for being immediately adjacent to a hole
    if min_dist_h == 1:
        val *= 0.5
    elif min_dist_h == 0:
        # Technically impossible since '@' is the agent, but for robustness:
        val = 0.0
        
    return float(val)