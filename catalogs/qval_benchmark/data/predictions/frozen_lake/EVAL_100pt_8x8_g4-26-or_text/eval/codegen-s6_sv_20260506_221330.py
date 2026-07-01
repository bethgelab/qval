import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake 8x8 grid.
    The value is approximated using the Manhattan distance to the goal
    and a penalty for proximity to holes.
    """
    grid = []
    lines = state.strip().split('\n')
    for line in lines:
        # Handle both space-separated and non-space-separated ASCII grids
        clean_line = "".join(line.split())
        if clean_line:
            grid.append(list(clean_line))
    
    if not grid:
        return 0.0
        
    rows = len(grid)
    agent_pos = None
    goal_pos = None
    
    # Find agent (@), goal (G), and potential start (S) positions
    for r in range(rows):
        row_len = len(grid[r])
        for c in range(row_len):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'S' and agent_pos is None:
                agent_pos = (r, c)
                
    # If the agent position or goal position cannot be determined, value is 0.0
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # Manhattan distance to the goal
    dr = abs(agent_pos[0] - goal_pos[0])
    dc = abs(agent_pos[1] - goal_pos[1])
    dist = dr + dc
    
    # If the agent is already at the goal, value is 1.0
    if dist == 0:
        return 1.0
        
    # Check for proximity to holes (H)
    # In stochastic environments, being adjacent to a hole is a risk factor
    adj_holes = 0
    for dr_m, dc_m in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_pos[0] + dr_m, agent_pos[1] + dc_m
        if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
            if grid[nr][nc] == 'H':
                adj_holes += 1
    
    # The heuristic uses a discount factor gamma to represent the expected reward
    # based on distance, as the reward is only granted at the goal.
    # V(s) is approximately gamma^dist.
    gamma = 0.95
    value = math.pow(gamma, dist)
    
    # Apply a penalty for being adjacent to holes to reflect the risk of falling in
    if adj_holes > 0:
        # The penalty reduces the expected value based on the number of nearby holes
        value *= math.pow(0.7, adj_holes)
        
    return float(value)