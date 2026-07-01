def signal_function(state: str) -> float:
    import math

    # Parse the ASCII grid representation
    lines = state.strip().splitlines()
    grid = []
    for line in lines:
        # Remove whitespace to handle various grid formatting styles
        row = line.replace(" ", "").replace("\t", "")
        if row:
            grid.append(list(row))
    
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    
    # Identify agent ('@') and goal ('G') positions
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # If agent or goal position is not found, we cannot estimate value
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # If agent is already at the goal, reward is achieved
    if agent_pos == goal_pos:
        return 1.0
        
    # Manhattan distance as a base proxy for efficiency and reward decay
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Base value derived from distance. Max distance in an 8x8 grid is 14.
    # We use 15.0 to ensure the value smoothly approaches zero at the edge.
    v = max(0.0, 1.0 - (dist / 15.0))
    
    # Check if the goal is effectively unreachable (surrounded by holes)
    # If all adjacent cells to the goal are 'H', the goal is blocked.
    adj_safe_goal = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = goal_pos[0] + dr, goal_pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 'H':
                adj_safe_goal += 1
    
    if adj_safe_goal == 0:
        return 0.0
        
    # Incorporate local risk (proximity to holes)
    # The value is penalized based on the density of holes around the current agent position.
    adj_holes = 0
    possible_neighbors = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            possible_neighbors += 1
            if grid[nr][nc] == 'H':
                adj_holes += 1
                
    if possible_neighbors > 0:
        # The risk factor scales the value down linearly based on the proportion of neighbors that are holes.
        risk = adj_holes / possible_neighbors
        v *= (1.0 - risk)
            
    return float(v)