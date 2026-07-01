import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given Frozen Lake grid state.
    The value is based on the Manhattan distance to the goal and a 
    risk penalty for proximity to holes.
    """
    # 1. Parse the grid from the state string
    # The grid consists of 8 rows, each with 8 characters.
    # Valid grid characters: '@' (agent), 'G' (goal), 'H' (hole), 'S' (start), '.' or 'F' (safe)
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Use regex to extract only valid grid characters from each line
        row = re.findall(r'[SGF.H@]', line)
        if row:
            grid.append(row)
    
    # Fallback: if the lines don't directly represent the grid, search for a continuous block of characters
    if len(grid) < 8:
        all_chars = re.findall(r'[SGF.H@]', state)
        if len(all_chars) >= 64:
            grid = [all_chars[i:i+8] for i in range(0, 64, 8)]
        else:
            return 0.0
    
    # Normalize to an 8x8 grid
    grid = [row[:8] for row in grid[:8]]
    if len(grid) < 8:
        return 0.0

    # 2. Identify key positions: agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    # Handle case where '@' might be represented as 'S' (start) in some views
    if agent_pos is None:
        for r in range(8):
            for c in range(8):
                if grid[r][c] == 'S':
                    agent_pos = (r, c)
                    break
            if agent_pos: break
    
    if agent_pos is None or goal_pos is None:
        return 0.0

    # 3. Calculate Manhattan distance to the goal
    dist_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the goal is unreachable within the step limit (30), the value is 0
    if dist_goal > 30:
        return 0.0
    
    # 4. Heuristic-based value estimation
    # Base value: reward decays with distance to the goal (discounting efficiency)
    # We use a discount factor of 0.9 as a proxy for the importance of speed.
    v_base = 0.9 ** dist_goal
    
    # 5. Adjust for risk: proximity to holes
    # Count how many holes are immediately adjacent to the agent.
    # This accounts for the increased probability of failure in stochastic environments.
    adj_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            if grid[nr][nc] == 'H':
                adj_holes += 1
    
    # Apply a risk penalty: being adjacent to holes reduces the expected value.
    # This is an approximation of the probability of falling due to movement or noise.
    risk_penalty = adj_holes * 0.15
    
    # Final estimation: product of base efficiency and safety factor
    v_final = v_base * (1.0 - risk_penalty)
    
    # Clamp the value to ensure it stays within the valid [0, 1] range
    return max(0.0, min(1.0, float(v_final)))