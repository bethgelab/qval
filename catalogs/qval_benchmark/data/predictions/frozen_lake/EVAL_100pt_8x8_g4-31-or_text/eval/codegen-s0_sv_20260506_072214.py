import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the Frozen Lake grid environment.
    V(s) is approximated as the probability of reaching the goal, 
    penalized by the distance to the goal and proximity to holes.
    """
    # Parse the grid
    lines = state.strip().split('\n')
    grid = [list(line.strip()) for line in lines if line.strip()]
    
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    agent_pos = None
    goal_pos = None
    
    # Locate the agent '@' and the goal 'G'
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                agent_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
                
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    ra, ca = agent_pos
    rg, cg = goal_pos
    
    # 1. Distance-based value
    # Use Manhattan distance as a heuristic for the remaining steps.
    dist = abs(ra - rg) + abs(ca - cg)
    
    # A discount factor gamma^dist represents the expected value if 
    # the path were clear and optimal play was followed.
    gamma = 0.92
    value = math.pow(gamma, dist)
    
    # 2. Risk-based penalty
    # Check the immediate neighbors for holes 'H'.
    # The more holes surrounding the agent, the lower the probability of survival.
    holes_nearby = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = ra + dr, ca + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 'H':
                holes_nearby += 1
    
    # Reduce value based on local risk. Each adjacent hole reduces value by 15%.
    risk_multiplier = 1.0 - (0.15 * holes_nearby)
    value *= max(0.0, risk_multiplier)
    
    # 3. Boundary/Trap considerations
    # If the agent is at the goal, value is 1.0.
    if dist == 0:
        return 1.0
        
    # Ensure the result is clamped between 0.0 and 1.0
    return max(0.0, min(1.0, value))