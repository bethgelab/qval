import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) of a given Frozen Lake state.
    The value is estimated based on the proximity to the goal, the density of holes,
    and the immediate safety of the agent's surroundings.
    """
    # 1. Parse the grid from the ASCII state string
    # We extract relevant characters and attempt to reconstruct the 8x8 grid.
    rows = []
    for line in state.strip().split('\n'):
        row = [c for c in line if c in '@GSFH']
        if row:
            rows.append(row)
    
    flat_grid = []
    for row in rows:
        flat_grid.extend(row)
        
    # If parsing via rows fails to find 64 cells, try a direct character scan
    if len(flat_grid) < 64:
        flat_grid = [c for c in state if c in '@GSFH']
        
    if len(flat_grid) < 64:
        return 0.0
            
    grid_8x8 = [flat_grid[i:i+8] for i in range(0, 64, 8)]

    # 2. Locate important grid elements: Agent (@), Goal (G), and Holes (H)
    agent_pos = None
    goal_pos = None
    hole_count = 0
    
    for r in range(8):
        for c in range(8):
            char = grid_8x8[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_count += 1
    
    # If the agent or goal is not found, return 0.0
    if not agent_pos or not goal_pos:
        return 0.0

    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Manhattan distance to the goal
    dist = abs(ar - gr) + abs(ac - gc)

    # 3. Estimate the value based on distance, safety, and efficiency
    # If the goal is out of reach within the 30-step limit, value is 0.
    if dist > 30:
        return 0.0
    # If the agent is already at the goal, value is 1.
    if dist == 0:
        return 1.0

    # Base value: Exponential decay based on distance to goal.
    # We assume a discount factor around 0.95.
    val = 0.95 ** dist

    # Penalty based on overall hole density in the grid.
    # A higher density of holes decreases the probability of a successful path.
    density_penalty = hole_count / 64.0
    val *= (1.0 - density_penalty)

    # Penalty based on immediate surroundings (local safety).
    adj_holes = 0
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            neighbors.append((nr, nc))
            if grid_8x8[nr][nc] == 'H':
                adj_holes += 1
    
    # If all reachable adjacent cells are holes, the agent is effectively trapped.
    if neighbors and all(grid_8x8[nr][nc] == 'H' for nr, nc in neighbors):
        return 0.0
    
    # Scale the value by the ratio of safe moves available.
    # This accounts for the risk of falling into a hole if an optimal move isn't immediately obvious.
    if neighbors:
        safe_moves_ratio = (len(neighbors) - adj_holes) / len(neighbors)
        # We interpolate between 0.5 and 1.0 to prevent the value from dropping too sharply
        # while still penalizing immediate danger.
        val *= (0.5 + 0.5 * safe_moves_ratio)

    # Ensure the returned value is within the valid [0.0, 1.0] range.
    return max(0.0, min(1.0, val))