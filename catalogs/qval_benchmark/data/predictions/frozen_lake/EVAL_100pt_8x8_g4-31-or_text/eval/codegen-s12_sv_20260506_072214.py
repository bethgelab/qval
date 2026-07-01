def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a reinforcement learning agent in a 
    Frozen Lake (8x8) environment. The state-value is the expected discounted 
    cumulative reward, which for this sparse reward environment is essentially 
    the probability of reaching the goal G from the current state @.
    """
    # --- Parsing the State ---
    # The state is an ASCII representation of the 8x8 grid.
    # We extract the grid as a 2D list of characters.
    lines = [line.strip() for line in state.strip().split('\n') if line.strip()]
    grid_chars = []
    for line in lines:
        grid_chars.append([char for char in line if char in 'SGH@.'])
    
    # Ensure we have a 8x8 representation. Handle potential formatting variations.
    if len(grid_chars) != 8:
        all_chars = "".join(["".join(row) for row in grid_chars])
        if len(all_chars) >= 64:
            all_chars = all_chars[:64]
            grid_chars = [list(all_chars[i:i+8]) for i in range(0, 64, 8)]
        else:
            return 0.0
    
    # Final check to ensure it's exactly 8x8
    if len(grid_chars) != 8 or any(len(row) != 8 for row in grid_chars):
        return 0.0

    # --- Feature Extraction ---
    ar, ac = -1, -1  # Agent position
    gr, gc = -1, -1  # Goal position
    holes = []       # List of hole positions
    
    for r in range(8):
        for c in range(8):
            char = grid_chars[r][c]
            if char == '@':
                ar, ac = r, c
            elif char == 'G':
                gr, gc = r, c
            elif char == 'H':
                holes.append((r, c))
    
    # Return 0 if agent or goal cannot be identified
    if ar == -1 or gr == -1:
        return 0.0
    
    # Distance to goal is the primary driver of value
    dist = abs(ar - gr) + abs(ac - gc)
    if dist == 0:
        return 1.0
    
    # --- Domain Knowledge Reasoning ---
    
    # 1. Trapped Check: If the agent is surrounded by holes and borders, it's doomed.
    safe_neighbors = 0
    on_grid_neighbors = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            on_grid_neighbors += 1
            if grid_chars[nr][nc] != 'H':
                safe_neighbors += 1
    
    # If there are neighbors on the grid and none are safe, value is 0.
    if on_grid_neighbors > 0 and safe_neighbors == 0:
        return 0.0

    # 2. Goal Accessibility: If the goal is completely blocked by holes, it's unreachable.
    goal_safe_neighbors = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = gr + dr, gc + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            if grid_chars[nr][nc] != 'H':
                goal_safe_neighbors += 1
    
    if goal_safe_neighbors == 0:
        return 0.0

    # --- Value Estimation ---
    # We assume a discount factor gamma (e.g., 0.95).
    # The closer the agent is to the goal, the higher the value.
    gamma = 0.95
    value = gamma ** dist
    
    # 3. Path Difficulty: Estimate how many holes are in the bounding box 
    # between the agent and the goal. More holes typically mean a higher 
    # probability that the optimal path is long or nonexistent.
    holes_in_way = 0
    for hr, hc in holes:
        if min(ar, gr) <= hr <= max(ar, gr) and min(ac, gc) <= hc <= max(ac, gc):
            # Exclude the agent's current position (though @ shouldn't be H)
            if (hr, hc) != (ar, ac):
                holes_in_way += 1
    
    # Apply a mild penalty for holes in the path area.
    # We use a soft decay to avoid zeroing out the value for many holes.
    value *= (0.98 ** holes_in_way)
    
    return float(value)