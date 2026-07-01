def signal_function(state: str):
    # Parse the grid from state string
    lines = [line for line in state.strip().split('\n') if line.strip()]
    if not lines:
        return 0.0, {"value": 0.0}
    
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    
    agent_pos = None
    goal_pos = None
    holes = []
    steps_remaining = 30  # Default
    
    # Parse grid and look for step count metadata
    for line in lines:
        # Check for step count in format like "Steps: 11/30" or "Step 11"
        if ':' in line and ('step' in line.lower() or 'Step' in line):
            import re
            match = re.search(r'(\d+)\s*/\s*(\d+)', line)
            if match:
                current_step = int(match.group(1))
                max_steps = int(match.group(2))
                steps_remaining = max_steps - current_step
                continue
            match = re.search(r'step\s+(\d+)', line, re.IGNORECASE)
            if match:
                current_step = int(match.group(1))
                steps_remaining = 30 - current_step
                continue
    
    # Parse grid with proper indices
    for i, line in enumerate(lines):
        for j, cell in enumerate(line):
            if cell == '@':
                agent_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
            elif cell == 'H':
                holes.append((i, j))
    
    if agent_pos is None or goal_pos is None:
        return 0.0, {"value": 0.0}
    
    # Check if already at goal
    if agent_pos == goal_pos:
        return 1.0, {"goal_component": 1.0, "safety_component": 1.0, "mobility_component": 1.0, "time_component": 1.0}
    
    # Manhattan distance to goal
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Count safe adjacent cells (not holes, within bounds)
    safe_neighbors = 0
    adjacent_holes = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = agent_pos[0] + di, agent_pos[1] + dj
        if 0 <= ni < height and 0 <= nj < width and nj < len(lines[ni]):
            if lines[ni][nj] == 'H':
                adjacent_holes += 1
            else:
                safe_neighbors += 1
    
    # Minimum Manhattan distance to any hole
    min_dist_to_hole = float('inf')
    for hole in holes:
        d = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        min_dist_to_hole = min(min_dist_to_hole, d)
    
    # Maximum possible Manhattan distance in grid
    max_dist = height + width - 2
    
    # Hole density - proportion of cells that are holes
    total_cells = height * width
    hole_density = len(holes) / total_cells if total_cells > 0 else 0
    
    # Count holes in the bounding box between agent and goal (path risk)
    path_holes = 0
    for hole in holes:
        min_row = min(agent_pos[0], goal_pos[0])
        max_row = max(agent_pos[0], goal_pos[0])
        min_col = min(agent_pos[1], goal_pos[1])
        max_col = max(agent_pos[1], goal_pos[1])
        if min_row <= hole[0] <= max_row and min_col <= hole[1] <= max_col:
            path_holes += 1
    
    # Goal component - calibrated for realistic probability in random maps
    if max_dist > 0:
        # More conservative decay to avoid over-optimism
        goal_component = max(0.0, 1.0 - (dist_to_goal / max_dist) ** 1.8)
    else:
        goal_component = 1.0
    
    # Time component - sensitive to remaining steps with stronger time pressure
    if steps_remaining <= 0:
        time_component = 0.0
    elif dist_to_goal > steps_remaining:
        # Impossible to reach goal in time - severe penalty
        time_component = 0.0
    else:
        # Ratio of distance needed to steps available
        time_ratio = dist_to_goal / steps_remaining
        # Stronger decay with time pressure - more aggressive when close to limit
        if time_ratio < 0.3:
            time_component = 1.0 - (time_ratio ** 1.2)
        elif time_ratio < 0.6:
            time_component = 1.0 - (time_ratio ** 1.5)
        else:
            time_component = max(0.0, 1.0 - (time_ratio ** 2.0))
    
    # Safety component - maintain high variance for discrimination
    if min_dist_to_hole == float('inf'):
        # No holes in grid - very safe
        safety_component = 1.0
    elif adjacent_holes >= 3:
        # Adjacent to 3+ holes - extremely dangerous
        base = 0.0
        density_penalty = hole_density * 0.5
        path_penalty = min(0.4, path_holes * 0.15)
        escape_bonus = 0.15 * (safe_neighbors / 4.0)
        safety_component = max(0.0, base - density_penalty - path_penalty + escape_bonus)
    elif adjacent_holes == 2:
        # Adjacent to 2 holes - very dangerous
        base = 0.05
        density_penalty = hole_density * 0.4
        path_penalty = min(0.3, path_holes * 0.12)
        escape_bonus = 0.2 * (safe_neighbors / 4.0)
        safety_component = max(0.0, base - density_penalty - path_penalty + escape_bonus)
    elif adjacent_holes == 1:
        # Adjacent to 1 hole - dangerous but manageable
        base = 0.15
        density_penalty = hole_density * 0.3
        path_penalty = min(0.22, path_holes * 0.09)
        escape_bonus = 0.25 * (safe_neighbors / 4.0)
        safety_component = max(0.0, base - density_penalty - path_penalty + escape_bonus)
    else:
        # No adjacent holes - use distance-based safety with topology factors
        dist_factor = min(1.0, 0.4 + min_dist_to_hole * 0.25)
        density_factor = max(0.65, 1.0 - hole_density * 0.5)
        path_factor = max(0.75, 1.0 - (path_holes / max(1, dist_to_goal)) * 0.3)
        safety_component = dist_factor * density_factor * path_factor
    
    # Mobility component - more safe neighbors means more options
    mobility_component = safe_neighbors / 4.0
    
    # Weighted combination - balanced weights with emphasis on time
    total = 0.28 * goal_component + 0.32 * time_component + 0.24 * safety_component + 0.16 * mobility_component
    
    return total, {
        "goal_component": goal_component,
        "time_component": time_component,
        "safety_component": safety_component,
        "mobility_component": mobility_component,
    }