def signal_function(state: str, action: str, next_state: str):
    def parse_grid(grid_str):
        grid_lines = grid_str.split('\n')
        return [list(line) for line in grid_lines]
    
    def find_positions(grid):
        agent_pos = None
        goal_pos = None
        holes = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                if cell == '@':
                    agent_pos = (r, c)
                elif cell == 'G':
                    goal_pos = (r, c)
                elif cell == 'H':
                    holes.append((r, c))
        return agent_pos, goal_pos, holes
    
    def manhattan_distance(p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        return abs(r1 - r2) + abs(c1 - c2)
    
    def get_action_delta(action):
        action_map = {
            'left': (0, -1),
            'down': (1, 0),
            'right': (0, 1),
            'up': (-1, 0)
        }
        return action_map.get(action, (0, 0))
    
    grid = parse_grid(state)
    next_grid = parse_grid(next_state)
    agent_pos, goal_pos, holes = find_positions(grid)
    next_agent_pos, next_goal_pos, next_holes = find_positions(next_grid)
    
    if agent_pos is None or goal_pos is None:
        return 0.0, {
            "goal_distance": 0.0,
            "hole_distance": 0.0,
            "action_toward_goal": 0.0,
            "safety_score": 0.0,
            "remaining_steps": 0.0,
            "immediate_reward": 0.0,
        }
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    current_distance = manhattan_distance(agent_pos, goal_pos)
    max_steps = 30
    
    # Normalize goal distance - account for grid size
    max_possible_distance = (rows - 1) + (cols - 1)
    goal_distance = 1.0 - (current_distance / max_possible_distance) if max_possible_distance > 0 else 0.0
    goal_distance = max(0.0, min(1.0, goal_distance))
    
    # Calculate distance to nearest hole
    min_hole_distance = float('inf')
    for hole in holes:
        hole_dist = manhattan_distance(agent_pos, hole)
        min_hole_distance = min(min_hole_distance, hole_dist)
    
    # Safety score based on hole proximity - sigmoid-like function
    hole_safety = 1.0 / (1.0 + min_hole_distance / 5.0) if min_hole_distance != float('inf') else 1.0
    
    # Check if action moves toward goal (decreases Manhattan distance)
    dr, dc = get_action_delta(action)
    new_r, new_c = agent_pos[0] + dr, agent_pos[1] + dc
    action_valid = (0 <= new_r < rows and 0 <= new_c < cols)
    
    # Get next cell content to evaluate action quality
    next_cell = grid[new_r][new_c] if action_valid else grid[agent_pos[0]][agent_pos[1]]
    
    # Check if action is safe (not falling into hole or off grid)
    action_safe = True
    if action_valid and next_cell == 'H':
        action_safe = False
    elif not action_valid:
        action_safe = False
    
    # Calculate distance improvement
    new_distance = manhattan_distance((new_r, new_c), goal_pos) if action_valid else current_distance
    distance_improvement = 1.0 if new_distance < current_distance else (0.0 if new_distance == current_distance else 0.0)
    
    # Action quality score: combines direction toward goal and safety
    action_quality = distance_improvement * 0.6 + action_safe * 0.4
    
    # Check if action reaches goal or falls in hole
    next_is_goal = (next_agent_pos == goal_pos)
    next_is_hole = (next_agent_pos in next_holes)
    
    # Immediate reward based on sparse reward structure
    immediate_reward = 1.0 if next_is_goal else (0.0 if not next_is_hole else -0.5)
    
    # Estimate remaining steps based on distance and grid size
    remaining_steps = max(0, max_steps - current_distance)
    remaining_steps_normalized = remaining_steps / max_steps
    
    # Weighted combination of features
    q_value = (
        goal_distance * 0.30 +
        hole_safety * 0.20 +
        distance_improvement * 0.15 +
        action_quality * 0.15 +
        remaining_steps_normalized * 0.10 +
        immediate_reward * 0.10
    )
    
    q_value = max(-0.5, min(1.0, q_value))
    
    return q_value, {
        "goal_distance": goal_distance,
        "hole_distance": hole_safety,
        "action_toward_goal": distance_improvement,
        "safety_score": action_quality,
        "remaining_steps": remaining_steps_normalized,
        "immediate_reward": immediate_reward,
    }