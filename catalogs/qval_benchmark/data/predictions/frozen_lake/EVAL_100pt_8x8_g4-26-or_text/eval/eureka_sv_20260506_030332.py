import collections

def signal_function(state: str):
    """
    Estimates the state-value V(s) for a Frozen Lake environment using a shortest-path
    heuristic with a refined safety penalty that is sensitive to the number of 
    adjacent holes along that path.
    """
    # Robustly parse the ASCII grid
    lines = state.split('\n')
    start_idx = 0
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    end_idx = len(lines) - 1
    while end_idx >= 0 and not lines[end_idx].strip():
        end_idx -= 1
    
    if start_idx > end_idx:
        return 0.0, {"error": 0.0}
    
    grid_lines = lines[start_idx:end_idx+1]
    rows = len(grid_lines)
    grid = []
    agent_pos = None
    goals = []
    
    for r in range(rows):
        row_chars = list(grid_lines[r])
        grid.append(row_chars)
        for c in range(len(row_chars)):
            char = row_chars[c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goals.append((r, c))
    
    if agent_pos is None:
        return 0.0, {"error": 0.0}
    
    # Immediate terminal state checks
    r_a, c_a = agent_pos
    if r_a >= rows or c_a >= len(grid[r_a]):
        return 0.0, {"error": 0.0}
        
    current_cell = grid[r_a][c_a]
    if current_cell == 'G':
        return 1.0, {"goal_reached": 1.0}
    if current_cell == 'H':
        return 0.0, {"hole_reached": 0.0}
        
    # BFS to find the shortest path to the goal (avoiding holes)
    queue = collections.deque([agent_pos])
    parent = {agent_pos: None}
    found_goal = None
    
    while queue:
        curr = queue.popleft()
        if curr in goals:
            found_goal = curr
            break
            
        r, c = curr
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
                if grid[nr][nc] != 'H' and (nr, nc) not in parent:
                    parent[(nr, nc)] = curr
                    queue.append((nr, nc))
                    
    if found_goal is None:
        return 0.0, {"dist_reward_component": 0.0, "safety_penalty_component": 0.0}
        
    # Reconstruct the shortest path
    path = []
    curr = found_goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    
    d_g = len(path) - 1
    
    # Step limit check
    if d_g > 30:
        return 0.0, {"dist_reward_component": 0.0, "safety_penalty_component": 0.0}
        
    # Component 1: Ideal discounted reward based on shortest path
    dist_reward = 0.99 ** d_g
    
    # Component 2: Safety penalty based on the riskiness of the path.
    # We calculate risk as the count of adjacent holes for each cell in the path.
    path_risk_score = 0.0
    for i in range(len(path) - 1):
        pr, pc = path[i]
        holes_around = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
                if grid[nr][nc] == 'H':
                    holes_around += 1
        # Increase risk weight to make it more discriminative
        path_risk_score += holes_around * 0.03
            
    # Scale the penalty so it refines the distance reward without overwhelming it entirely.
    # A high path_risk_score reduces the value significantly.
    safety_penalty = -(path_risk_score * dist_reward * 0.6)
    
    total = dist_reward + safety_penalty
    
    # Clamp total to ensure it stays non-negative
    if total < 0:
        total = 0.0
        safety_penalty = -dist_reward
        
    return total, {
        "dist_reward_component": dist_reward,
        "safety_penalty_component": safety_penalty,
    }