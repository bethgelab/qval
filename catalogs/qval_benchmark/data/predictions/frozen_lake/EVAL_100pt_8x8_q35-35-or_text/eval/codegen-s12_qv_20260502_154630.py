def signal_function(state: str, action: str, next_state: str) -> float:
    def find_coords(grid_str, char):
        lines = grid_str.strip().split('\n')
        for r, line in enumerate(lines):
            idx = line.find(char)
            if idx != -1:
                return (r, idx)
        return None

    def find_all_coords(grid_str, char):
        lines = grid_str.strip().split('\n')
        coords = []
        for r, line in enumerate(lines):
            start = 0
            while True:
                idx = line.find(char, start)
                if idx == -1:
                    break
                coords.append((r, idx))
                start = idx + 1
        return coords

    # Parse next_state
    agent_next = find_coords(next_state, '@')
    goal_pos = find_coords(next_state, 'G')
    if goal_pos is None:
        goal_pos = find_coords(state, 'G')
    
    # Check if goal reached
    if agent_next == goal_pos:
        return 1.0
    
    # Check if hole reached
    holes = find_all_coords(next_state, 'H')
    if agent_next in holes:
        return 0.0
    
    # Calculate distances
    if agent_next is None or goal_pos is None:
        return 0.0
        
    dist_next = abs(agent_next[0] - goal_pos[0]) + abs(agent_next[1] - goal_pos[1])
    
    agent_start = find_coords(state, '@')
    dist_start = float('inf')
    if agent_start is not None and goal_pos is not None:
        dist_start = abs(agent_start[0] - goal_pos[0]) + abs(agent_start[1] - goal_pos[1])
        
    # Base value based on distance to goal (heuristic potential)
    # Closer to goal -> higher value. 1/(1+dist) maps 0->1, large->0.
    base_q = 1.0 / (1.0 + dist_next)
    
    # Adjust for action efficiency
    if dist_start != float('inf'):
        if dist_next < dist_start:
            base_q += 0.15  # Moved closer
        elif dist_next > dist_start:
            base_q -= 0.15  # Moved away
            
    # Adjust for safety (proximity to holes in next_state)
    safety_penalty = 0.0
    if agent_next:
        r, c = agent_next
        # Check 4-neighbors
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        for nr, nc in neighbors:
            if (nr, nc) in holes:
                safety_penalty += 0.2
                
    base_q -= safety_penalty
    
    # Clip to valid range [0.0, 1.0]
    return max(0.0, min(1.0, base_q))