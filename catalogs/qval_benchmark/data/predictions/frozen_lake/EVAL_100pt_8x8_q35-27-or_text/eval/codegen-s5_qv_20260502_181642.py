import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse grid to find positions
    def find_positions(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))
        return agent_pos, goal_pos, holes
    
    # Get positions from both states
    curr_agent, goal_pos, holes = find_positions(state)
    next_agent, _, _ = find_positions(next_state)
    
    # Check if goal reached in next state
    if next_agent == goal_pos:
        return 1.0
    
    # Check if fell in hole in next state
    if next_agent in holes:
        return 0.0
    
    # Calculate Manhattan distance
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    dist_to_goal = manhattan_distance(next_agent, goal_pos)
    dist_curr_to_goal = manhattan_distance(curr_agent, goal_pos)
    
    # Base Q-value on distance to goal (closer = better)
    # Max distance in 8x8 grid is 14 (from corner to opposite corner)
    max_dist = 14.0
    base_value = 1.0 - (dist_to_goal / max_dist)
    
    # Bonus if moved closer to goal
    if dist_to_goal < dist_curr_to_goal:
        base_value += 0.15
    elif dist_to_goal > dist_curr_to_goal:
        base_value -= 0.1
    
    # Penalize if next position is adjacent to holes (dangerous)
    adjacent_holes = 0
    if next_agent:
        nr, nc = next_agent
        for hr, hc in holes:
            if abs(nr - hr) <= 1 and abs(nc - hc) <= 1:
                adjacent_holes += 1
        base_value -= adjacent_holes * 0.08
    
    # Bonus for being on safe ground (not adjacent to any holes)
    if adjacent_holes == 0:
        base_value += 0.05
    
    # Ensure value is in [0, 1]
    return max(0.0, min(1.0, base_value))