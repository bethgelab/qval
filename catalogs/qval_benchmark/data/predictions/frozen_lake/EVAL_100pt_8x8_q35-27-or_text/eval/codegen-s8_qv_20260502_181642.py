import re

def signal_function(state: str, action: str, next_state: str) -> float:
    def find_positions(grid_str):
        """Find agent, goal, and hole positions in a grid string"""
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = set()
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.add((r, c))
        
        return agent_pos, goal_pos, holes
    
    # Parse both states
    curr_agent, curr_goal, curr_holes = find_positions(state)
    next_agent, next_goal, next_holes = find_positions(next_state)
    
    # Handle missing positions
    if next_agent is None or next_goal is None:
        return 0.0
    
    # Check if reached goal - highest Q-value
    if next_agent == next_goal:
        return 1.0
    
    # Check if fell in hole - lowest Q-value
    if next_agent in next_holes:
        return -1.0
    
    # Calculate Manhattan distances to goal
    if curr_agent is None:
        curr_dist = 14  # default max distance
    else:
        curr_dist = abs(curr_agent[0] - curr_goal[0]) + abs(curr_agent[1] - curr_goal[1])
    next_dist = abs(next_agent[0] - next_goal[0]) + abs(next_agent[1] - next_goal[1])
    
    # Base Q-value on distance to goal (normalized)
    # Maximum distance in 8x8 grid is 14 (from corner to opposite corner)
    max_dist = 14
    base_q = 1.0 - (next_dist / max_dist)
    
    # Bonus for moving closer to goal, penalty for moving away
    if next_dist < curr_dist:
        base_q += 0.2
    elif next_dist > curr_dist:
        base_q -= 0.2
    
    # Penalty if agent didn't move (hit wall or invalid action)
    if next_agent == curr_agent:
        base_q -= 0.1
    
    # Bonus for being in a relatively safe position (not adjacent to many holes)
    if next_agent:
        adjacent_holes = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (next_agent[0] + dr, next_agent[1] + dc)
            if neighbor in next_holes:
                adjacent_holes += 1
        base_q -= 0.05 * adjacent_holes
    
    # Clamp to reasonable Q-value range
    return max(-1.0, min(1.0, base_q))