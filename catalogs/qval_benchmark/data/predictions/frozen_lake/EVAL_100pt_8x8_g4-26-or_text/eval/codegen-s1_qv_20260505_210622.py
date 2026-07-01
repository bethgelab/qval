import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) based on the proximity to the goal, 
    the avoidance of holes, and the efficiency of the move.
    """
    
    def parse_grid(grid_str: str):
        # Extract the grid part from the state string
        # The grid is typically a block of text containing characters like S, G, H, @, etc.
        # We look for the character grid pattern.
        lines = [line.strip() for line in grid_str.split('\n') if line.strip()]
        grid = []
        for line in lines:
            # Filter out lines that are likely just text descriptions
            if any(c in line for c in ('@', 'S', 'G', 'H', '.')):
                # Keep only valid grid characters
                row = [c for c in line if c in ('@', 'S', 'G', 'H', '.')]
                if row:
                    grid.append(row)
        return grid

    def find_positions(grid):
        agent_pos = None
        goal_pos = None
        holes = []
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == '@' or val == 'S':
                    agent_pos = (r, c)
                elif val == 'G':
                    goal_pos = (r, c)
                elif val == 'H':
                    holes.append((r, c))
        return agent_pos, goal_pos, holes

    grid = parse_grid(state)
    if not grid:
        return 0.0
        
    agent_pos, goal_pos, holes = find_positions(grid)
    
    # If agent is already in a terminal state (though usually handled by env)
    if not agent_pos or not goal_pos:
        return 0.0

    # Analyze the next state to see if the action was immediate success or failure
    next_grid = parse_grid(next_state)
    next_agent_pos, _, _ = find_positions(next_grid)
    
    # 1. Immediate reward/penalty detection
    if next_agent_pos == goal_pos:
        return 1.0
    
    # Check if next state resulted in a hole (agent position would be lost or H)
    # Since the agent moves *to* a position, we check if next_agent_pos is in holes
    # or if the agent is gone (implies fall in hole in some representations)
    if next_agent_pos is None:
        return 0.0
    
    if next_agent_pos in holes:
        return 0.0

    # 2. Heuristic: Distance to goal
    # Q(s, a) is roughly (1 / (distance_to_goal + 1)) * discount_factor
    # We use a Manhattan distance approach
    dist_to_goal = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    
    # 3. Heuristic: Safety (Avoidance of holes)
    # Check if the move brings us closer to a hole
    near_hole = False
    for hr, hc in holes:
        if abs(next_agent_pos[0] - hr) + abs(next_agent_pos[1] - hc) <= 1:
            near_hole = True
            break
    
    # Base value based on distance
    # We want higher values for smaller distances.
    # We use a decay to represent the difficulty of reaching the goal.
    # Max distance in 8x8 is 14.
    value = 1.0 / (dist_to_goal + 1.0)
    
    # Penalty for being next to a hole
    if near_hole:
        value *= 0.5
        
    # Penalty for moving away from goal
    curr_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    next_dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    
    if next_dist > curr_dist:
        value *= 0.8  # discourage moving away
        
    return float(value)