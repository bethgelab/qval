import math

def signal_function(state: str) -> float:
    # Parse the state string into a grid
    lines = state.strip().split('\n')
    rows = len(lines)
    cols = len(lines[0]) if lines else 0
    
    # Find agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '@':
                agent_pos = (i, j)
            elif char == 'G':
                goal_pos = (i, j)
            elif char == 'H':
                holes.append((i, j))
    
    # If agent is at goal, value is 1.0 (goal reached)
    if agent_pos == goal_pos:
        return 1.0
    
    # If agent fell into a hole, value is 0.0 (episode ended)
    if agent_pos in holes:
        return 0.0
    
    # If we can't find agent or goal, return low value
    if agent_pos is None or goal_pos is None:
        return 0.1
    
    # Calculate Manhattan distance to goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Maximum possible distance on 8x8 grid (corner to corner)
    max_dist = 14
    
    # Distance factor - exponential decay for smoother gradient
    # Closer to goal = higher value
    dist_factor = math.exp(-dist / 4)
    
    # Hole density factor - more holes = lower probability of success
    total_cells = rows * cols
    hole_density = len(holes) / total_cells if total_cells > 0 else 0
    # Assume typical frozen lake has 10-20 holes, penalize higher density
    hole_factor = max(0, 1 - hole_density * 3)
    
    # Check if agent is adjacent to holes (dangerous position)
    danger_penalty = 0
    if agent_pos:
        agent_row, agent_col = agent_pos
        neighbors = [
            (agent_row - 1, agent_col),
            (agent_row + 1, agent_col),
            (agent_row, agent_col - 1),
            (agent_row, agent_col + 1)
        ]
        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) in holes:
                    danger_penalty += 0.15
    
    # Combine factors: distance * hole_density * safety
    value = dist_factor * hole_factor * (1 - danger_penalty)
    
    # Ensure value is in [0, 1]
    return max(0.0, min(1.0, value))