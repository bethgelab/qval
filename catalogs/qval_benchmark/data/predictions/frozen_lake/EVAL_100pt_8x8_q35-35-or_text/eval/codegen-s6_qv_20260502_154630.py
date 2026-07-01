import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        goal_pos = None
        holes = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '@':
                    agent_pos = (r, c)
                elif grid[r][c] == 'G':
                    goal_pos = (r, c)
                elif grid[r][c] == 'H':
                    holes.append((r, c))
        
        return grid, agent_pos, goal_pos, holes, rows, cols
    
    def manhattan_distance(p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def is_safe(next_agent_pos, next_grid, next_holes, rows, cols):
        if next_agent_pos is None:
            return False
        r, c = next_agent_pos
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if next_grid[r][c] == 'H':
            return False
        return True
    
    def action_effectiveness(action, curr_agent, next_agent):
        if curr_agent is None or next_agent is None:
            return 0.0
        
        dr = next_agent[0] - curr_agent[0]
        dc = next_agent[1] - curr_agent[1]
        
        if action == 'up':
            return -dr
        elif action == 'down':
            return dr
        elif action == 'left':
            return -dc
        elif action == 'right':
            return dc
        return 0.0
    
    def count_holes_nearby(agent_pos, holes, rows, cols):
        if agent_pos is None:
            return 0
        r, c = agent_pos
        count = 0
        for hr, hc in holes:
            if abs(r - hr) <= 1 and abs(c - hc) <= 1:
                count += 1
        return count
    
    # Parse both states
    _, curr_agent, curr_goal, _, _, _ = parse_grid(state)
    next_grid, next_agent, next_goal, next_holes, next_rows, next_cols = parse_grid(next_state)
    
    # Check if goal reached
    goal_reached = next_goal is not None and next_agent == next_goal
    if goal_reached:
        return 0.95
    
    # Check if agent fell into hole
    if next_agent is not None and next_grid[next_agent[0]][next_agent[1]] == 'H':
        return 0.0
    
    # Check if agent is in a hole (agent position shows H)
    if next_agent is not None:
        r, c = next_agent
        if r >= 0 and r < next_rows and c >= 0 and c < next_cols:
            if next_grid[r][c] == 'H':
                return 0.0
    
    # Calculate distances
    curr_dist = manhattan_distance(curr_agent, curr_goal)
    next_dist = manhattan_distance(next_agent, next_goal)
    
    # Distance improvement (positive if closer to goal)
    dist_improvement = curr_dist - next_dist
    if next_goal is None:
        dist_improvement = 0
    
    # Safety score (fewer holes nearby = better)
    holes_nearby = count_holes_nearby(next_agent, next_holes, next_rows, next_cols)
    safety_score = max(0, 1.0 - holes_nearby * 0.2)
    
    # Action effectiveness (how well action moved toward goal)
    action_eff = action_effectiveness(action, curr_agent, next_agent)
    
    # Combined Q-value estimate
    # Base reward for making progress, adjusted by safety
    progress_bonus = min(0.3, max(0.0, dist_improvement) * 0.1)
    
    # Penalty for moving away or not improving
    if dist_improvement < 0:
        progress_bonus = progress_bonus + dist_improvement * 0.05
    
    # Combine factors
    q_value = 0.1 + progress_bonus * 0.5 + safety_score * 0.3
    
    # Cap at reasonable values
    q_value = max(0.0, min(0.9, q_value))
    
    return q_value