import math

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        rows = len(lines)
        cols = max(len(line) for line in lines) if lines else 0
        
        for row_idx, line in enumerate(lines):
            for col_idx, cell in enumerate(line):
                if cell == '@':
                    agent_pos = (row_idx, col_idx)
                elif cell == 'G':
                    goal_pos = (row_idx, col_idx)
                elif cell == 'H':
                    holes.append((row_idx, col_idx))
        
        return agent_pos, goal_pos, holes, rows, cols
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    agent_pos, goal_pos, holes, rows, cols = parse_grid(state)
    next_agent_pos, _, _, _, _ = parse_grid(next_state)
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    dist_to_nearest_hole = min(
        [manhattan_distance(agent_pos, hole) for hole in holes],
        default=float('inf')
    )
    
    next_dist_to_goal = manhattan_distance(next_agent_pos, goal_pos)
    next_dist_to_nearest_hole = min(
        [manhattan_distance(next_agent_pos, hole) for hole in holes],
        default=float('inf')
    )
    
    if dist_to_goal == 0:
        return 1.0
    
    base_q = max(0.0, 1.0 - (dist_to_goal / 30.0))
    
    hole_penalty = max(0.0, (15.0 - dist_to_nearest_hole) / 15.0)
    
    direction_bonus = 0.0
    if next_dist_to_goal < dist_to_goal:
        direction_bonus = 0.15
    
    hole_avoidance_bonus = 0.0
    if next_dist_to_nearest_hole > dist_to_nearest_hole:
        hole_avoidance_bonus = 0.1
    
    stay_penalty = 0.0
    if agent_pos == next_agent_pos:
        stay_penalty = -0.2
    
    if dist_to_nearest_hole == 0:
        return 0.0
    
    q_value = base_q - hole_penalty + direction_bonus + hole_avoidance_bonus + stay_penalty
    return max(-0.3, min(1.0, q_value))