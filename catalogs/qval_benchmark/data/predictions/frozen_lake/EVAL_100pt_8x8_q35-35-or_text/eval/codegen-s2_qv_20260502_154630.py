import math
import re

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [line for line in lines if line]
        pos = {}
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == '@':
                    pos['agent'] = (r, c)
                elif cell == 'G':
                    pos['goal'] = (r, c)
                elif cell == 'H':
                    if 'holes' not in pos:
                        pos['holes'] = []
                    pos['holes'].append((r, c))
        if 'goal' not in pos:
            pos['goal'] = None
        if 'holes' not in pos:
            pos['holes'] = []
        return pos
    
    def manhattan_distance(p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def is_goal_reached(pos):
        return pos.get('agent') == pos.get('goal')
    
    def is_in_hole(pos):
        agent_pos = pos.get('agent')
        if agent_pos is None:
            return False
        return agent_pos in pos.get('holes', [])
    
    def action_direction(action):
        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        return directions.get(action, (0, 0))
    
    state_pos = parse_grid(state)
    next_pos = parse_grid(next_state)
    
    agent_pos = state_pos.get('agent')
    goal_pos = state_pos.get('goal')
    holes = state_pos.get('holes', [])
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    
    min_dist_to_hole = float('inf')
    for hole in holes:
        d = manhattan_distance(agent_pos, hole)
        min_dist_to_hole = min(min_dist_to_hole, d)
    
    goal_reached = is_goal_reached(next_pos)
    in_hole = is_in_hole(next_pos)
    
    if goal_reached:
        return 1.0
    
    if in_hole:
        return 0.0
    
    action_dir = action_direction(action)
    next_agent = next_pos.get('agent')
    if next_agent is None:
        return 0.0
    
    dist_to_goal_next = manhattan_distance(next_agent, goal_pos)
    moved = (next_agent[0] != agent_pos[0] or next_agent[1] != agent_pos[1])
    
    dist_improvement = dist_to_goal - dist_to_goal_next
    
    safety_score = min(min_dist_to_hole / 14.0, 1.0)
    progress_score = max(0, dist_improvement) / 14.0
    
    if moved:
        movement_bonus = 0.1
    else:
        movement_bonus = 0.0
    
    if action_dir[0] < 0 and goal_pos[0] < agent_pos[0]:
        toward_goal = 0.1
    elif action_dir[0] > 0 and goal_pos[0] > agent_pos[0]:
        toward_goal = 0.1
    elif action_dir[1] < 0 and goal_pos[1] < agent_pos[1]:
        toward_goal = 0.1
    elif action_dir[1] > 0 and goal_pos[1] > agent_pos[1]:
        toward_goal = 0.1
    else:
        toward_goal = 0.0
    
    toward_hole = 0.0
    for hole in holes:
        if min_dist_to_hole < 3:
            if action_dir[0] < 0 and hole[0] < agent_pos[0]:
                toward_hole = 0.15
            elif action_dir[0] > 0 and hole[0] > agent_pos[0]:
                toward_hole = 0.15
            elif action_dir[1] < 0 and hole[1] < agent_pos[1]:
                toward_hole = 0.15
            elif action_dir[1] > 0 and hole[1] > agent_pos[1]:
                toward_hole = 0.15
            break
    
    base_value = 0.5
    q_value = base_value + progress_score + toward_goal + movement_bonus - toward_hole
    q_value = q_value + (safety_score - 0.5) * 0.3
    
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value