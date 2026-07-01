import re
from collections import defaultdict

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        goal_pos = None
        holes = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                cell = grid[r][c]
                if cell == '@':
                    agent_pos = (r, c)
                elif cell == 'G':
                    goal_pos = (r, c)
                elif cell == 'H':
                    holes.append((r, c))
        return grid, agent_pos, goal_pos, holes

    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_safe(cell):
        return cell not in ['H', 'X']

    grid, agent_pos, goal_pos, holes = parse_grid(state)
    next_grid, next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)

    if agent_pos is None or goal_pos is None:
        return 0.0

    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, next_goal_pos)

    action_delta = 0
    if action == 'up':
        action_delta = -1
    elif action == 'down':
        action_delta = 1
    elif action == 'left':
        action_delta = -1
    elif action == 'right':
        action_delta = 1

    dist_change = next_dist - current_dist

    is_goal_reached = next_goal_pos == next_agent_pos
    is_safe_next = is_safe(next_grid[next_agent_pos[0]][next_agent_pos[1]])

    safety_penalty = 0.5 if not is_safe_next else 0.0
    progress_bonus = 0.3 if dist_change < 0 else 0.0
    goal_bonus = 0.7 if is_goal_reached else 0.0
    position_factor = 1.0 / (current_dist + 1)

    base_estimate = 0.2
    q_value = base_estimate + progress_bonus + goal_bonus - safety_penalty + position_factor * 0.1

    if action_delta < 0 and dist_change >= 0:
        q_value -= 0.1
    if action_delta > 0 and dist_change >= 0:
        q_value -= 0.1
    if action_delta < 0 and dist_change < 0:
        q_value += 0.2
    if action_delta > 0 and dist_change < 0:
        q_value += 0.2

    return max(0.0, min(1.0, q_value))