import collections
import math

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value Q(s, a) for a given state-action pair in an 8x8 Frozen Lake environment.
    
    The Q-value is approximated as the discounted reward for the shortest safe path from the 
    next state to the goal, scaled by a safety factor based on the slip risk of the action taken.
    """
    def get_coords(grid_str, char):
        lines = grid_str.splitlines()
        for r, line in enumerate(lines):
            for c, char_at in enumerate(line):
                if char_at == char:
                    return r, c
        return None

    # Extract grid and positions
    state_lines = state.splitlines()
    rows = len(state_lines)
    cols = len(state_lines[0]) if rows > 0 else 0
    
    agent_pos = get_coords(state, '@')
    next_agent_pos = get_coords(next_state, '@')
    goal_pos = get_coords(state, 'G')

    if agent_pos is None or next_agent_pos is None or goal_pos is None:
        return 0.0, {"coord_error": 0.0}

    # Resulting cell type
    nr, nc = next_agent_pos
    cell_type = state_lines[nr][nc]

    # Immediate outcomes
    if cell_type == 'G':
        return 1.0, {"goal_reached": 1.0}
    
    if cell_type == 'H':
        # Immediate penalty for falling into a hole
        return -1.0, {"hole_penalty": -1.0}

    # BFS to find the shortest path from all cells to the goal, avoiding holes
    dist = {goal_pos: 0}
    queue = collections.deque([goal_pos])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            vr, vc = r + dr, c + dc
            if 0 <= vr < rows and 0 <= vc < cols:
                if state_lines[vr][vc] != 'H' and (vr, vc) not in dist:
                    dist[(vr, vc)] = dist[(r, c)] + 1
                    queue.append((vr, vc))

    # Parameters for approximation
    gamma = 0.95
    slip_prob = 0.2 # probability of slipping into a perpendicular direction

    # Calculate slip risk for the specific action taken in state s
    action_map = {'left': (0, -1), 'down': (1, 0), 'right': (0, 1), 'up': (-1, 0)}
    # If action is not in map (though should be), default to no movement
    move = action_map.get(action, (0, 0))
    dr, dc = move
    # Perpendicular directions relative to the intended move (dr, dc)
    perps = [(-dc, dr), (dc, -dr)]
    
    risk = 0.0
    for pr, pc in perps:
        ar, ac = agent_pos[0] + pr, agent_pos[1] + pc
        if 0 <= ar < rows and 0 <= ac < cols:
            if state_lines[ar][ac] == 'H':
                risk += slip_prob

    # Estimate V(s') using shortest distance to goal
    # If the next state cannot reach the goal, value is 0
    d = dist.get(next_agent_pos, float('inf'))
    if d == float('inf'):
        return 0.0, {"no_path": 0.0}
    
    # Q(s, a) ≈ gamma^d * (1 - risk)
    # gamma^d represents the discounted reward for reaching the goal in d more steps.
    dist_val = gamma ** d
    total_value = dist_val * (1.0 - risk)
    
    return total_value, {
        "dist_val": float(dist_val),
        "risk_penalty": float(total_value - dist_val),
        "dist_to_goal": float(d)
    }